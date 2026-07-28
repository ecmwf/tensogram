! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> examples/fortran/frame_walker.f90
!>
!> Look INSIDE a message without decoding it — the structural view the other
!> readers never expose:
!>   * tensogram_message_header_read — the 24-byte preamble as a typed record:
!>     wire version, total length, and eight named flags that say whether the
!>     message is random-access (metadata / index / hashes in the HEADER) or
!>     streaming (in the FOOTER).
!>   * tensogram_frames + %next — a lazy walk over every frame, one
!>     tensogram_frame value at a time: type, version, flags, offset, length,
!>     inline-hash flag and the payload bytes (COPIED out of the message, so
!>     they outlive both the iterator and the buffer).
!>   * tensogram_encode_with_options — aggregate_hash = BOTH, whose effect is
!>     then VERIFIED with the walker: a hash frame in the header AND one in
!>     the footer.
!>
!> Mirrors the C++ `examples/cpp/26_frame_walker.cpp` and the Python
!> `examples/python/21_frames_and_message_header.py`.
program frame_walker
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'frame_walker_stream.tgm'
   integer, parameter :: NI = 4, NJ = 3
   real(c_float)                  :: field(NI, NJ)
   integer(c_int8_t), allocatable :: wire(:), data(:)
   integer(c_size_t)              :: lens(1)
   type(tensogram_buffer) :: buf
   integer(c_int) :: err
   integer :: i, j

   do j = 1, NJ; do i = 1, NI; field(i, j) = real(i * 10 + j, c_float); end do; end do

   ! ── 1. A buffered (random-access) message ───────────────────────────────
   print '(a)', '=== buffered message ==='
   call tensogram_encode(field, buf, err)
   call tensogram_check(err, 'encode')
   call buf%as_array(wire)
   call show_header(wire)
   call show_frames(wire)

   ! ── 2. A streaming message: metadata / index / hash move to the footer ──
   print '(a)', ''
   print '(a)', '=== streaming message ==='
   call write_stream(path, field)
   call read_first_message(path, wire)
   call show_header(wire)
   call show_frames(wire)
   call delete_file(path)

   ! ── 3. aggregate_hash = BOTH, verified with the walker ──────────────────
   print '(a)', ''
   print '(a)', '=== encode_with_options(aggregate_hash = BOTH) ==='
   data    = transfer(field, [0_c_int8_t], NI * NJ * 4)
   lens(1) = int(NI * NJ * 4, c_size_t)
   call tensogram_encode_with_options(descriptors_json(NI * NJ), data, lens, buf, err, &
                                      aggregate_hash=TGM_AGGREGATE_HASH_POLICY_BOTH)
   call tensogram_check(err, 'encode_with_options')
   call buf%as_array(wire)
   print '(a,l1,a,l1)', 'header hash frame: ', has_frame(wire, TGM_FRAME_TYPE_HEADER_HASH), &
                        '   footer hash frame: ', has_frame(wire, TGM_FRAME_TYPE_FOOTER_HASH)
   call show_frames(wire)

contains

   !> Print the message envelope: version, byte count and the eight flags.
   subroutine show_header(msg)
      integer(c_int8_t), intent(in) :: msg(:)
      type(tensogram_message_header) :: hdr
      integer(c_int) :: err
      call tensogram_message_header_read(msg, hdr, err)
      call tensogram_check(err, 'message_header_read')
      print '(a,i0,a,i0,a,i0,a)', 'preamble: version=', hdr%version(),         &
         '  total_length=', hdr%total_length(), '  (buffer holds ', size(msg), ' bytes)'
      print '(a,l1,a,l1,a,l1,a,l1)',                                           &
         '  metadata: header=', hdr%has_header_metadata(),                     &
         ' footer=', hdr%has_footer_metadata(),                                &
         '   index: header=', hdr%has_header_index(),                          &
         ' footer=', hdr%has_footer_index()
      print '(a,l1,a,l1,a,l1,a,l1)',                                           &
         '  hashes:   header=', hdr%has_header_hashes(),                       &
         ' footer=', hdr%has_footer_hashes(),                                  &
         '   per-frame=', hdr%has_hashes_present(),                            &
         '   preceder metadata=', hdr%has_preceder_metadata()
   end subroutine show_header

   !> Walk every frame and print one line each. A .false. `found` ends the
   !> loop; `err` then tells a clean end from a malformed frame chain.
   subroutine show_frames(msg)
      integer(c_int8_t), intent(in) :: msg(:)
      type(tensogram_frame_iterator) :: it
      type(tensogram_frame)          :: fr
      integer(c_int8_t), allocatable :: payload(:)
      integer(c_int) :: err
      integer :: n
      logical :: found
      call tensogram_frames(msg, it, err)
      call tensogram_check(err, 'frames')
      n = 0
      do
         call it%next(fr, found, err)
         if (.not. found) exit
         n = n + 1
         payload = fr%payload()          ! an independent copy of the bytes
         print '(a,i0,a,a,a,i0,a,i0,a,i0,a,l1)',                              &
            '  frame ', n, ': ', frame_name(fr%frame_type()),                 &
            '  offset=', fr%offset(), ' length=', fr%length(),                &
            ' payload=', size(payload), ' hashed=', fr%has_hash()
      end do
      if (err /= TGM_ERROR_OK) then
         ! Malformed chain: the frames before the damage were still yielded.
         print '(a,a)', '  !! malformed frame chain: ', tensogram_last_error()
      else
         print '(a,i0,a)', '  (', n, ' frames, walk ended cleanly)'
      end if
      call it%free()
   end subroutine show_frames

   !> .true. when the message contains at least one frame of type `ftype`.
   logical function has_frame(msg, ftype)
      integer(c_int8_t), intent(in) :: msg(:)
      integer(c_int),    intent(in) :: ftype
      type(tensogram_frame_iterator) :: it
      type(tensogram_frame)          :: fr
      integer(c_int) :: err
      logical :: found
      has_frame = .false.
      call tensogram_frames(msg, it, err)
      call tensogram_check(err, 'frames')
      do
         call it%next(fr, found, err)
         if (.not. found) exit
         if (fr%frame_type() == ftype) has_frame = .true.
      end do
      call it%free()
   end function has_frame

   !> Human-readable label for a TGM_FRAME_TYPE_* code (4 is reserved).
   function frame_name(ftype) result(name)
      integer(c_int), intent(in) :: ftype
      character(len=17) :: name
      select case (ftype)
      case (TGM_FRAME_TYPE_HEADER_METADATA);   name = 'header metadata  '
      case (TGM_FRAME_TYPE_HEADER_INDEX);      name = 'header index     '
      case (TGM_FRAME_TYPE_HEADER_HASH);       name = 'header hash      '
      case (TGM_FRAME_TYPE_FOOTER_HASH);       name = 'footer hash      '
      case (TGM_FRAME_TYPE_FOOTER_INDEX);      name = 'footer index     '
      case (TGM_FRAME_TYPE_FOOTER_METADATA);   name = 'footer metadata  '
      case (TGM_FRAME_TYPE_PRECEDER_METADATA); name = 'preceder metadata'
      case (TGM_FRAME_TYPE_NTENSOR);           name = 'ntensor (data)   '
      case default;                            name = 'unknown          '
      end select
   end function frame_name

   !> Stream one object to `path` (metadata / index / hash land in the footer).
   subroutine write_stream(file_path, a)
      character(len=*), intent(in) :: file_path
      real(c_float),    intent(in) :: a(:,:)
      type(tensogram_streaming_encoder) :: enc
      integer(c_int) :: err
      call tensogram_streaming_encoder_create(file_path, enc, err)
      call tensogram_check(err, 'streaming create')
      call tensogram_streaming_encoder_write(enc, a, err)
      call tensogram_check(err, 'streaming write')
      call tensogram_streaming_encoder_finish(enc, err)
      call tensogram_check(err, 'streaming finish')
      call enc%free()
   end subroutine write_stream

   !> Read message 1 of `path` back as raw wire bytes.
   subroutine read_first_message(file_path, msg)
      character(len=*),               intent(in)  :: file_path
      integer(c_int8_t), allocatable, intent(out) :: msg(:)
      type(tensogram_file)   :: f
      type(tensogram_buffer) :: raw
      integer(c_int) :: err
      call tensogram_file_open(file_path, f, err)
      call tensogram_check(err, 'file_open')
      call tensogram_file_read_message(f, 1, raw, err)
      call tensogram_check(err, 'read_message')
      call raw%as_array(msg)
      call f%close()
   end subroutine read_first_message

   subroutine delete_file(file_path)
      character(len=*), intent(in) :: file_path
      integer :: unit, ios
      open(newunit=unit, file=file_path, status='old', iostat=ios)
      if (ios == 0) close(unit, status='delete')
   end subroutine delete_file

   !> A one-object float32 descriptor envelope for `n` elements.
   function descriptors_json(n) result(s)
      integer, intent(in) :: n
      character(len=:), allocatable :: s
      character(len=32) :: ns
      write (ns, '(i0)') n
      s = '{"descriptors":[{"type":"ndarray","ndim":1,"shape":[' // trim(ns) // &
          '],"strides":[1],"dtype":"float32","byte_order":"' // host_bo() //    &
          '","encoding":"none","filter":"none","compression":"none"}]}'
   end function descriptors_json

   !> Host byte order as the wire descriptor spells it ("little" / "big").
   function host_bo() result(bo)
      character(len=:), allocatable :: bo
      integer(c_int8_t) :: probe(4)
      probe = transfer(1_c_int32_t, 0_c_int8_t, 4)
      if (probe(1) == 1_c_int8_t) then
         bo = 'little'
      else
         bo = 'big'
      end if
   end function host_bo

end program frame_walker

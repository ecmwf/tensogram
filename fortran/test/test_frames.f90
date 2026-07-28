! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Deferred Wave-B, component A3 — the lazy frame walker and the typed message
!> header:
!>   * TGM_FRAME_TYPE_*            (tgm_frame_type, note the reserved gap at 4)
!>   * tensogram_frame             (TgmFrame, as a copyable VALUE type)
!>   * tensogram_frame_iterator    (tgm_frame_iter_create / _next / _free)
!>   * tensogram_message_header_read (tgm_message_header)
!>
!> The structural assertions are statements about the wire format
!> (plans/WIRE_FORMAT.md): a 24-byte preamble, 16-byte frame headers, a 20-byte
!> footer on NTENSOR frames and 12 bytes on every other type, frames that tile
!> the message without overlapping, and a 24-byte postamble that is NOT a frame.
program test_frames
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   ! Wire-format structural constants (plans/WIRE_FORMAT.md §§2, 3, 7).
   integer, parameter :: PREAMBLE     = 24
   integer, parameter :: POSTAMBLE    = 24
   integer, parameter :: FRAME_HEADER = 16
   integer, parameter :: FOOTER_ANY   = 12   ! [hash][ENDF]
   integer, parameter :: FOOTER_NTEN  = 20   ! [cbor_offset][hash][ENDF]
   integer, parameter :: HASH_PRESENT = 2    ! §2.5 frame flag bit 1
   integer, parameter :: MAXF         = 32   ! plenty for these fixtures

   integer :: npass

   npass = 0

   call frame_type_constants()
   call default_values()
   call buffered_frame_sequence()
   call frame_geometry()
   call payload_and_hash_flags()
   call payload_outlives_the_iterator()
   call streaming_frame_sequence()
   call end_versus_malformed()
   call bad_buffers_rejected()
   call message_header_buffered()
   call message_header_streaming()

   print '(a,i0,a)', 'test_frames: PASS (', npass, ' checks)'

contains

   ! ---- The mirrored wire numbers, gap at 4 included -----------------------
   subroutine frame_type_constants()
      call assert(TGM_FRAME_TYPE_HEADER_METADATA   == 1, 'frame type HEADER_METADATA = 1')
      call assert(TGM_FRAME_TYPE_HEADER_INDEX      == 2, 'frame type HEADER_INDEX = 2')
      call assert(TGM_FRAME_TYPE_HEADER_HASH       == 3, 'frame type HEADER_HASH = 3')
      ! 4 is reserved (the obsolete v2 data-object layout) — no parameter.
      call assert(TGM_FRAME_TYPE_FOOTER_HASH       == 5, 'frame type FOOTER_HASH = 5')
      call assert(TGM_FRAME_TYPE_FOOTER_INDEX      == 6, 'frame type FOOTER_INDEX = 6')
      call assert(TGM_FRAME_TYPE_FOOTER_METADATA   == 7, 'frame type FOOTER_METADATA = 7')
      call assert(TGM_FRAME_TYPE_PRECEDER_METADATA == 8, 'frame type PRECEDER_METADATA = 8')
      call assert(TGM_FRAME_TYPE_NTENSOR           == 9, 'frame type NTENSOR = 9')
   end subroutine frame_type_constants

   ! ---- A default-initialised frame / header is inert, never a trap --------
   subroutine default_values()
      type(tensogram_frame)          :: fr
      type(tensogram_message_header) :: hdr
      call assert(fr%frame_type() == 0, 'default frame: no type')
      call assert(fr%version() == 0 .and. fr%flags() == 0, 'default frame: no version / flags')
      call assert(fr%offset() == 0_c_size_t .and. fr%length() == 0_c_size_t, 'default frame: no span')
      call assert(.not. fr%has_hash(), 'default frame: no inline hash')
      ! %payload() must answer with an empty array rather than an unallocated
      ! one — the component is only allocated by a successful %next.
      call assert(size(fr%payload()) == 0, 'default frame: empty payload')
      call assert(hdr%version() == 0, 'default header: no version')
      call assert(hdr%total_length() == 0_c_int64_t, 'default header: no length')
      call assert(.not. hdr%has_header_metadata() .and. .not. hdr%has_footer_metadata() .and. &
                  .not. hdr%has_header_index() .and. .not. hdr%has_footer_index() .and.       &
                  .not. hdr%has_header_hashes() .and. .not. hdr%has_footer_hashes() .and.     &
                  .not. hdr%has_preceder_metadata() .and. .not. hdr%has_hashes_present(),     &
                  'default header: every flag clear')
   end subroutine default_values

   ! ---- The expected frame-type sequence of a buffered message -------------
   subroutine buffered_frame_sequence()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int) :: err
      integer :: n

      call buffered(2, .true., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'buffered walk: clean end')
      call assert(n == 5, 'buffered 2-object message: 5 frames')
      call assert(frames(1)%frame_type() == TGM_FRAME_TYPE_HEADER_METADATA, 'buffered[1] header metadata')
      call assert(frames(2)%frame_type() == TGM_FRAME_TYPE_HEADER_INDEX,    'buffered[2] header index')
      call assert(frames(3)%frame_type() == TGM_FRAME_TYPE_HEADER_HASH,     'buffered[3] header hash')
      call assert(frames(4)%frame_type() == TGM_FRAME_TYPE_NTENSOR,         'buffered[4] ntensor')
      call assert(frames(5)%frame_type() == TGM_FRAME_TYPE_NTENSOR,         'buffered[5] ntensor')

      ! An unhashed message carries no aggregate hash frame at all.
      call buffered(1, .false., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'unhashed walk: clean end')
      call assert(n == 3, 'unhashed 1-object message: 3 frames')
      call assert(frames(1)%frame_type() == TGM_FRAME_TYPE_HEADER_METADATA, 'unhashed[1] header metadata')
      call assert(frames(2)%frame_type() == TGM_FRAME_TYPE_HEADER_INDEX,    'unhashed[2] header index')
      call assert(frames(3)%frame_type() == TGM_FRAME_TYPE_NTENSOR,         'unhashed[3] ntensor')
   end subroutine buffered_frame_sequence

   ! ---- Offsets / lengths: in bounds, ordered, non-overlapping -------------
   subroutine frame_geometry()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int)    :: err
      integer(c_size_t) :: prev_end, first, last
      integer :: n, k

      call buffered(2, .true., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'geometry: clean end')
      call assert(n > 0, 'geometry: some frames')

      ! %offset() is a 1-BASED Fortran index (the tensogram_scan convention),
      ! so the first frame starts just past the 24-byte preamble.
      prev_end = int(PREAMBLE, c_size_t)          ! last byte of the preamble
      do k = 1, n
         first = frames(k)%offset()
         last  = first + frames(k)%length() - 1_c_size_t
         call assert(first > prev_end, 'geometry: frames do not overlap')
         call assert(last <= size(wire, kind=c_size_t), 'geometry: frame stays in bounds')
         call assert(frames(k)%length() >= int(FRAME_HEADER, c_size_t), 'geometry: frame spans its header')
         ! The whole-frame span always ends on the ENDF sentinel.
         call assert(tag_at(wire, last - 3_c_size_t) == 'ENDF', 'geometry: length spans the frame footer')
         prev_end = last
      end do
      call assert(prev_end <= size(wire, kind=c_size_t) - int(POSTAMBLE, c_size_t), &
                  'geometry: the postamble is not a frame')
      call assert(frames(1)%offset() == int(PREAMBLE + 1, c_size_t), 'geometry: first frame follows the preamble')
   end subroutine frame_geometry

   ! ---- Payload span, per-frame hash flag, version -------------------------
   subroutine payload_and_hash_flags()
      integer(c_int8_t), allocatable :: wire(:), payload(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int)    :: err
      integer(c_size_t) :: want, start
      integer :: n, k, footer

      call buffered(2, .true., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'payload: clean end')
      do k = 1, n
         if (frames(k)%frame_type() == TGM_FRAME_TYPE_NTENSOR) then
            footer = FOOTER_NTEN
         else
            footer = FOOTER_ANY
         end if
         payload = frames(k)%payload()
         want = frames(k)%length() - int(FRAME_HEADER + footer, c_size_t)
         call assert(size(payload, kind=c_size_t) == want, 'payload: length - header - footer')
         ! The copy holds exactly the bytes between header and footer.
         start = frames(k)%offset() + int(FRAME_HEADER, c_size_t)
         call assert(bytes_eq(payload, wire(start:start + want - 1_c_size_t)), 'payload: copied from the message')
         call assert(frames(k)%has_hash(), 'hashed message: every frame has HASH_PRESENT')
         call assert(iand(frames(k)%flags(), HASH_PRESENT) /= 0, 'hashed message: flag bit 1 agrees')
         call assert(frames(k)%version() == 1, 'frame version 1')
      end do

      call buffered(1, .false., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'unhashed payload: clean end')
      do k = 1, n
         call assert(.not. frames(k)%has_hash(), 'unhashed message: no HASH_PRESENT')
         call assert(iand(frames(k)%flags(), HASH_PRESENT) == 0, 'unhashed message: flag bit 1 clear')
      end do
   end subroutine payload_and_hash_flags

   ! ---- The payload copy survives the iterator (it is a copy, not a view) --
   subroutine payload_outlives_the_iterator()
      integer(c_int8_t), allocatable :: wire(:), payload(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int) :: err
      integer :: n

      call buffered(1, .true., wire)
      call walk(wire, frames, n, err)          ! walk() frees the iterator
      call assert(n > 0, 'payload copy: some frames')
      payload = frames(1)%payload()
      call assert(size(payload) > 0, 'payload copy: readable after the iterator is gone')
      call assert(bytes_eq(payload, wire(frames(1)%offset() + int(FRAME_HEADER, c_size_t): &
                                         frames(1)%offset() + int(FRAME_HEADER, c_size_t) + &
                                         size(payload, kind=c_size_t) - 1_c_size_t)), &
                  'payload copy: still equals the message bytes')
      ! A frame is a plain value: assigning it copies the payload too.
      frames(2) = frames(1)
      call assert(bytes_eq(frames(2)%payload(), payload), 'frame value type: copy-assignable')
   end subroutine payload_outlives_the_iterator

   ! ---- A streaming message puts metadata / index / hash in the footer -----
   subroutine streaming_frame_sequence()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int) :: err
      integer :: n

      call streamed(wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'streaming walk: clean end')
      call assert(n == 5, 'streamed message: 5 frames')
      call assert(frames(1)%frame_type() == TGM_FRAME_TYPE_HEADER_METADATA, 'streamed[1] header metadata')
      call assert(frames(2)%frame_type() == TGM_FRAME_TYPE_NTENSOR,         'streamed[2] ntensor')
      call assert(frames(3)%frame_type() == TGM_FRAME_TYPE_FOOTER_METADATA, 'streamed[3] footer metadata')
      call assert(frames(4)%frame_type() == TGM_FRAME_TYPE_FOOTER_HASH,     'streamed[4] footer hash')
      call assert(frames(5)%frame_type() == TGM_FRAME_TYPE_FOOTER_INDEX,    'streamed[5] footer index')
   end subroutine streaming_frame_sequence

   ! ---- End vs malformed: both stop the walk, only one sets `err` ----------
   subroutine end_versus_malformed()
      integer(c_int8_t), allocatable :: wire(:), cut(:)
      type(tensogram_frame) :: frames(MAXF), fr
      type(tensogram_frame_iterator) :: it
      integer(c_int) :: err, e2
      integer :: n, yielded
      integer(c_size_t) :: chop
      logical :: found

      call buffered(2, .true., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'clean end: err stays OK')

      ! Advancing an exhausted cursor keeps reporting a clean end.
      call tensogram_frames(wire, it, err)
      call assert(err == TGM_ERROR_OK, 'clean end: iterator created')
      do
         call it%next(fr, found, e2)
         if (.not. found) exit
      end do
      call assert(e2 == TGM_ERROR_OK, 'clean end: no error code')
      call it%next(fr, found, e2)
      call assert(.not. found .and. e2 == TGM_ERROR_OK, 'clean end: still ended, still no error')
      call it%free()

      ! Cut INSIDE the last frame: the preamble still parses, the chain does not.
      chop = frames(n)%offset() + 8_c_size_t
      cut  = wire(1:chop)
      call tensogram_frames(cut, it, err)
      call assert(err == TGM_ERROR_OK, 'truncated: preamble still parses')
      yielded = 0
      do
         call it%next(fr, found, e2)
         if (.not. found) exit
         yielded = yielded + 1
      end do
      call assert(yielded > 0, 'truncated: the intact frames are still yielded')
      call assert(e2 /= TGM_ERROR_OK, 'truncated: malformed chain sets err')
      call assert(len(tensogram_last_error()) > 0, 'truncated: a reason is recorded')
      call it%free()

      ! next() on a freed / never-created iterator is an argument error.
      call it%next(fr, found, e2)
      call assert(.not. found .and. e2 == TGM_ERROR_INVALID_ARG, 'next on a dead iterator -> INVALID_ARG')

      ! `err` is optional: without it the walk still runs, it just cannot tell
      ! a clean end from a malformed chain (both simply stop the loop).
      call tensogram_frames(wire, it, err)
      call assert(err == TGM_ERROR_OK, 'next without err: iterator created')
      yielded = 0
      do
         call it%next(fr, found)
         if (.not. found) exit
         yielded = yielded + 1
      end do
      call assert(yielded == n, 'next without err: same frames yielded')
      call it%free()
   end subroutine end_versus_malformed

   ! ---- Buffers that are not messages are rejected at creation -------------
   subroutine bad_buffers_rejected()
      integer(c_int8_t), allocatable :: wire(:), junk(:), empty(:)
      type(tensogram_frame_iterator) :: it
      type(tensogram_message_header) :: hdr
      integer(c_int) :: err
      integer :: k

      allocate(junk(40))
      do k = 1, 40
         junk(k) = int(mod(k, 97), c_int8_t)
      end do
      call tensogram_frames(junk, it, err)
      call assert(err /= TGM_ERROR_OK, 'junk buffer: no iterator')
      call tensogram_message_header_read(junk, hdr, err)
      call assert(err /= TGM_ERROR_OK, 'junk buffer: no header')

      allocate(empty(0))
      call tensogram_frames(empty, it, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'empty buffer -> INVALID_ARG')
      call tensogram_message_header_read(empty, hdr, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'empty buffer header -> INVALID_ARG')

      call buffered(1, .true., wire)
      call tensogram_frames(wire(1:8), it, err)
      call assert(err /= TGM_ERROR_OK, 'truncated preamble: no iterator')
      call tensogram_message_header_read(wire(1:8), hdr, err)
      call assert(err /= TGM_ERROR_OK, 'truncated preamble: no header')
   end subroutine bad_buffers_rejected

   ! ---- Header flags match the frames of a BUFFERED message exactly --------
   subroutine message_header_buffered()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      type(tensogram_message_header) :: hdr
      integer(c_int) :: err
      integer :: n

      call buffered(2, .true., wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'header: walk ok')
      call tensogram_message_header_read(wire, hdr, err)
      call assert(err == TGM_ERROR_OK, 'header: read ok')

      call assert(hdr%version() == TGM_WIRE_VERSION, 'header: wire version')
      call assert(hdr%total_length() == size(wire, kind=c_int64_t), 'header: total_length is the message size')

      call assert(hdr%has_header_metadata() .eqv. seen(frames, n, TGM_FRAME_TYPE_HEADER_METADATA), &
                  'header: has_header_metadata matches the frames')
      call assert(hdr%has_footer_metadata() .eqv. seen(frames, n, TGM_FRAME_TYPE_FOOTER_METADATA), &
                  'header: has_footer_metadata matches the frames')
      call assert(hdr%has_header_index() .eqv. seen(frames, n, TGM_FRAME_TYPE_HEADER_INDEX), &
                  'header: has_header_index matches the frames')
      call assert(hdr%has_footer_index() .eqv. seen(frames, n, TGM_FRAME_TYPE_FOOTER_INDEX), &
                  'header: has_footer_index matches the frames')
      call assert(hdr%has_header_hashes() .eqv. seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), &
                  'header: has_header_hashes matches the frames')
      call assert(hdr%has_footer_hashes() .eqv. seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), &
                  'header: has_footer_hashes matches the frames')
      call assert(hdr%has_preceder_metadata() .eqv. seen(frames, n, TGM_FRAME_TYPE_PRECEDER_METADATA), &
                  'header: has_preceder_metadata matches the frames')
      call assert(hdr%has_hashes_present(), 'header: the default encode hashes every frame')

      ! Random access: metadata and index live in the header, not the footer.
      call assert(hdr%has_header_metadata() .and. .not. hdr%has_footer_metadata(), 'header: random-access metadata')
      call assert(hdr%has_header_index() .and. .not. hdr%has_footer_index(), 'header: random-access index')

      ! An unhashed message advertises neither aggregate hashes nor per-frame ones.
      call buffered(1, .false., wire)
      call tensogram_message_header_read(wire, hdr, err)
      call assert(err == TGM_ERROR_OK, 'unhashed header: read ok')
      call assert(.not. hdr%has_header_hashes(), 'unhashed header: no header hash frame')
      call assert(.not. hdr%has_footer_hashes(), 'unhashed header: no footer hash frame')
      call assert(.not. hdr%has_hashes_present(), 'unhashed header: no per-frame hashes')
   end subroutine message_header_buffered

   ! ---- Streaming: only "frame present => flag set" is guaranteed ----------
   subroutine message_header_streaming()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      type(tensogram_message_header) :: hdr
      integer(c_int) :: err
      integer :: n

      call streamed(wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'streaming header: walk ok')
      call tensogram_message_header_read(wire, hdr, err)
      call assert(err == TGM_ERROR_OK, 'streaming header: read ok')

      call assert(hdr%version() == TGM_WIRE_VERSION, 'streaming header: wire version')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_HEADER_METADATA), hdr%has_header_metadata()), &
                  'streaming header: header metadata frame => flag')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_FOOTER_METADATA), hdr%has_footer_metadata()), &
                  'streaming header: footer metadata frame => flag')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_HEADER_INDEX), hdr%has_header_index()), &
                  'streaming header: header index frame => flag')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_FOOTER_INDEX), hdr%has_footer_index()), &
                  'streaming header: footer index frame => flag')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), hdr%has_header_hashes()), &
                  'streaming header: header hash frame => flag')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), hdr%has_footer_hashes()), &
                  'streaming header: footer hash frame => flag')
      call assert(implies(seen(frames, n, TGM_FRAME_TYPE_PRECEDER_METADATA), hdr%has_preceder_metadata()), &
                  'streaming header: preceder frame => flag')
      ! Streaming mode: metadata + index land in the footer, and the
      ! non-seeking sink never back-fills the preamble's total_length.
      call assert(hdr%has_footer_metadata(), 'streaming header: footer metadata')
      call assert(hdr%has_footer_index(), 'streaming header: footer index')
      call assert(hdr%total_length() == 0_c_int64_t, 'streaming header: total_length never back-filled')
   end subroutine message_header_streaming

   ! ---- helpers ------------------------------------------------------------

   !> Walk `wire` to exhaustion, collecting the frames. `err` is the walk's
   !> final status: OK for a clean end, non-OK for a malformed chain.
   subroutine walk(wire, frames, n, err)
      integer(c_int8_t),     intent(in)  :: wire(:)
      type(tensogram_frame), intent(out) :: frames(:)
      integer,               intent(out) :: n
      integer(c_int),        intent(out) :: err
      type(tensogram_frame_iterator) :: it
      type(tensogram_frame)          :: fr
      logical :: found
      n = 0
      call tensogram_frames(wire, it, err)
      if (err /= TGM_ERROR_OK) return
      do
         call it%next(fr, found, err)
         if (.not. found) exit
         n = n + 1
         call assert(n <= size(frames), 'walk: frame buffer overflow')
         frames(n) = fr
      end do
      call it%free()
   end subroutine walk

   !> .true. when a frame of type `ftype` was yielded.
   logical function seen(frames, n, ftype)
      type(tensogram_frame), intent(in) :: frames(:)
      integer,               intent(in) :: n
      integer(c_int),        intent(in) :: ftype
      integer :: k
      seen = .false.
      do k = 1, n
         if (frames(k)%frame_type() == ftype) seen = .true.
      end do
   end function seen

   pure logical function implies(p, q)
      logical, intent(in) :: p, q
      implies = (.not. p) .or. q
   end function implies

   !> The four ASCII bytes of `wire` starting at the 1-based index `at`.
   function tag_at(wire, at) result(s)
      integer(c_int8_t), intent(in) :: wire(:)
      integer(c_size_t), intent(in) :: at
      character(len=4) :: s
      integer :: k
      do k = 1, 4
         s(k:k) = achar(iand(int(wire(at + int(k - 1, c_size_t))), 255))
      end do
   end function tag_at

   !> A buffered (random-access) message of `nobj` float32 objects, hashed or
   !> not: metadata / index / hash frames all live in the header.
   subroutine buffered(nobj, hashed, wire)
      integer, intent(in)  :: nobj
      logical, intent(in)  :: hashed
      integer(c_int8_t), allocatable, intent(out) :: wire(:)
      real(c_float)                  :: vals(4)
      integer(c_int8_t), allocatable :: data(:)
      integer(c_size_t), allocatable :: lens(:)
      character(len=:),  allocatable :: json
      type(tensogram_buffer) :: buf
      integer(c_int) :: err
      integer :: k
      vals = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
      allocate(lens(nobj), data(0))
      json = '{"descriptors":['
      do k = 1, nobj
         if (k > 1) json = json // ','
         json = json // desc1d(4)
         data = [data, transfer(vals, [0_c_int8_t], 4 * 4)]
         lens(k) = int(4 * 4, c_size_t)
      end do
      json = json // ']}'
      if (hashed) then
         call tensogram_encode_pre_encoded(json, data, lens, buf, err)
      else
         call tensogram_encode_pre_encoded(json, data, lens, buf, err, hash='')
      end if
      call assert(err == TGM_ERROR_OK, 'fixture: buffered encode')
      call buf%as_array(wire)
   end subroutine buffered

   !> A streaming message (one object): metadata / index / hashes land in the
   !> FOOTER and total_length is never back-filled.
   subroutine streamed(wire)
      integer(c_int8_t), allocatable, intent(out) :: wire(:)
      character(len=*), parameter :: path = 'test_frames_stream.tgm'
      real(c_float) :: vals(4)
      type(tensogram_streaming_encoder) :: enc
      type(tensogram_file)   :: f
      type(tensogram_buffer) :: buf
      integer(c_int) :: err
      integer :: unit, ios
      vals = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
      call tensogram_streaming_encoder_create(path, enc, err)
      call assert(err == TGM_ERROR_OK, 'fixture: stream create')
      call tensogram_streaming_encoder_write(enc, vals, err)
      call assert(err == TGM_ERROR_OK, 'fixture: stream write')
      call tensogram_streaming_encoder_finish(enc, err)
      call assert(err == TGM_ERROR_OK, 'fixture: stream finish')
      call enc%free()
      call tensogram_file_open(path, f, err)
      call assert(err == TGM_ERROR_OK, 'fixture: stream reopen')
      call tensogram_file_read_message(f, 1, buf, err)
      call assert(err == TGM_ERROR_OK, 'fixture: stream read')
      call buf%as_array(wire)
      call f%close()
      open(newunit=unit, file=path, status='old', iostat=ios)
      if (ios == 0) close(unit, status='delete')
   end subroutine streamed

   !> A bare 1-D float32 descriptor (no encoding pipeline) for `n` elements.
   function desc1d(n) result(s)
      integer, intent(in) :: n
      character(len=:), allocatable :: s
      character(len=32) :: ns
      write (ns, '(i0)') n
      s = '{"type":"ndarray","ndim":1,"shape":[' // trim(ns) //          &
          '],"strides":[4],"dtype":"float32","byte_order":"' // host_bo() // &
          '","encoding":"none","filter":"none","compression":"none"}'
   end function desc1d

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

   logical function bytes_eq(x, y)
      integer(c_int8_t), intent(in) :: x(:), y(:)
      bytes_eq = size(x) == size(y)
      if (bytes_eq) bytes_eq = all(x == y)
   end function bytes_eq

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_frames: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

end program test_frames

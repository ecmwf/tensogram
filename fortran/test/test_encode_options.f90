! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Deferred Wave-B, component C — the full encode-side option set
!> (TgmEncodeOptions), which supersedes the mask-only options by also carrying
!> the hash algorithm, the aggregate-hash PLACEMENT and the codec BACKEND:
!>   * tensogram_encode_with_options                    (tgm_encode_with_encode_options)
!>   * tensogram_file_append_with_options               (tgm_file_append_with_encode_options)
!>   * tensogram_streaming_encoder_create_with_options  (tgm_streaming_encoder_create_with_encode_options)
!>
!> The aggregate-hash assertions are a CLOSED LOOP: what the option asked for
!> is verified by walking the resulting message with this binding's own frame
!> walker (component A3) and by reading its message header.
program test_encode_options
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic
   use tensogram
   implicit none

   integer, parameter :: MAXF = 32
   integer :: npass

   npass = 0

   call default_options_match_plain_encode()
   call aggregate_hash_placements()
   call aggregate_hash_both_is_visible_in_the_frames()
   call hash_algorithm_knob()
   call compression_backend_knob()
   call mask_options_still_reachable()
   call file_append_with_options_cases()
   call streaming_create_with_options_cases()

   print '(a,i0,a)', 'test_encode_options: PASS (', npass, ' checks)'

contains

   ! ---- No options at all == the library defaults --------------------------
   subroutine default_options_match_plain_encode()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int) :: err
      integer :: n
      call encode_opts(wire, err)
      call assert(err == TGM_ERROR_OK, 'defaults: encode OK')
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'defaults: walk OK')
      ! Buffered + default hashing => the aggregate hash frame goes in the header.
      call assert(seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'defaults: header hash frame')
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'defaults: no footer hash frame')
      call assert(values_round_trip(wire), 'defaults: values round-trip')
   end subroutine default_options_match_plain_encode

   ! ---- AUTO / NONE / HEADER / FOOTER placements ---------------------------
   subroutine aggregate_hash_placements()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int) :: err
      integer :: n, k

      call encode_opts(wire, err, aggregate_hash=TGM_AGGREGATE_HASH_POLICY_AUTO)
      call assert(err == TGM_ERROR_OK, 'AUTO: encode OK')
      call walk(wire, frames, n, err)
      call assert(seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'AUTO: buffered => header hash')
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'AUTO: no footer hash')

      call encode_opts(wire, err, aggregate_hash=TGM_AGGREGATE_HASH_POLICY_NONE)
      call assert(err == TGM_ERROR_OK, 'NONE: encode OK')
      call walk(wire, frames, n, err)
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'NONE: no header hash frame')
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'NONE: no footer hash frame')
      ! NONE drops only the AGGREGATE frame — the per-frame inline slots stay.
      do k = 1, n
         call assert(frames(k)%has_hash(), 'NONE: per-frame inline hashes are unaffected')
      end do

      call encode_opts(wire, err, aggregate_hash=TGM_AGGREGATE_HASH_POLICY_HEADER)
      call assert(err == TGM_ERROR_OK, 'HEADER: encode OK')
      call walk(wire, frames, n, err)
      call assert(seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'HEADER: header hash frame')
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'HEADER: no footer hash frame')

      call encode_opts(wire, err, aggregate_hash=TGM_AGGREGATE_HASH_POLICY_FOOTER)
      call assert(err == TGM_ERROR_OK, 'FOOTER: encode OK')
      call walk(wire, frames, n, err)
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'FOOTER: no header hash frame')
      call assert(seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'FOOTER: footer hash frame')
      call assert(values_round_trip(wire), 'FOOTER: values round-trip')
   end subroutine aggregate_hash_placements

   ! ---- BOTH: a hash frame in the header AND in the footer -----------------
   subroutine aggregate_hash_both_is_visible_in_the_frames()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      type(tensogram_message_header) :: hdr
      integer(c_int) :: err
      integer :: n, k, header_hashes, footer_hashes

      call encode_opts(wire, err, aggregate_hash=TGM_AGGREGATE_HASH_POLICY_BOTH)
      call assert(err == TGM_ERROR_OK, 'BOTH: encode OK')
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'BOTH: walk OK')

      header_hashes = 0
      footer_hashes = 0
      do k = 1, n
         if (frames(k)%frame_type() == TGM_FRAME_TYPE_HEADER_HASH) header_hashes = header_hashes + 1
         if (frames(k)%frame_type() == TGM_FRAME_TYPE_FOOTER_HASH) footer_hashes = footer_hashes + 1
      end do
      call assert(header_hashes == 1, 'BOTH: exactly one header hash frame')
      call assert(footer_hashes == 1, 'BOTH: exactly one footer hash frame')
      ! Both frames carry the same hash list, so their payloads are identical.
      call assert(bytes_eq(payload_of(frames, n, TGM_FRAME_TYPE_HEADER_HASH), &
                           payload_of(frames, n, TGM_FRAME_TYPE_FOOTER_HASH)), &
                  'BOTH: header and footer hash frames carry identical lists')
      ! The header frame precedes the data objects; the footer frame follows them.
      call assert(index_of(frames, n, TGM_FRAME_TYPE_HEADER_HASH) < &
                  index_of(frames, n, TGM_FRAME_TYPE_NTENSOR), 'BOTH: header hash comes first')
      call assert(index_of(frames, n, TGM_FRAME_TYPE_FOOTER_HASH) > &
                  index_of(frames, n, TGM_FRAME_TYPE_NTENSOR), 'BOTH: footer hash comes last')

      ! The preamble advertises both, matching the frames exactly.
      call tensogram_message_header_read(wire, hdr, err)
      call assert(err == TGM_ERROR_OK, 'BOTH: header read OK')
      call assert(hdr%has_header_hashes(), 'BOTH: preamble flags header hashes')
      call assert(hdr%has_footer_hashes(), 'BOTH: preamble flags footer hashes')
      call assert(values_round_trip(wire), 'BOTH: values round-trip')
   end subroutine aggregate_hash_both_is_visible_in_the_frames

   ! ---- The hash algorithm now lives in the options ------------------------
   subroutine hash_algorithm_knob()
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_frame) :: frames(MAXF)
      integer(c_int) :: err
      integer :: n, k

      ! '' disables hashing entirely: no aggregate frame, no inline slots.
      call encode_opts(wire, err, hash='')
      call assert(err == TGM_ERROR_OK, 'hash="": encode OK')
      call walk(wire, frames, n, err)
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'hash="": no aggregate frame')
      do k = 1, n
         call assert(.not. frames(k)%has_hash(), 'hash="": no per-frame inline hash')
      end do

      call encode_opts(wire, err, hash='xxh3')
      call assert(err == TGM_ERROR_OK, 'hash="xxh3": encode OK')
      call walk(wire, frames, n, err)
      call assert(seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'hash="xxh3": aggregate frame')

      call encode_opts(wire, err, hash='not-an-algorithm')
      call assert(err == TGM_ERROR_INVALID_ARG, 'unknown hash algorithm -> INVALID_ARG')
   end subroutine hash_algorithm_knob

   ! ---- The codec backend is an implementation choice, not a format one ---
   subroutine compression_backend_knob()
      integer(c_int8_t), allocatable :: wire_auto(:), wire_ffi(:), wire_pure(:)
      integer(c_int) :: err
      call encode_opts(wire_auto, err, compression='zstd', &
                       compression_backend=TGM_COMPRESSION_BACKEND_AUTO)
      call assert(err == TGM_ERROR_OK, 'backend AUTO: encode OK')
      call encode_opts(wire_ffi, err, compression='zstd', &
                       compression_backend=TGM_COMPRESSION_BACKEND_FFI)
      call assert(err == TGM_ERROR_OK, 'backend FFI: encode OK')
      call encode_opts(wire_pure, err, compression='zstd', &
                       compression_backend=TGM_COMPRESSION_BACKEND_PURE)
      call assert(err == TGM_ERROR_OK, 'backend PURE: encode OK')
      ! Whichever codec produced the bytes, the values come back bit-identical.
      call assert(values_round_trip(wire_auto), 'backend AUTO: lossless round-trip')
      call assert(values_round_trip(wire_ffi),  'backend FFI: lossless round-trip')
      call assert(values_round_trip(wire_pure), 'backend PURE: lossless round-trip')
   end subroutine compression_backend_knob

   ! ---- The superseded mask knobs are still reachable ----------------------
   subroutine mask_options_still_reachable()
      real(c_float) :: vals(4)
      integer(c_int8_t), allocatable :: data(:), wire(:), mask(:)
      integer(c_size_t) :: lens(1)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      vals(1) = 1.0_c_float
      vals(2) = ieee_value(1.0_c_float, ieee_quiet_nan)
      vals(3) = ieee_value(1.0_c_float, ieee_positive_inf)
      vals(4) = 2.0_c_float
      data = transfer(vals, [0_c_int8_t], 4 * 4)
      lens(1) = int(4 * 4, c_size_t)

      ! Without allow_nan the encoder rejects the non-finite input …
      call tensogram_encode_with_options(json1(4, 'none'), data, lens, buf, err)
      call assert(err /= TGM_ERROR_OK, 'masks: non-finite rejected by default')

      ! … and with it the companion masks are written and recoverable.
      call tensogram_encode_with_options(json1(4, 'none'), data, lens, buf, err, &
                                         allow_nan=.true., allow_inf=.true.,     &
                                         nan_mask_method='rle',                  &
                                         small_mask_threshold_bytes=0_c_int64_t)
      call assert(err == TGM_ERROR_OK, 'masks: allow_nan/allow_inf + explicit method')
      call buf%as_array(wire)
      call tensogram_decode_with_masks(wire, msg, err)
      call assert(err == TGM_ERROR_OK, 'masks: decode_with_masks OK')
      call assert(tensogram_object_has_masks(msg, 1), 'masks: object carries masks')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NAN, mask, err)
      call assert(err == TGM_ERROR_OK .and. size(mask) == 4, 'masks: NaN mask has one byte per element')
      call assert(mask(2) == 1_c_int8_t, 'masks: NaN position recorded')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_POS_INF, mask, err)
      call assert(err == TGM_ERROR_OK .and. mask(3) == 1_c_int8_t, 'masks: +Inf position recorded')
   end subroutine mask_options_still_reachable

   ! ---- file_append_with_options -------------------------------------------
   subroutine file_append_with_options_cases()
      character(len=*), parameter :: path = 'test_encode_options_append.tgm'
      real(c_float) :: vals(4)
      integer(c_int8_t), allocatable :: data(:), wire(:)
      integer(c_size_t) :: lens(1)
      type(tensogram_file)   :: f
      type(tensogram_buffer) :: raw
      type(tensogram_frame)  :: frames(MAXF)
      integer(c_int) :: err
      integer :: n, unit, ios
      vals = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
      data = transfer(vals, [0_c_int8_t], 4 * 4)
      lens(1) = int(4 * 4, c_size_t)

      call tensogram_file_create(path, f, err)
      call assert(err == TGM_ERROR_OK, 'append_with_options: create')
      call tensogram_file_append_with_options(f, json1(4, 'none'), data, lens, err, &
                                              aggregate_hash=TGM_AGGREGATE_HASH_POLICY_BOTH)
      call assert(err == TGM_ERROR_OK, 'append_with_options: append BOTH')
      call tensogram_file_append_with_options(f, json1(4, 'zstd'), data, lens, err, &
                                              compression_backend=TGM_COMPRESSION_BACKEND_PURE)
      call assert(err == TGM_ERROR_OK, 'append_with_options: append zstd via the pure backend')
      call f%close()

      call tensogram_file_open(path, f, err)
      call assert(err == TGM_ERROR_OK, 'append_with_options: reopen')
      call tensogram_file_read_message(f, 1, raw, err)
      call assert(err == TGM_ERROR_OK, 'append_with_options: read message 1')
      call raw%as_array(wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'append_with_options: walk message 1')
      call assert(seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'append_with_options: header hash frame')
      call assert(seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'append_with_options: footer hash frame')
      call f%close()

      open(newunit=unit, file=path, status='old', iostat=ios)
      if (ios == 0) close(unit, status='delete')
   end subroutine file_append_with_options_cases

   ! ---- streaming_encoder_create_with_options ------------------------------
   subroutine streaming_create_with_options_cases()
      character(len=*), parameter :: path = 'test_encode_options_stream.tgm'
      real(c_float) :: vals(4)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_streaming_encoder) :: enc
      type(tensogram_file)   :: f
      type(tensogram_buffer) :: raw
      type(tensogram_frame)  :: frames(MAXF)
      integer(c_int) :: err
      integer :: n, unit, ios
      vals = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]

      ! HEADER / BOTH cannot work when the header is written first.
      call tensogram_streaming_encoder_create_with_options(path, enc, err, &
              aggregate_hash=TGM_AGGREGATE_HASH_POLICY_HEADER)
      call assert(err == TGM_ERROR_ENCODING, 'streaming: HEADER policy rejected')
      call tensogram_streaming_encoder_create_with_options(path, enc, err, &
              aggregate_hash=TGM_AGGREGATE_HASH_POLICY_BOTH)
      call assert(err == TGM_ERROR_ENCODING, 'streaming: BOTH policy rejected')
      call assert(len(tensogram_last_error()) > 0, 'streaming: rejection explains itself')

      ! FOOTER (and AUTO, which resolves to it) is the valid streaming choice.
      call tensogram_streaming_encoder_create_with_options(path, enc, err,      &
              aggregate_hash=TGM_AGGREGATE_HASH_POLICY_FOOTER,                  &
              compression_backend=TGM_COMPRESSION_BACKEND_PURE)
      call assert(err == TGM_ERROR_OK, 'streaming: FOOTER policy accepted')
      call tensogram_streaming_encoder_write(enc, vals, err, compression='zstd')
      call assert(err == TGM_ERROR_OK, 'streaming: write object')
      call tensogram_streaming_encoder_finish(enc, err)
      call assert(err == TGM_ERROR_OK, 'streaming: finish')
      call enc%free()

      call tensogram_file_open(path, f, err)
      call assert(err == TGM_ERROR_OK, 'streaming: reopen')
      call tensogram_file_read_message(f, 1, raw, err)
      call assert(err == TGM_ERROR_OK, 'streaming: read message')
      call raw%as_array(wire)
      call walk(wire, frames, n, err)
      call assert(err == TGM_ERROR_OK, 'streaming: walk')
      call assert(seen(frames, n, TGM_FRAME_TYPE_FOOTER_HASH), 'streaming: footer hash frame')
      call assert(.not. seen(frames, n, TGM_FRAME_TYPE_HEADER_HASH), 'streaming: no header hash frame')
      call f%close()

      open(newunit=unit, file=path, status='old', iostat=ios)
      if (ios == 0) close(unit, status='delete')
   end subroutine streaming_create_with_options_cases

   ! ---- helpers ------------------------------------------------------------

   !> Encode the fixed 4-element float32 fixture through
   !> tensogram_encode_with_options, forwarding whichever knobs were given.
   subroutine encode_opts(wire, err, hash, aggregate_hash, compression_backend, compression)
      integer(c_int8_t), allocatable, intent(out) :: wire(:)
      integer(c_int),                 intent(out) :: err
      character(len=*), intent(in), optional :: hash, compression
      integer(c_int),   intent(in), optional :: aggregate_hash, compression_backend
      real(c_float) :: vals(4)
      integer(c_int8_t), allocatable :: data(:)
      integer(c_size_t) :: lens(1)
      character(len=:), allocatable :: cmp
      type(tensogram_buffer) :: buf
      vals = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
      data = transfer(vals, [0_c_int8_t], 4 * 4)
      lens(1) = int(4 * 4, c_size_t)
      cmp = 'none'
      if (present(compression)) cmp = compression
      call tensogram_encode_with_options(json1(4, cmp), data, lens, buf, err, &
                                         hash=hash, aggregate_hash=aggregate_hash, &
                                         compression_backend=compression_backend)
      if (err == TGM_ERROR_OK) then
         call buf%as_array(wire)
      else
         allocate(wire(0))
      end if
   end subroutine encode_opts

   !> Decode `wire` and check the fixture's values survived bit-exactly.
   logical function values_round_trip(wire)
      integer(c_int8_t), intent(in) :: wire(:)
      real(c_float), allocatable  :: got(:)
      type(tensogram_message) :: msg
      integer(c_int) :: err
      values_round_trip = .false.
      call tensogram_decode(wire, msg, err)
      if (err /= TGM_ERROR_OK) return
      call tensogram_to_array(msg, 1, got, err)
      if (err /= TGM_ERROR_OK) return
      if (size(got) /= 4) return
      values_round_trip = all(transfer(got, [0_c_int32_t]) == &
                              transfer([1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float], &
                                       [0_c_int32_t]))
   end function values_round_trip

   !> One-object descriptor envelope: `n` float32 elements, pipeline `cmp`.
   function json1(n, cmp) result(s)
      integer,          intent(in) :: n
      character(len=*), intent(in) :: cmp
      character(len=:), allocatable :: s
      character(len=32) :: ns
      write (ns, '(i0)') n
      s = '{"descriptors":[{"type":"ndarray","ndim":1,"shape":[' // trim(ns) //  &
          '],"strides":[1],"dtype":"float32","byte_order":"' // host_bo() //     &
          '","encoding":"none","filter":"none","compression":"' // cmp // '"}]}'
   end function json1

   !> Walk `wire` to exhaustion, collecting the frames (component A3).
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

   logical function seen(frames, n, ftype)
      type(tensogram_frame), intent(in) :: frames(:)
      integer,               intent(in) :: n
      integer(c_int),        intent(in) :: ftype
      seen = index_of(frames, n, ftype) > 0
   end function seen

   !> Position of the first frame of type `ftype` (0 when absent).
   integer function index_of(frames, n, ftype)
      type(tensogram_frame), intent(in) :: frames(:)
      integer,               intent(in) :: n
      integer(c_int),        intent(in) :: ftype
      integer :: k
      index_of = 0
      do k = n, 1, -1
         if (frames(k)%frame_type() == ftype) index_of = k
      end do
   end function index_of

   !> Payload bytes of the first frame of type `ftype` (size 0 when absent).
   function payload_of(frames, n, ftype) result(p)
      type(tensogram_frame), intent(in) :: frames(:)
      integer,               intent(in) :: n
      integer(c_int),        intent(in) :: ftype
      integer(c_int8_t), allocatable :: p(:)
      integer :: k
      k = index_of(frames, n, ftype)
      if (k == 0) then
         allocate(p(0))
      else
         p = frames(k)%payload()
      end if
   end function payload_of

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
      bytes_eq = size(x) == size(y) .and. size(x) > 0
      if (bytes_eq) bytes_eq = all(x == y)
   end function bytes_eq

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_encode_options: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

end program test_encode_options

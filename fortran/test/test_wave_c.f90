! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Wave-C symmetry surface (Wave-B C ABI): buffer-level validate, compute_common
!> over a decoded metadata handle, canonical-CBOR verification, the scan-options
!> / file-scan family (scan_file / scan_with_options / scan_file_with_options),
!> per-message inline object hashes, and the mask-aware decode
!> (decode_with_masks + object_has_masks + object_mask) with its mask-companion
!> encode entry point (encode_with_masks). Each procedure is exercised for both
!> its success path and (where it applies) a graceful error / absent path.
program test_wave_c
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic
   use tensogram
   implicit none

   integer :: npass

   npass = 0

   call validate_buffer_cases()
   call compute_common_cases()
   call verify_canonical_cbor_cases()
   call scan_family_cases()
   call inline_hashes_cases()
   call mask_kind_constants()
   call decode_with_masks_cases()

   print '(a,i0,a)', 'test_wave_c: PASS (', npass, ' checks)'

contains

   ! ---- Task 1: validate_buffer (multi-message wire buffer) ----------------
   subroutine validate_buffer_cases()
      real(c_float)      :: a(3)
      integer(c_int32_t) :: b(5)
      integer(c_int8_t), allocatable :: wa(:), wb(:), wire(:), empty(:)
      type(tensogram_buffer) :: bufa, bufb
      character(len=:), allocatable :: rep
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      b = [10_c_int32_t, 20_c_int32_t, 30_c_int32_t, 40_c_int32_t, 50_c_int32_t]
      call tensogram_encode(a, bufa, err); call assert(err == TGM_ERROR_OK, 'validate_buffer: encode A')
      call tensogram_encode(b, bufb, err); call assert(err == TGM_ERROR_OK, 'validate_buffer: encode B')
      call bufa%as_array(wa)
      call bufb%as_array(wb)
      wire = [wa, wb]                          ! two messages back-to-back

      rep = tensogram_validate_buffer(wire, err=err)
      call assert(err == TGM_ERROR_OK, 'validate_buffer: OK')
      call assert(has(rep, '"file_issues"'), 'validate_buffer: has file_issues')
      call assert(has(rep, '"messages"'), 'validate_buffer: has messages')
      call assert(has(rep, '"file_issues":[]'), 'validate_buffer: clean buffer, no file issues')

      rep = tensogram_validate_buffer(wire, level='full', check_canonical=.true.)
      call assert(has(rep, '"messages"'), 'validate_buffer full+canonical: has messages')

      allocate(empty(0))
      rep = tensogram_validate_buffer(empty, err=err)
      call assert(err == TGM_ERROR_OK, 'validate_buffer empty: OK report')
      call assert(len(rep) > 0, 'validate_buffer empty: non-empty report')

      rep = tensogram_validate_buffer(wire, level='bogus', err=err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'validate_buffer bogus level -> INVALID_ARG')
      call assert(len(rep) == 0, 'validate_buffer bogus level -> empty report')
   end subroutine validate_buffer_cases

   ! ---- Task 2a: compute_common (over a decoded metadata handle) -----------
   subroutine compute_common_cases()
      real(c_float)                  :: c3(3), d2(2)
      integer(c_int8_t), allocatable :: data2(:), wire(:)
      integer(c_size_t)              :: lens2(2)
      character(len=:), allocatable  :: json
      type(tensogram_buffer)   :: buf
      type(tensogram_metadata) :: meta, empty_meta
      integer(c_int) :: err
      integer :: pos_units, pos_remaining
      c3 = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      d2 = [10.0_c_float, 20.0_c_float]
      ! Two objects: "units" is shared (=> common), "name" differs (=> remaining).
      json  = '{"descriptors":[' // desc1d(3, host_bo()) // ',' // desc1d(2, host_bo()) // '],' // &
              '"base":[{"units":"K","name":"a"},{"units":"K","name":"b"}]}'
      data2 = [transfer(c3, [0_c_int8_t]), transfer(d2, [0_c_int8_t])]
      lens2(1) = int(3 * 4, c_size_t)
      lens2(2) = int(2 * 4, c_size_t)
      call tensogram_encode_pre_encoded(json, data2, lens2, buf, err)
      call assert(err == TGM_ERROR_OK, 'compute_common: build 2-object buffer')
      call buf%as_array(wire)
      call tensogram_decode_metadata(wire, meta, err)
      call assert(err == TGM_ERROR_OK, 'compute_common: decode_metadata')
      call assert(tensogram_metadata_num_objects(meta) == 2, 'compute_common: 2 objects')

      json = tensogram_compute_common(meta, err)
      call assert(err == TGM_ERROR_OK, 'compute_common: OK')
      call assert(has(json, '"common"'), 'compute_common: has common')
      call assert(has(json, '"remaining"'), 'compute_common: has remaining')
      call assert(has(json, '"units"'), 'compute_common: has shared key units')
      call assert(has(json, '"name"'), 'compute_common: has varying key name')
      ! The shared "units" lives in the "common" section, which precedes
      ! "remaining" in the {"common":{...},"remaining":[...]} layout.
      pos_units     = index(json, '"units"')
      pos_remaining = index(json, '"remaining"')
      call assert(pos_units < pos_remaining, 'compute_common: units is in common (before remaining)')
      call meta%free()

      ! Error path: a null (unassigned) metadata handle -> INVALID_ARG, ''.
      json = tensogram_compute_common(empty_meta, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'compute_common null handle -> INVALID_ARG')
      call assert(len(json) == 0, 'compute_common null handle -> empty')
   end subroutine compute_common_cases

   ! ---- Task 2b: verify_canonical_cbor -------------------------------------
   subroutine verify_canonical_cbor_cases()
      integer(c_int8_t) :: canon(5), noncanon(5), scalar(1), invalid(1)
      integer(c_int8_t), allocatable :: empty(:)
      ! Canonical map {1:0, 2:0}  = A2 01 00 02 00  (keys sorted 1 < 2).
      canon    = [u8(int(z'A2')), u8(1), u8(0), u8(2), u8(0)]
      ! Non-canonical map {2:0, 1:0} = A2 02 00 01 00  (keys out of order).
      noncanon = [u8(int(z'A2')), u8(2), u8(0), u8(1), u8(0)]
      scalar   = [u8(1)]                        ! bare uint 1 — no maps, canonical
      invalid  = [u8(int(z'FF'))]               ! lone break — not valid CBOR
      allocate(empty(0))

      call assert(tensogram_verify_canonical_cbor(canon), 'verify_canonical: canonical map -> true')
      call assert(tensogram_verify_canonical_cbor(scalar), 'verify_canonical: bare int -> true')
      call assert(.not. tensogram_verify_canonical_cbor(noncanon), 'verify_canonical: reordered map -> false')
      call assert(.not. tensogram_verify_canonical_cbor(invalid), 'verify_canonical: bad CBOR -> false')
      call assert(.not. tensogram_verify_canonical_cbor(empty), 'verify_canonical: empty -> false')
   end subroutine verify_canonical_cbor_cases

   ! ---- Task 3: scan family (buffer/file, with & without options) ----------
   subroutine scan_family_cases()
      character(len=*), parameter :: path = 'test_wave_c_scan.tgm'
      real(c_float)      :: a(3)
      integer(c_int32_t) :: b(5)
      integer(c_int8_t), allocatable :: wa(:), wb(:), wire(:)
      integer(c_size_t), allocatable :: offs(:), lens(:), o2(:), l2(:)
      type(tensogram_buffer) :: bufa, bufb
      type(tensogram_file)   :: f
      type(tensogram_scan_options) :: opts
      integer(c_int) :: err
      integer :: ios
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      b = [10_c_int32_t, 20_c_int32_t, 30_c_int32_t, 40_c_int32_t, 50_c_int32_t]
      call tensogram_encode(a, bufa, err); call assert(err == TGM_ERROR_OK, 'scan_family: encode A')
      call tensogram_encode(b, bufb, err); call assert(err == TGM_ERROR_OK, 'scan_family: encode B')
      call bufa%as_array(wa)
      call bufb%as_array(wb)
      wire = [wa, wb]

      ! scan_with_options, default opts (== library defaults == NULL pointer).
      call tensogram_scan_with_options(wire, opts, offs, lens, err)
      call assert(err == TGM_ERROR_OK, 'scan_with_options default: OK')
      call assert(size(offs) == 2, 'scan_with_options default: two messages')
      call assert(offs(1) == 1_c_size_t, 'scan_with_options: first offset 1-based')
      call assert(int(lens(1)) == size(wa), 'scan_with_options: first length')

      ! scan_with_options, forward-only walk (bidirectional = .false.).
      opts%bidirectional = .false.
      call tensogram_scan_with_options(wire, opts, o2, l2, err)
      call assert(err == TGM_ERROR_OK, 'scan_with_options forward: OK')
      call assert(size(o2) == 2, 'scan_with_options forward: two messages')
      call assert(all(o2 == offs) .and. all(l2 == lens), 'scan_with_options forward: same boundaries')

      ! Omitting opts entirely (NULL pointer path) still uses library defaults.
      call tensogram_scan_with_options(wire, offsets=o2, lengths=l2, err=err)
      call assert(err == TGM_ERROR_OK, 'scan_with_options absent-opts: OK')
      call assert(size(o2) == 2, 'scan_with_options absent-opts: two messages')

      ! scan_file / scan_file_with_options on a 2-message file.
      call tensogram_file_create(path, f, err); call assert(err == TGM_ERROR_OK, 'scan_file: create')
      call tensogram_file_append(f, a, err);     call assert(err == TGM_ERROR_OK, 'scan_file: append A')
      call tensogram_file_append(f, b, err);     call assert(err == TGM_ERROR_OK, 'scan_file: append B')
      call f%close()

      call tensogram_scan_file(path, offs, lens, err)
      call assert(err == TGM_ERROR_OK, 'scan_file: OK')
      call assert(size(offs) == 2, 'scan_file: two messages')
      call assert(offs(1) == 1_c_size_t, 'scan_file: first offset 1-based')

      call tensogram_scan_file_with_options(path, opts, o2, l2, err)
      call assert(err == TGM_ERROR_OK, 'scan_file_with_options: OK')
      call assert(size(o2) == 2, 'scan_file_with_options: two messages')
      call assert(all(o2 == offs) .and. all(l2 == lens), 'scan_file_with_options: same boundaries')

      ! Error path: a missing file -> IO, empty results.
      call tensogram_scan_file('/nonexistent/tensogram_wave_c_missing.tgm', offs, lens, err)
      call assert(err == TGM_ERROR_IO, 'scan_file missing -> IO')
      call assert(size(offs) == 0, 'scan_file missing -> empty')

      open(newunit=ios, file=path, status='old', iostat=err)
      if (err == 0) close(ios, status='delete')
   end subroutine scan_family_cases

   ! ---- Task 3b: data_object_inline_hashes ---------------------------------
   subroutine inline_hashes_cases()
      real(c_float)                  :: a(4)
      integer(c_int8_t), allocatable :: wire(:)
      character(len=:), allocatable  :: json, digest
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]

      ! Hashed message: the single-object inline-hash array carries the digest.
      call tensogram_encode(a, buf, err, hash='xxh3'); call assert(err == TGM_ERROR_OK, 'inline_hashes: encode hashed')
      call buf%as_array(wire)
      json = tensogram_data_object_inline_hashes(wire, err)
      call assert(err == TGM_ERROR_OK, 'inline_hashes: OK')
      call assert(len(json) > 0 .and. json(1:1) == '[', 'inline_hashes: JSON array')
      call assert(.not. has(json, 'null'), 'inline_hashes hashed: no null slot')
      ! Cross-check the array digest against the per-object accessor.
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'inline_hashes: decode')
      digest = tensogram_object_hash_value(msg, 1)
      call assert(len(digest) == 16, 'inline_hashes: object digest is 16 hex chars')
      call assert(has(json, digest), 'inline_hashes: array carries the object digest')

      ! Unhashed message: the slot is JSON null.
      call tensogram_encode(a, buf, err, hash=''); call assert(err == TGM_ERROR_OK, 'inline_hashes: encode unhashed')
      call buf%as_array(wire)
      json = tensogram_data_object_inline_hashes(wire, err)
      call assert(err == TGM_ERROR_OK, 'inline_hashes unhashed: OK')
      call assert(has(json, 'null'), 'inline_hashes unhashed: null slot')
   end subroutine inline_hashes_cases

   ! ---- Task 4a: mask-kind constants ---------------------------------------
   subroutine mask_kind_constants()
      call assert(TGM_MASK_KIND_NAN     == 0, 'TGM_MASK_KIND_NAN == 0')
      call assert(TGM_MASK_KIND_POS_INF == 1, 'TGM_MASK_KIND_POS_INF == 1')
      call assert(TGM_MASK_KIND_NEG_INF == 2, 'TGM_MASK_KIND_NEG_INF == 2')
   end subroutine mask_kind_constants

   ! ---- Task 4b: decode_with_masks + has_masks + object_mask ---------------
   subroutine decode_with_masks_cases()
      real(c_float)                  :: vals(5), fin(3)
      real(c_float), allocatable     :: payload(:)
      integer(c_int8_t), allocatable :: data(:), wire(:), mask(:)
      integer(c_size_t)              :: lens1(1)
      character(len=:), allocatable  :: json, json3
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      ! [finite, NaN, +Inf, -Inf, finite]
      vals(1) = 1.0_c_float
      vals(2) = ieee_value(1.0_c_float, ieee_quiet_nan)
      vals(3) = ieee_value(1.0_c_float, ieee_positive_inf)
      vals(4) = ieee_value(1.0_c_float, ieee_negative_inf)
      vals(5) = 2.0_c_float
      json  = '{"descriptors":[' // desc1d(5, host_bo()) // ']}'
      data  = transfer(vals, [0_c_int8_t], 5 * 4)
      lens1(1) = int(5 * 4, c_size_t)

      ! Default encode REJECTS non-finite values.
      call tensogram_encode_with_masks(json, data, lens1, buf, err)
      call assert(err /= TGM_ERROR_OK, 'encode_with_masks: rejects non-finite by default')

      ! With allow_nan + allow_inf the non-finite positions become companion masks.
      call tensogram_encode_with_masks(json, data, lens1, buf, err, &
                                       allow_nan=.true., allow_inf=.true.)
      call assert(err == TGM_ERROR_OK, 'encode_with_masks: allow_nan+allow_inf OK')
      call buf%as_array(wire)

      call tensogram_decode_with_masks(wire, msg, err)
      call assert(err == TGM_ERROR_OK, 'decode_with_masks: OK')
      call assert(tensogram_num_objects(msg) == 1, 'decode_with_masks: one object')
      call assert(tensogram_object_has_masks(msg, 1), 'decode_with_masks: object has masks')

      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NAN, mask, err)
      call assert(err == TGM_ERROR_OK, 'object_mask NaN: OK')
      call assert(meq(mask, [0_c_int8_t, 1_c_int8_t, 0_c_int8_t, 0_c_int8_t, 0_c_int8_t]), &
                  'object_mask NaN: byte at NaN position')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_POS_INF, mask, err)
      call assert(err == TGM_ERROR_OK, 'object_mask +Inf: OK')
      call assert(meq(mask, [0_c_int8_t, 0_c_int8_t, 1_c_int8_t, 0_c_int8_t, 0_c_int8_t]), &
                  'object_mask +Inf: byte at +Inf position')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NEG_INF, mask, err)
      call assert(err == TGM_ERROR_OK, 'object_mask -Inf: OK')
      call assert(meq(mask, [0_c_int8_t, 0_c_int8_t, 0_c_int8_t, 1_c_int8_t, 0_c_int8_t]), &
                  'object_mask -Inf: byte at -Inf position')

      ! decode_with_masks leaves the canonical 0.0 substitution in place.
      call tensogram_to_array(msg, 1, payload, err)
      call assert(err == TGM_ERROR_OK, 'decode_with_masks: to_array')
      call assert(feq(payload, [1.0_c_float, 0.0_c_float, 0.0_c_float, 0.0_c_float, 2.0_c_float]), &
                  'decode_with_masks: payload holds 0.0 at masked positions')

      ! Argument errors: unknown kind and out-of-range object -> INVALID_ARG.
      call tensogram_object_mask(msg, 1, 99_c_int, mask, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'object_mask bad kind -> INVALID_ARG')
      call assert(size(mask) == 0, 'object_mask bad kind -> empty')
      call tensogram_object_mask(msg, 9, TGM_MASK_KIND_NAN, mask, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'object_mask OOR index -> INVALID_ARG')
      call assert(size(mask) == 0, 'object_mask OOR index -> empty')

      ! Plain decode of the SAME wire restores in place and keeps no masks.
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'plain decode: OK')
      call assert(.not. tensogram_object_has_masks(msg, 1), 'plain decode: no masks retained')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NAN, mask, err)
      call assert(err == TGM_ERROR_OK, 'plain decode object_mask: OK (absent)')
      call assert(size(mask) == 0, 'plain decode object_mask: empty (absent)')

      ! NaN-only: the two Inf kinds are genuinely absent (empty, not an error).
      json3 = '{"descriptors":[' // desc1d(3, host_bo()) // ']}'
      fin(1) = 1.0_c_float
      fin(2) = ieee_value(1.0_c_float, ieee_quiet_nan)
      fin(3) = 2.0_c_float
      data  = transfer(fin, [0_c_int8_t], 3 * 4)
      lens1(1) = int(3 * 4, c_size_t)
      call tensogram_encode_with_masks(json3, data, lens1, buf, err, allow_nan=.true.)
      call assert(err == TGM_ERROR_OK, 'encode_with_masks NaN-only: OK')
      call buf%as_array(wire)
      call tensogram_decode_with_masks(wire, msg, err)
      call assert(err == TGM_ERROR_OK, 'decode_with_masks NaN-only: OK')
      call assert(tensogram_object_has_masks(msg, 1), 'NaN-only: has masks')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NAN, mask, err)
      call assert(meq(mask, [0_c_int8_t, 1_c_int8_t, 0_c_int8_t]), 'NaN-only: nan mask')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_POS_INF, mask, err)
      call assert(err == TGM_ERROR_OK .and. size(mask) == 0, 'NaN-only: +Inf mask absent')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NEG_INF, mask, err)
      call assert(err == TGM_ERROR_OK .and. size(mask) == 0, 'NaN-only: -Inf mask absent')

      ! Finite data: no non-finite -> no masks written at all.
      fin = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      data = transfer(fin, [0_c_int8_t], 3 * 4)
      call tensogram_encode_with_masks(json3, data, lens1, buf, err, &
                                       allow_nan=.true., allow_inf=.true.)
      call assert(err == TGM_ERROR_OK, 'encode_with_masks finite: OK')
      call buf%as_array(wire)
      call tensogram_decode_with_masks(wire, msg, err)
      call assert(err == TGM_ERROR_OK, 'decode_with_masks finite: OK')
      call assert(.not. tensogram_object_has_masks(msg, 1), 'finite: no masks')
      call tensogram_object_mask(msg, 1, TGM_MASK_KIND_NAN, mask, err)
      call assert(err == TGM_ERROR_OK .and. size(mask) == 0, 'finite: nan mask absent')
   end subroutine decode_with_masks_cases

   ! ---- helpers ------------------------------------------------------------

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_wave_c: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

   !> Substring test (JSON key presence).
   logical function has(hay, needle)
      character(len=*), intent(in) :: hay, needle
      has = index(hay, needle) > 0
   end function has

   !> Exact equality of two int8 mask vectors.
   logical function meq(x, y)
      integer(c_int8_t), intent(in) :: x(:), y(:)
      meq = size(x) == size(y)
      if (meq) meq = all(x == y)
   end function meq

   !> Bit-exact equality of two float32 vectors (avoids the -Wcompare-reals
   !> real `==` warning; also the correct test for a lossless round-trip).
   logical function feq(x, y)
      real(c_float), intent(in) :: x(:), y(:)
      feq = size(x) == size(y)
      if (feq) feq = all(transfer(x, [0_c_int32_t]) == transfer(y, [0_c_int32_t]))
   end function feq

   !> An unsigned byte (0..255) as its signed c_int8_t bit pattern.
   pure function u8(x) result(r)
      integer, intent(in) :: x
      integer(c_int8_t) :: r
      if (x > 127) then
         r = int(x - 256, c_int8_t)
      else
         r = int(x, c_int8_t)
      end if
   end function u8

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

   !> A bare 1-D float32 descriptor (encoding=none) for `n` elements — the
   !> pipeline-less path where pre-encoded bytes equal the raw element bytes.
   function desc1d(n, bo) result(s)
      integer,          intent(in) :: n
      character(len=*), intent(in) :: bo
      character(len=:), allocatable :: s
      character(len=32) :: ns
      write (ns, '(i0)') n
      s = '{"type":"ndarray","ndim":1,"shape":[' // trim(ns) //          &
          '],"strides":[4],"dtype":"float32","byte_order":"' // bo //     &
          '","encoding":"none","filter":"none","compression":"none"}'
   end function desc1d

end program test_wave_c

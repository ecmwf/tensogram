! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> examples/fortran/masks.f90
!>
!> Wave-C reader/writer helpers (the widened C ABI):
!>   * encode_with_masks + decode_with_masks + object_has_masks / object_mask —
!>     encode a NaN / ±Inf tensor with bitmask companion frames, then decode
!>     WITHOUT restoring the non-finite values in place and recover the
!>     per-element masks instead.
!>   * validate_buffer + data_object_inline_hashes — validate every message in
!>     a multi-message buffer and recover each frame's inline xxh3 slot.
!>   * scan_file / scan_with_options — locate message boundaries on disk / in
!>     memory (with explicit TgmScanOptions).
!>   * compute_common — split the per-object metadata into the shared "common"
!>     keys and the per-object "remaining" keys.
!>   * verify_canonical_cbor — check RFC 8949 canonical CBOR key ordering.
!>
!> Mirrors the C++ `examples/cpp/17_masks_scan_validate.cpp`.
program masks
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'tensogram_example_masks_scan.tgm'
   real(c_float)                  :: vals(5)
   real(c_float), allocatable     :: payload(:)
   integer(c_int8_t), allocatable :: data(:), wire(:), wa(:), wb(:), buffer(:), mask(:)
   integer(c_size_t), allocatable :: offsets(:), lengths(:)
   integer(c_size_t)              :: lens1(1)
   character(len=:), allocatable  :: json, report
   type(tensogram_buffer)   :: buf, bufa, bufb
   type(tensogram_message)  :: msg
   type(tensogram_metadata) :: meta
   type(tensogram_scan_options) :: opts
   type(tensogram_file)     :: f
   integer(c_int) :: err, kind
   integer :: k

   ! ── 1. decode_with_masks: keep the non-finite masks ─────────────────────
   print '(a)', '=== decode_with_masks ==='
   vals(1) = 1.0_c_float
   vals(2) = ieee_value(1.0_c_float, ieee_quiet_nan)
   vals(3) = ieee_value(1.0_c_float, ieee_positive_inf)
   vals(4) = ieee_value(1.0_c_float, ieee_negative_inf)
   vals(5) = 2.0_c_float
   data = transfer(vals, [0_c_int8_t], 5 * 4)
   lens1(1) = int(5 * 4, c_size_t)
   call tensogram_encode_with_masks('{"descriptors":[' // desc1d(5) // ']}', &
        data, lens1, buf, err, allow_nan=.true., allow_inf=.true.)
   call tensogram_check(err, 'encode_with_masks')
   call buf%as_array(wire)

   call tensogram_decode_with_masks(wire, msg, err)
   call tensogram_check(err, 'decode_with_masks')
   print '(a,l1)', 'object has_masks: ', tensogram_object_has_masks(msg, 1)

   ! The payload keeps the canonical 0.0 at every masked position.
   call tensogram_to_array(msg, 1, payload, err)
   call tensogram_check(err, 'to_array')
   write (*, '(a)', advance='no') 'payload (0.0 where non-finite):'
   do k = 1, size(payload)
      write (*, '(1x,f4.1)', advance='no') payload(k)
   end do
   print '(a)', ''

   do kind = TGM_MASK_KIND_NAN, TGM_MASK_KIND_NEG_INF
      call tensogram_object_mask(msg, 1, kind, mask, err)
      call tensogram_check(err, 'object_mask')
      write (*, '(a,a,a)', advance='no') '  mask[', kind_name(kind), ']:'
      if (size(mask) == 0) then
         write (*, '(a)', advance='no') ' (absent)'
      else
         do k = 1, size(mask)
            write (*, '(1x,i0)', advance='no') int(mask(k))
         end do
      end if
      print '(a)', ''
   end do

   ! ── 2. validate_buffer + inline hashes (multi-message buffer) ───────────
   print '(a)', ''
   print '(a)', '=== validate_buffer (multi-message) ==='
   call tensogram_encode([1.0_c_float, 2.0_c_float, 3.0_c_float], bufa, err); call tensogram_check(err, 'encode A')
   call tensogram_encode([4.0_c_float, 5.0_c_float], bufb, err);              call tensogram_check(err, 'encode B')
   call bufa%as_array(wa)
   call bufb%as_array(wb)
   buffer = [wa, wb]
   report = tensogram_validate_buffer(buffer, level='full', err=err)
   call tensogram_check(err, 'validate_buffer')
   print '(a,l1,a,l1)', 'has messages=', index(report, '"messages"') > 0, &
      '  file_issues empty=', index(report, '"file_issues":[]') > 0

   json = tensogram_data_object_inline_hashes(wa, err)
   call tensogram_check(err, 'data_object_inline_hashes')
   print '(a,a)', 'inline hashes of message 0: ', json

   ! ── 3. scan_file / scan_with_options: message boundaries ────────────────
   print '(a)', ''
   print '(a)', '=== scan_file / scan_with_options ==='
   call tensogram_file_create(path, f, err); call tensogram_check(err, 'file_create')
   call tensogram_file_append(f, [1.0_c_float, 2.0_c_float, 3.0_c_float], err); call tensogram_check(err, 'append A')
   call tensogram_file_append(f, [4.0_c_float, 5.0_c_float], err);              call tensogram_check(err, 'append B')
   call f%close()

   call tensogram_scan_file(path, offsets, lengths, err)
   call tensogram_check(err, 'scan_file')
   print '(a,i0,a)', 'scan_file found ', size(offsets), ' message(s):'
   do k = 1, size(offsets)
      print '(a,i0,a,i0,a,i0)', '  [', k, '] offset=', offsets(k), ' length=', lengths(k)
   end do

   opts%bidirectional = .false.               ! force the forward-only walk
   call tensogram_scan_with_options(buffer, opts, offsets, lengths, err)
   call tensogram_check(err, 'scan_with_options')
   print '(a,i0,a)', 'scan_with_options (forward) found ', size(offsets), ' message(s)'

   open(newunit=k, file=path, status='old', iostat=err)
   if (err == 0) close(k, status='delete')

   ! ── 4. compute_common: shared vs per-object metadata ────────────────────
   print '(a)', ''
   print '(a)', '=== compute_common ==='
   call build_two_object_message(wire)
   call tensogram_decode_metadata(wire, meta, err)
   call tensogram_check(err, 'decode_metadata')
   json = tensogram_compute_common(meta, err)
   call tensogram_check(err, 'compute_common')
   print '(a,a)', 'common/remaining: ', json
   call meta%free()

   ! ── 5. verify_canonical_cbor: RFC 8949 key ordering ─────────────────────
   print '(a)', ''
   print '(a)', '=== verify_canonical_cbor ==='
   ! map {1:0, 2:0} = A2 01 00 02 00 (canonical); {2:0, 1:0} is reordered.
   print '(a,l1)', 'canonical map   {1:0,2:0}: ', &
      tensogram_verify_canonical_cbor([u8(int(z'A2')), u8(1), u8(0), u8(2), u8(0)])
   print '(a,l1)', 'reordered map   {2:0,1:0}: ', &
      tensogram_verify_canonical_cbor([u8(int(z'A2')), u8(2), u8(0), u8(1), u8(0)])

contains

   !> A 1-D float32 descriptor for `n` elements (no encoding pipeline).
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

   !> Human-readable mask-kind label.
   function kind_name(kind) result(name)
      integer(c_int), intent(in) :: kind
      character(len=:), allocatable :: name
      select case (kind)
      case (TGM_MASK_KIND_NAN);     name = 'nan '
      case (TGM_MASK_KIND_POS_INF); name = '+inf'
      case (TGM_MASK_KIND_NEG_INF); name = '-inf'
      case default;                 name = '?   '
      end select
   end function kind_name

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

   !> Encode a 2-object message whose base metadata shares "units" but differs
   !> in "name" — so compute_common yields a non-trivial common / remaining.
   subroutine build_two_object_message(out_wire)
      integer(c_int8_t), allocatable, intent(out) :: out_wire(:)
      real(c_float)                  :: c3(3), d2(2)
      integer(c_int8_t), allocatable :: bytes(:)
      integer(c_size_t)              :: lens(2)
      type(tensogram_buffer) :: b
      integer(c_int) :: e
      c3 = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      d2 = [10.0_c_float, 20.0_c_float]
      bytes = [transfer(c3, [0_c_int8_t]), transfer(d2, [0_c_int8_t])]
      lens(1) = int(3 * 4, c_size_t)
      lens(2) = int(2 * 4, c_size_t)
      call tensogram_encode_pre_encoded('{"descriptors":[' // desc1d(3) // ',' // desc1d(2) // '],' // &
           '"base":[{"units":"K","name":"a"},{"units":"K","name":"b"}]}', bytes, lens, b, e)
      call tensogram_check(e, 'build_two_object_message')
      call b%as_array(out_wire)
   end subroutine build_two_object_message

end program masks

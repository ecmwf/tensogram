! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Wave-A symmetry surface: wire-version accessors + TGM_WIRE_VERSION, the
!> environment doctor, message/file validate, buffer scan, standalone
!> compute_hash and the inline object-hash accessors, the object descriptor
!> accessors (type / byte_order / filter / compression / encoding / strides),
!> pre-encoded encode, and the decode-object / decode-range / decode-metadata
!> variants. Each new procedure is exercised for both success and (where it
!> applies) a graceful error / out-of-range path.
program test_symmetry
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   integer :: npass

   npass = 0

   call version_and_wire_constant()
   call doctor_report()
   call validate_buffer_and_file()
   call scan_multi_message()
   call hash_and_object_hash()
   call object_descriptor_accessors()
   call pre_encoded_roundtrip()
   call decode_variants()

   print '(a,i0,a)', 'test_symmetry: PASS (', npass, ' checks)'

contains

   ! ---- Task 1: wire version ----------------------------------------------
   subroutine version_and_wire_constant()
      real(c_float)                  :: a(3)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)   :: buf
      type(tensogram_message)  :: msg
      type(tensogram_metadata) :: meta
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      call assert(TGM_WIRE_VERSION == 3, 'TGM_WIRE_VERSION == 3')
      call tensogram_encode(a, buf, err);    call assert(err == TGM_ERROR_OK, 'version: encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'version: decode')
      call assert(tensogram_message_version(msg) == TGM_WIRE_VERSION, 'message_version == wire version')
      call tensogram_decode_metadata(wire, meta, err)
      call assert(err == TGM_ERROR_OK, 'version: decode_metadata')
      call assert(tensogram_metadata_version(meta) == TGM_WIRE_VERSION, 'metadata_version == wire version')
      call meta%free()
   end subroutine version_and_wire_constant

   ! ---- Task 2: doctor -----------------------------------------------------
   subroutine doctor_report()
      character(len=:), allocatable :: json
      integer(c_int) :: err
      json = tensogram_doctor(err)
      call assert(err == TGM_ERROR_OK, 'doctor: OK')
      call assert(len(json) > 0, 'doctor: non-empty')
      call assert(json(1:1) == '{', 'doctor: JSON object')
      call assert(has(json, '"build"'), 'doctor: has build')
      call assert(has(json, '"wire_version"'), 'doctor: has wire_version')
      call assert(has(json, '"self_test"'), 'doctor: has self_test')
      json = tensogram_doctor()                 ! callable with no err argument
      call assert(len(json) > 0, 'doctor(): non-empty (no err arg)')
   end subroutine doctor_report

   ! ---- Task 3: validate + validate_file -----------------------------------
   subroutine validate_buffer_and_file()
      character(len=*), parameter :: path = 'test_symmetry_validate.tgm'
      real(c_float)                  :: a(4)
      integer(c_int8_t), allocatable :: wire(:), empty(:)
      type(tensogram_buffer) :: buf
      type(tensogram_file)   :: f
      character(len=:), allocatable :: rep
      integer(c_int) :: err
      integer :: ios
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
      call tensogram_encode(a, buf, err); call assert(err == TGM_ERROR_OK, 'validate: encode')
      call buf%as_array(wire)

      rep = tensogram_validate(wire, err=err)
      call assert(err == TGM_ERROR_OK, 'validate: OK')
      call assert(has(rep, '"issues"'), 'validate: has issues')
      call assert(has(rep, '"object_count"'), 'validate: has object_count')
      call assert(has(rep, '"hash_verified"'), 'validate: has hash_verified')
      call assert(has(rep, '"issues":[]'), 'validate: clean message has no issues')

      rep = tensogram_validate(wire, level='full', check_canonical=.true.)
      call assert(has(rep, '"issues":[]'), 'validate full+canonical: clean')

      rep = tensogram_validate(wire, level='quick')
      call assert(has(rep, '"hash_verified":false'), 'validate quick: hash not verified')

      allocate(empty(0))
      rep = tensogram_validate(empty, err=err)
      call assert(err == TGM_ERROR_OK, 'validate empty: OK report')
      call assert(has(rep, '"buffer_too_short"'), 'validate empty: buffer_too_short')

      rep = tensogram_validate(wire, level='bogus', err=err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'validate bogus level -> INVALID_ARG')
      call assert(len(rep) == 0, 'validate bogus level -> empty report')

      call tensogram_file_create(path, f, err); call assert(err == TGM_ERROR_OK, 'validate_file: create')
      call tensogram_file_append(f, a, err);     call assert(err == TGM_ERROR_OK, 'validate_file: append')
      call f%close()
      rep = tensogram_validate_file(path, err=err)
      call assert(err == TGM_ERROR_OK, 'validate_file: OK')
      call assert(has(rep, '"file_issues"'), 'validate_file: has file_issues')
      call assert(has(rep, '"messages"'), 'validate_file: has messages')

      rep = tensogram_validate_file('/nonexistent/tensogram_missing_xyz.tgm', err=err)
      call assert(err == TGM_ERROR_IO, 'validate_file missing -> IO')
      call assert(len(rep) == 0, 'validate_file missing -> empty report')

      open(newunit=ios, file=path, status='old', iostat=err)
      if (err == 0) close(ios, status='delete')
   end subroutine validate_buffer_and_file

   ! ---- Task 4: scan -------------------------------------------------------
   subroutine scan_multi_message()
      real(c_float)      :: a(3)
      integer(c_int32_t) :: b(5)
      type(tensogram_buffer) :: bufa, bufb
      integer(c_int8_t), allocatable :: wa(:), wb(:), wire(:), seg(:)
      integer(c_size_t), allocatable :: offs(:), lens(:)
      type(tensogram_message)         :: msg
      real(c_float),      allocatable :: oa(:)
      integer(c_int32_t), allocatable :: ob(:)
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      b = [10_c_int32_t, 20_c_int32_t, 30_c_int32_t, 40_c_int32_t, 50_c_int32_t]
      call tensogram_encode(a, bufa, err); call assert(err == TGM_ERROR_OK, 'scan: encode A')
      call tensogram_encode(b, bufb, err); call assert(err == TGM_ERROR_OK, 'scan: encode B')
      call bufa%as_array(wa)
      call bufb%as_array(wb)
      wire = [wa, wb]                          ! two messages back-to-back

      call tensogram_scan(wire, offs, lens, err)
      call assert(err == TGM_ERROR_OK, 'scan: OK')
      call assert(size(offs) == 2, 'scan: two messages')
      call assert(size(lens) == 2, 'scan: two lengths')
      call assert(offs(1) == 1_c_size_t, 'scan: first offset (1-based) == 1')
      call assert(int(lens(1)) == size(wa), 'scan: first length == size(wa)')
      call assert(offs(2) == int(size(wa), c_size_t) + 1_c_size_t, 'scan: second offset follows first')
      call assert(int(lens(2)) == size(wb), 'scan: second length == size(wb)')

      ! The returned 1-based offset/length slices `wire` back into each message.
      seg = wire(offs(1) : offs(1) + lens(1) - 1_c_size_t)
      call tensogram_decode(seg, msg, err); call assert(err == TGM_ERROR_OK, 'scan: decode sliced message 1')
      call tensogram_to_array(msg, 1, oa, err)
      call assert(err == TGM_ERROR_OK, 'scan: to_array message 1')
      call assert(feq(oa, a), 'scan: message 1 values')

      seg = wire(offs(2) : offs(2) + lens(2) - 1_c_size_t)
      call tensogram_decode(seg, msg, err); call assert(err == TGM_ERROR_OK, 'scan: decode sliced message 2')
      call tensogram_to_array(msg, 1, ob, err)
      call assert(err == TGM_ERROR_OK, 'scan: to_array message 2')
      call assert(all(ob == b), 'scan: message 2 values')
   end subroutine scan_multi_message

   ! ---- Task 5: compute_hash + inline object-hash accessors ----------------
   subroutine hash_and_object_hash()
      real(c_float)                  :: a(4)
      integer(c_int8_t), allocatable :: data(:), wire(:)
      character(len=:), allocatable  :: hex, hex2
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float]
      data = transfer(a, [0_c_int8_t])
      hex = tensogram_compute_hash(data, err=err)
      call assert(err == TGM_ERROR_OK, 'compute_hash: OK')
      call assert(len(hex) == 16, 'compute_hash: xxh3-64 -> 16 hex chars')
      call assert(is_hex(hex), 'compute_hash: hex digits only')
      hex2 = tensogram_compute_hash(data, algo='xxh3')
      call assert(hex == hex2, 'compute_hash: deterministic')

      ! Encoded WITH a hash: the inline slot is populated.
      call tensogram_encode(a, buf, err, hash='xxh3'); call assert(err == TGM_ERROR_OK, 'hash: encode w/ hash')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err);           call assert(err == TGM_ERROR_OK, 'hash: decode w/ hash')
      call assert(tensogram_object_has_hash(msg, 1), 'object has_hash (hashed)')
      call assert(tensogram_object_hash_type(msg, 1) == 'xxh3', 'object hash_type == xxh3')
      call assert(len(tensogram_object_hash_value(msg, 1)) > 0, 'object hash_value non-empty')

      ! Encoded WITHOUT a hash: the slot is empty.
      call tensogram_encode(a, buf, err, hash=''); call assert(err == TGM_ERROR_OK, 'hash: encode no hash')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err);        call assert(err == TGM_ERROR_OK, 'hash: decode no hash')
      call assert(.not. tensogram_object_has_hash(msg, 1), 'object has no hash (hash="")')
      call assert(len(tensogram_object_hash_type(msg, 1)) == 0, 'no-hash: hash_type empty')
   end subroutine hash_and_object_hash

   ! ---- Task 6: object descriptor accessors --------------------------------
   subroutine object_descriptor_accessors()
      integer, parameter :: NI = 4, NJ = 3
      real(c_float)                  :: field(NI, NJ)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int64_t), allocatable :: strd(:)
      integer(c_int) :: err
      integer :: i, j
      do j = 1, NJ; do i = 1, NI; field(i, j) = real(i * 10 + j, c_float); end do; end do
      call tensogram_encode(field, buf, err, compression='zstd')
      call assert(err == TGM_ERROR_OK, 'accessors: encode (zstd)')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'accessors: decode')

      call assert(tensogram_object_type(msg, 1)        == 'ntensor',  'object_type == ntensor')
      call assert(tensogram_object_byte_order(msg, 1)  == host_bo(),  'object_byte_order == host')
      call assert(tensogram_object_dtype(msg, 1)       == 'float32',  'object_dtype == float32')
      call assert(tensogram_payload_encoding(msg, 1)   == 'none',     'payload_encoding == none')
      call assert(tensogram_object_filter(msg, 1)      == 'none',     'object_filter == none')
      call assert(tensogram_object_compression(msg, 1) == 'zstd',     'object_compression == zstd')

      ! Fortran-order element strides: [1, NI] for a contiguous (NI, NJ) array.
      strd = tensogram_object_strides(msg, 1)
      call assert(size(strd) == 2, 'strides: rank 2')
      call assert(strd(1) == 1_c_int64_t, 'strides(1) == 1 (fastest axis)')
      call assert(strd(2) == int(NI, c_int64_t), 'strides(2) == NI')

      ! Out-of-range indices degrade gracefully (empty string / empty array).
      call assert(len(tensogram_object_type(msg, 9)) == 0, 'oor object_type -> empty')
      call assert(len(tensogram_object_byte_order(msg, 9)) == 0, 'oor byte_order -> empty')
      call assert(size(tensogram_object_strides(msg, 9)) == 0, 'oor strides -> empty')
   end subroutine object_descriptor_accessors

   ! ---- Task 7: encode_pre_encoded -----------------------------------------
   subroutine pre_encoded_roundtrip()
      real(c_float)                  :: a(5)
      real(c_float), allocatable     :: oa(:)
      integer(c_int8_t), allocatable :: data(:), wire(:)
      integer(c_size_t)              :: lens1(1)
      character(len=:), allocatable  :: json
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float, 4.0_c_float, 5.0_c_float]

      ! Single object, encoding=none: pre-encoded bytes ARE the raw element bytes.
      json = '{"descriptors":[' // desc1d(5, host_bo()) // ']}'
      data = transfer(a, [0_c_int8_t])
      lens1(1) = int(size(data), c_size_t)
      call tensogram_encode_pre_encoded(json, data, lens1, buf, err)
      call assert(err == TGM_ERROR_OK, 'pre_encoded: encode single')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'pre_encoded: decode single')
      call assert(tensogram_num_objects(msg) == 1, 'pre_encoded: one object')
      call tensogram_to_array(msg, 1, oa, err)
      call assert(err == TGM_ERROR_OK, 'pre_encoded: to_array single')
      call assert(feq(oa, a), 'pre_encoded: single round-trip')

      ! Two objects concatenated in one flat byte buffer + per-object lengths.
      block
         real(c_float)                  :: c3(3), d2(2)
         real(c_float), allocatable     :: oc(:), od(:)
         integer(c_int8_t), allocatable :: data2(:)
         integer(c_size_t)              :: lens2(2)
         c3 = [1.0_c_float, 2.0_c_float, 3.0_c_float]
         d2 = [10.0_c_float, 20.0_c_float]
         json  = '{"descriptors":[' // desc1d(3, host_bo()) // ',' // desc1d(2, host_bo()) // ']}'
         data2 = [transfer(c3, [0_c_int8_t]), transfer(d2, [0_c_int8_t])]
         lens2(1) = int(3 * 4, c_size_t)       ! 3 float32 = 12 bytes
         lens2(2) = int(2 * 4, c_size_t)       ! 2 float32 =  8 bytes
         call tensogram_encode_pre_encoded(json, data2, lens2, buf, err)
         call assert(err == TGM_ERROR_OK, 'pre_encoded: encode two')
         call buf%as_array(wire)
         call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'pre_encoded: decode two')
         call assert(tensogram_num_objects(msg) == 2, 'pre_encoded: two objects')
         call tensogram_to_array(msg, 1, oc, err)
         call assert(err == TGM_ERROR_OK .and. feq(oc, c3), 'pre_encoded: object 1 values')
         call tensogram_to_array(msg, 2, od, err)
         call assert(err == TGM_ERROR_OK .and. feq(od, d2), 'pre_encoded: object 2 values')
      end block
   end subroutine pre_encoded_roundtrip

   ! ---- Task 8: decode_object / decode_metadata / decode_range -------------
   subroutine decode_variants()
      real(c_float)                  :: c3(3), d2(2)
      real(c_float), allocatable     :: oc(:), od(:)
      integer(c_int8_t), allocatable :: data2(:), wire(:)
      integer(c_size_t)              :: lens2(2)
      character(len=:), allocatable  :: json
      type(tensogram_buffer)   :: buf
      type(tensogram_message)  :: msg
      type(tensogram_metadata) :: meta
      integer(c_int) :: err
      c3 = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      d2 = [10.0_c_float, 20.0_c_float]
      json  = '{"descriptors":[' // desc1d(3, host_bo()) // ',' // desc1d(2, host_bo()) // ']}'
      data2 = [transfer(c3, [0_c_int8_t]), transfer(d2, [0_c_int8_t])]
      lens2(1) = int(3 * 4, c_size_t)
      lens2(2) = int(2 * 4, c_size_t)
      call tensogram_encode_pre_encoded(json, data2, lens2, buf, err)
      call assert(err == TGM_ERROR_OK, 'decode_variants: build 2-object buffer')
      call buf%as_array(wire)

      ! decode_object: 1-based index, message holds exactly that one object.
      call tensogram_decode_object(wire, 1, msg, err)
      call assert(err == TGM_ERROR_OK, 'decode_object 1: OK')
      call assert(tensogram_num_objects(msg) == 1, 'decode_object 1: one object')
      call tensogram_to_array(msg, 1, oc, err)
      call assert(err == TGM_ERROR_OK .and. feq(oc, c3), 'decode_object 1: values')
      call tensogram_decode_object(wire, 2, msg, err)
      call assert(err == TGM_ERROR_OK, 'decode_object 2: OK')
      call tensogram_to_array(msg, 1, od, err)
      call assert(err == TGM_ERROR_OK .and. feq(od, d2), 'decode_object 2: values')

      ! decode_metadata: global metadata only, no payloads read.
      call tensogram_decode_metadata(wire, meta, err)
      call assert(err == TGM_ERROR_OK, 'decode_metadata: OK')
      call assert(tensogram_metadata_num_objects(meta) == 2, 'decode_metadata: 2 objects')
      call assert(tensogram_metadata_version(meta) == TGM_WIRE_VERSION, 'decode_metadata: version')
      call meta%free()

      call decode_range_cases()
   end subroutine decode_variants

   subroutine decode_range_cases()
      real(c_float)                  :: e8(8)
      real(c_float), allocatable     :: rf(:)
      integer(c_int8_t), allocatable :: w(:), rb(:)
      integer(c_size_t), allocatable :: rl(:)
      integer(c_int64_t)             :: offs1(1), cnts1(1), offs2(2), cnts2(2)
      type(tensogram_buffer) :: buf
      integer(c_int) :: err
      integer :: k
      do k = 1, 8; e8(k) = real(k - 1, c_float); end do    ! 0.0 .. 7.0
      call tensogram_encode(e8, buf, err); call assert(err == TGM_ERROR_OK, 'decode_range: encode')
      call buf%as_array(w)

      ! Split, one range: 0-based elements [2, 3, 4].
      offs1(1) = 2_c_int64_t; cnts1(1) = 3_c_int64_t
      call tensogram_decode_range(w, 1, offs1, cnts1, rb, rl, err)
      call assert(err == TGM_ERROR_OK, 'decode_range split: OK')
      call assert(size(rl) == 1, 'decode_range split: one buffer')
      call assert(int(rl(1)) == 3 * 4, 'decode_range split: 12 bytes')
      rf = transfer(rb, [0.0_c_float], 3)
      call assert(feq(rf, [2.0_c_float, 3.0_c_float, 4.0_c_float]), 'decode_range split: values [2,3,4]')

      ! Join, two ranges [1,2] and [5,3] -> elements 1,2,5,6,7 concatenated.
      offs2 = [1_c_int64_t, 5_c_int64_t]; cnts2 = [2_c_int64_t, 3_c_int64_t]
      call tensogram_decode_range(w, 1, offs2, cnts2, rb, rl, err, join=.true.)
      call assert(err == TGM_ERROR_OK, 'decode_range join: OK')
      call assert(size(rl) == 1, 'decode_range join: single joined buffer')
      call assert(int(rl(1)) == 5 * 4, 'decode_range join: 20 bytes')
      rf = transfer(rb, [0.0_c_float], 5)
      call assert(feq(rf, [1.0_c_float, 2.0_c_float, 5.0_c_float, 6.0_c_float, 7.0_c_float]), &
                  'decode_range join: values')

      ! Split, two ranges -> one length entry per range.
      call tensogram_decode_range(w, 1, offs2, cnts2, rb, rl, err, join=.false.)
      call assert(err == TGM_ERROR_OK, 'decode_range split-2: OK')
      call assert(size(rl) == 2, 'decode_range split-2: two buffers')
      call assert(int(rl(1)) == 2 * 4 .and. int(rl(2)) == 3 * 4, 'decode_range split-2: per-range lengths')
   end subroutine decode_range_cases

   ! ---- helpers ------------------------------------------------------------

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_symmetry: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

   !> Substring test (JSON key presence).
   logical function has(hay, needle)
      character(len=*), intent(in) :: hay, needle
      has = index(hay, needle) > 0
   end function has

   !> Bit-exact equality of two float32 vectors (the correct test for a
   !> lossless round-trip; avoids the -Wcompare-reals real `==` warning).
   logical function feq(x, y)
      real(c_float), intent(in) :: x(:), y(:)
      feq = size(x) == size(y)
      if (feq) feq = all(transfer(x, [0_c_int32_t]) == transfer(y, [0_c_int32_t]))
   end function feq

   !> .true. when every character is a hex digit (and the string is non-empty).
   logical function is_hex(s)
      character(len=*), intent(in) :: s
      integer :: i, c
      is_hex = len(s) > 0
      do i = 1, len(s)
         c = iachar(s(i:i))
         if (.not. ((c >= iachar('0') .and. c <= iachar('9')) .or. &
                    (c >= iachar('a') .and. c <= iachar('f')) .or. &
                    (c >= iachar('A') .and. c <= iachar('F')))) then
            is_hex = .false.
            return
         end if
      end do
   end function is_hex

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

end program test_symmetry

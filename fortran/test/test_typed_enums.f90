! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Deferred Wave-B, component C — the typed enum surface:
!>   * TGM_DTYPE_* / TGM_BYTE_ORDER_*                  (tgm_dtype/tgm_byte_order)
!>   * TGM_AGGREGATE_HASH_POLICY_* / TGM_COMPRESSION_BACKEND_*
!>   * tensogram_object_dtype_enum                     (tgm_object_dtype_enum)
!>   * tensogram_object_byte_order_enum                (tgm_object_byte_order_enum)
!>
!> The parameter values themselves are guarded against the C header by
!> fortran/test/check_enum_mirror.sh; what a running test adds is that the
!> CODES agree with the STRING getters they are the typed companion of, for
!> every dtype this binding can put on the wire.
program test_typed_enums
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   integer :: npass

   npass = 0

   call dtype_constants()
   call byte_order_constants()
   call policy_constants()
   call dtype_enum_agrees_with_the_string_getter()
   call byte_order_enum_agrees_with_the_string_getter()
   call out_of_range_is_reported()

   print '(a,i0,a)', 'test_typed_enums: PASS (', npass, ' checks)'

contains

   ! ---- The mirrored C ABI numbers (frozen; the wire stores dtype as text) -
   subroutine dtype_constants()
      call assert(TGM_DTYPE_FLOAT16    ==  0, 'TGM_DTYPE_FLOAT16 = 0')
      call assert(TGM_DTYPE_BFLOAT16   ==  1, 'TGM_DTYPE_BFLOAT16 = 1')
      call assert(TGM_DTYPE_FLOAT32    ==  2, 'TGM_DTYPE_FLOAT32 = 2')
      call assert(TGM_DTYPE_FLOAT64    ==  3, 'TGM_DTYPE_FLOAT64 = 3')
      call assert(TGM_DTYPE_COMPLEX64  ==  4, 'TGM_DTYPE_COMPLEX64 = 4')
      call assert(TGM_DTYPE_COMPLEX128 ==  5, 'TGM_DTYPE_COMPLEX128 = 5')
      call assert(TGM_DTYPE_INT8       ==  6, 'TGM_DTYPE_INT8 = 6')
      call assert(TGM_DTYPE_INT16      ==  7, 'TGM_DTYPE_INT16 = 7')
      call assert(TGM_DTYPE_INT32      ==  8, 'TGM_DTYPE_INT32 = 8')
      call assert(TGM_DTYPE_INT64      ==  9, 'TGM_DTYPE_INT64 = 9')
      call assert(TGM_DTYPE_UINT8      == 10, 'TGM_DTYPE_UINT8 = 10')
      call assert(TGM_DTYPE_UINT16     == 11, 'TGM_DTYPE_UINT16 = 11')
      call assert(TGM_DTYPE_UINT32     == 12, 'TGM_DTYPE_UINT32 = 12')
      call assert(TGM_DTYPE_UINT64     == 13, 'TGM_DTYPE_UINT64 = 13')
      call assert(TGM_DTYPE_BITMASK    == 14, 'TGM_DTYPE_BITMASK = 14')
   end subroutine dtype_constants

   subroutine byte_order_constants()
      call assert(TGM_BYTE_ORDER_LITTLE == 0, 'TGM_BYTE_ORDER_LITTLE = 0')
      call assert(TGM_BYTE_ORDER_BIG    == 1, 'TGM_BYTE_ORDER_BIG = 1')
   end subroutine byte_order_constants

   subroutine policy_constants()
      call assert(TGM_AGGREGATE_HASH_POLICY_AUTO   == 0, 'AGGREGATE_HASH_POLICY_AUTO = 0')
      call assert(TGM_AGGREGATE_HASH_POLICY_NONE   == 1, 'AGGREGATE_HASH_POLICY_NONE = 1')
      call assert(TGM_AGGREGATE_HASH_POLICY_HEADER == 2, 'AGGREGATE_HASH_POLICY_HEADER = 2')
      call assert(TGM_AGGREGATE_HASH_POLICY_FOOTER == 3, 'AGGREGATE_HASH_POLICY_FOOTER = 3')
      call assert(TGM_AGGREGATE_HASH_POLICY_BOTH   == 4, 'AGGREGATE_HASH_POLICY_BOTH = 4')
      call assert(TGM_COMPRESSION_BACKEND_AUTO == 0, 'COMPRESSION_BACKEND_AUTO = 0')
      call assert(TGM_COMPRESSION_BACKEND_FFI  == 1, 'COMPRESSION_BACKEND_FFI = 1')
      call assert(TGM_COMPRESSION_BACKEND_PURE == 2, 'COMPRESSION_BACKEND_PURE = 2')
   end subroutine policy_constants

   ! ---- dtype_enum is the typed companion of the dtype string --------------
   subroutine dtype_enum_agrees_with_the_string_getter()
      character(len=8), parameter :: names(10) = &
         ['float32 ', 'float64 ', 'int8    ', 'int16   ', 'int32   ', &
          'int64   ', 'uint8   ', 'uint16  ', 'uint32  ', 'uint64  ']
      integer, parameter :: widths(10) = [4, 8, 1, 2, 4, 8, 1, 2, 4, 8]
      integer(c_int8_t), allocatable :: data(:), wire(:)
      integer(c_size_t), allocatable :: lens(:)
      character(len=:),  allocatable :: json
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err, code
      integer :: k

      json = '{"descriptors":['
      allocate(lens(size(names)), data(0))
      do k = 1, size(names)
         if (k > 1) json = json // ','
         json = json // desc1(trim(names(k)), widths(k))
         data = [data, spread(int(k, c_int8_t), 1, widths(k))]
         lens(k) = int(widths(k), c_size_t)
      end do
      json = json // ']}'

      call tensogram_encode_pre_encoded(json, data, lens, buf, err)
      call assert(err == TGM_ERROR_OK, 'dtype_enum: encode the dtype zoo')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err)
      call assert(err == TGM_ERROR_OK, 'dtype_enum: decode the dtype zoo')
      call assert(tensogram_num_objects(msg) == size(names), 'dtype_enum: every object decoded')

      do k = 1, size(names)
         code = tensogram_object_dtype_enum(msg, k, err)
         call assert(err == TGM_ERROR_OK, 'dtype_enum: in-range index is not an error')
         call assert(code == expected_dtype(tensogram_object_dtype(msg, k)), &
                     'dtype_enum agrees with the dtype string: ' // trim(names(k)))
      end do
   end subroutine dtype_enum_agrees_with_the_string_getter

   ! ---- byte_order_enum likewise -------------------------------------------
   subroutine byte_order_enum_agrees_with_the_string_getter()
      real(c_float) :: a(3)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err, code
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      call tensogram_encode(a, buf, err)
      call assert(err == TGM_ERROR_OK, 'byte_order_enum: encode')
      call buf%as_array(wire)
      ! native_byte_order = .false. keeps the descriptor's own byte order.
      call tensogram_decode(wire, msg, err, native_byte_order=.false.)
      call assert(err == TGM_ERROR_OK, 'byte_order_enum: decode')
      code = tensogram_object_byte_order_enum(msg, 1, err)
      call assert(err == TGM_ERROR_OK, 'byte_order_enum: in-range index is not an error')
      call assert(code == expected_byte_order(tensogram_object_byte_order(msg, 1)), &
                  'byte_order_enum agrees with the byte-order string')
      call assert(code == expected_byte_order(host_bo()), 'byte_order_enum reports the host order')
   end subroutine byte_order_enum_agrees_with_the_string_getter

   ! ---- Out of range: the zero variant is ambiguous, so `err` says so -----
   subroutine out_of_range_is_reported()
      real(c_float) :: a(2)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg, null_msg
      integer(c_int) :: err, code
      a = [1.0_c_float, 2.0_c_float]
      call tensogram_encode(a, buf, err)
      call assert(err == TGM_ERROR_OK, 'out of range: encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err)
      call assert(err == TGM_ERROR_OK, 'out of range: decode')

      code = tensogram_object_dtype_enum(msg, 2, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'dtype_enum past the end -> INVALID_ARG')
      call assert(code == TGM_DTYPE_FLOAT16, 'dtype_enum past the end -> zero variant')
      call assert(len(tensogram_object_dtype(msg, 2)) == 0, 'the string getter agrees (empty)')

      code = tensogram_object_dtype_enum(msg, 0, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'dtype_enum index 0 -> INVALID_ARG')
      code = tensogram_object_byte_order_enum(msg, 2, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'byte_order_enum past the end -> INVALID_ARG')
      call assert(code == TGM_BYTE_ORDER_LITTLE, 'byte_order_enum past the end -> zero variant')

      ! A null (unassigned) handle has no objects at all.
      code = tensogram_object_dtype_enum(null_msg, 1, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'dtype_enum on a null handle -> INVALID_ARG')
      code = tensogram_object_byte_order_enum(null_msg, 1, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'byte_order_enum on a null handle -> INVALID_ARG')

      ! `err` is optional: the accessors are usable in an expression.
      call assert(tensogram_object_dtype_enum(msg, 1) == TGM_DTYPE_FLOAT32, 'dtype_enum without err')
      call assert(tensogram_object_byte_order_enum(msg, 1) == expected_byte_order(host_bo()), &
                  'byte_order_enum without err')
   end subroutine out_of_range_is_reported

   ! ---- helpers ------------------------------------------------------------

   !> The TGM_DTYPE_* code a dtype string must map to (the test's own table,
   !> so an accessor that silently returned the wrong variant is caught).
   integer(c_int) function expected_dtype(name)
      character(len=*), intent(in) :: name
      select case (name)
      case ('float16');    expected_dtype = TGM_DTYPE_FLOAT16
      case ('bfloat16');   expected_dtype = TGM_DTYPE_BFLOAT16
      case ('float32');    expected_dtype = TGM_DTYPE_FLOAT32
      case ('float64');    expected_dtype = TGM_DTYPE_FLOAT64
      case ('complex64');  expected_dtype = TGM_DTYPE_COMPLEX64
      case ('complex128'); expected_dtype = TGM_DTYPE_COMPLEX128
      case ('int8');       expected_dtype = TGM_DTYPE_INT8
      case ('int16');      expected_dtype = TGM_DTYPE_INT16
      case ('int32');      expected_dtype = TGM_DTYPE_INT32
      case ('int64');      expected_dtype = TGM_DTYPE_INT64
      case ('uint8');      expected_dtype = TGM_DTYPE_UINT8
      case ('uint16');     expected_dtype = TGM_DTYPE_UINT16
      case ('uint32');     expected_dtype = TGM_DTYPE_UINT32
      case ('uint64');     expected_dtype = TGM_DTYPE_UINT64
      case ('bitmask');    expected_dtype = TGM_DTYPE_BITMASK
      case default
         print '(a,a)', 'test_typed_enums: FAIL: unknown dtype string ', name
         error stop 1
      end select
   end function expected_dtype

   integer(c_int) function expected_byte_order(name)
      character(len=*), intent(in) :: name
      select case (name)
      case ('little'); expected_byte_order = TGM_BYTE_ORDER_LITTLE
      case ('big');    expected_byte_order = TGM_BYTE_ORDER_BIG
      case default
         print '(a,a)', 'test_typed_enums: FAIL: unknown byte order string ', name
         error stop 1
      end select
   end function expected_byte_order

   !> A single-element 1-D descriptor of `dtype` (`width` bytes per element,
   !> no encoding pipeline, so the pre-encoded bytes ARE the element bytes).
   function desc1(dtype, width) result(s)
      character(len=*), intent(in) :: dtype
      integer,          intent(in) :: width
      character(len=:), allocatable :: s
      character(len=32) :: ws
      write (ws, '(i0)') width
      s = '{"type":"ndarray","ndim":1,"shape":[1],"strides":[' // trim(ws) //   &
          '],"dtype":"' // dtype // '","byte_order":"' // host_bo() //          &
          '","encoding":"none","filter":"none","compression":"none"}'
   end function desc1

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

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_typed_enums: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

end program test_typed_enums

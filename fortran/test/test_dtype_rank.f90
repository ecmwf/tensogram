! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Generic encode/decode across a (dtype x rank) matrix: real32/real64 and
!> int32/int64 over ranks 1, 2 (non-contiguous), 3, 4, and scalar (rank 0).
!> Each case round-trips bit-identically and preserves the Fortran shape.
!> Also checks that a dtype mismatch on decode returns TGM_ERROR_OBJECT.
program test_dtype_rank
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   call rt_f32_r1()
   call rt_f64_r3()
   call rt_i32_r2_noncontig()
   call rt_i64_r4()
   call rt_f64_scalar()
   call dtype_mismatch()

   print '(a)', 'test_dtype_rank: PASS'

contains

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_dtype_rank: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   !> Byte-exact equality of two same-shape arrays of any numeric type,
   !> compared through their bit patterns (lint-clean, ±0.0/NaN aware).
   logical function bytes_eq(a, b)
      integer(c_int8_t), intent(in) :: a(:), b(:)
      bytes_eq = size(a) == size(b)
      if (bytes_eq) bytes_eq = all(a == b)
   end function bytes_eq

   subroutine rt_f32_r1()
      real(c_float)              :: x(7)
      real(c_float), allocatable :: y(:)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      integer :: i
      do i = 1, size(x); x(i) = real(i, c_float) * 1.5_c_float - 0.25_c_float; end do
      call tensogram_encode(x, buf, err);    call assert(err == TGM_ERROR_OK, 'f32_r1 encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'f32_r1 decode')
      call assert(tensogram_object_dtype(msg, 1) == 'float32', 'f32_r1 dtype')
      call tensogram_to_array(msg, 1, y, err); call assert(err == TGM_ERROR_OK, 'f32_r1 to_array')
      call assert(size(y) == size(x), 'f32_r1 shape')
      call assert(bytes_eq(transfer(y, [0_c_int8_t]), transfer(x, [0_c_int8_t])), 'f32_r1 bytes')
   end subroutine rt_f32_r1

   subroutine rt_f64_r3()
      real(c_double)              :: x(2, 3, 4)
      real(c_double), allocatable :: y(:,:,:)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      integer :: i, j, k
      do k = 1, 4; do j = 1, 3; do i = 1, 2
         x(i, j, k) = real(i * 100 + j * 10 + k, c_double)
      end do; end do; end do
      call tensogram_encode(x, buf, err);    call assert(err == TGM_ERROR_OK, 'f64_r3 encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'f64_r3 decode')
      call tensogram_to_array(msg, 1, y, err); call assert(err == TGM_ERROR_OK, 'f64_r3 to_array')
      call assert(all(shape(y) == shape(x)), 'f64_r3 shape')
      call assert(bytes_eq(transfer(y, [0_c_int8_t]), transfer(x, [0_c_int8_t])), 'f64_r3 bytes')
   end subroutine rt_f64_r3

   !> Non-contiguous input: a strided slice forces the compiler to gather into
   !> a contiguous temporary for the `contiguous` dummy, so c_loc stays valid.
   subroutine rt_i32_r2_noncontig()
      integer(c_int32_t)              :: big(4, 6)
      integer(c_int32_t), allocatable :: y(:,:)
      integer(c_int8_t),  allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      integer :: i, j
      do j = 1, 6; do i = 1, 4; big(i, j) = i * 1000 + j; end do; end do
      ! Every other row -> shape (2, 6), non-contiguous in memory.
      call tensogram_encode(big(1:4:2, :), buf, err); call assert(err == TGM_ERROR_OK, 'i32 encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'i32 decode')
      call tensogram_to_array(msg, 1, y, err); call assert(err == TGM_ERROR_OK, 'i32 to_array')
      call assert(all(shape(y) == [2, 6]), 'i32 shape')
      call assert(all(y == big(1:4:2, :)), 'i32 values (non-contiguous gather)')
   end subroutine rt_i32_r2_noncontig

   subroutine rt_i64_r4()
      integer(c_int64_t)              :: x(2, 2, 2, 3)
      integer(c_int64_t), allocatable :: y(:,:,:,:)
      integer(c_int8_t),  allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      integer :: i, j, k, l
      do l = 1, 3; do k = 1, 2; do j = 1, 2; do i = 1, 2
         x(i, j, k, l) = int(i, c_int64_t) + 10_c_int64_t * j + 100_c_int64_t * k + 1000_c_int64_t * l
      end do; end do; end do; end do
      call tensogram_encode(x, buf, err);    call assert(err == TGM_ERROR_OK, 'i64_r4 encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'i64_r4 decode')
      call tensogram_to_array(msg, 1, y, err); call assert(err == TGM_ERROR_OK, 'i64_r4 to_array')
      call assert(all(shape(y) == shape(x)), 'i64_r4 shape')
      call assert(all(y == x), 'i64_r4 values')
   end subroutine rt_i64_r4

   subroutine rt_f64_scalar()
      real(c_double)              :: x
      real(c_double), allocatable :: y
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      x = 3.14159265358979_c_double
      call tensogram_encode(x, buf, err);    call assert(err == TGM_ERROR_OK, 'scalar encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'scalar decode')
      call assert(tensogram_object_ndim(msg, 1) == 0, 'scalar ndim 0')
      call tensogram_to_array(msg, 1, y, err); call assert(err == TGM_ERROR_OK, 'scalar to_array')
      call assert(bytes_eq(transfer(y, [0_c_int8_t]), transfer(x, [0_c_int8_t])), 'scalar bytes')
   end subroutine rt_f64_scalar

   !> Decoding into the wrong dtype must report TGM_ERROR_OBJECT, not corrupt.
   subroutine dtype_mismatch()
      real(c_float)              :: x(3)
      integer(c_int32_t), allocatable :: y(:)
      integer(c_int8_t),  allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      x = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      call tensogram_encode(x, buf, err);    call assert(err == TGM_ERROR_OK, 'mismatch encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'mismatch decode')
      call tensogram_to_array(msg, 1, y, err)   ! int32 out for a float32 object
      call assert(err == TGM_ERROR_OBJECT, 'dtype mismatch -> OBJECT error')
   end subroutine dtype_mismatch

end program test_dtype_rank

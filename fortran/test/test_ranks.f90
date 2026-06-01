! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Exercise the per-rank decode (`select rank`) branches across the high ranks
!> (0, 3..7) that the functional tests do not otherwise reach, for both a real
!> and an integer dtype. Each round-trips bit-identically and preserves shape.
program test_ranks
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   call ranks_f32()
   call ranks_i64()
   call ranks_f64()
   call ranks_i32()

   print '(a)', 'test_ranks: PASS'

contains

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_ranks: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   logical function beq(x, y)
      integer(c_int8_t), intent(in) :: x(:), y(:)
      beq = size(x) == size(y)
      if (beq) beq = all(x == y)
   end function beq

   subroutine ranks_f32()
      real(c_float)              :: s0
      real(c_float)              :: a3(2,1,1), a4(2,1,1,1), a5(2,1,1,1,1)
      real(c_float)              :: a6(2,1,1,1,1,1), a7(2,1,1,1,1,1,1)
      real(c_float), allocatable :: o0
      real(c_float), allocatable :: o3(:,:,:), o4(:,:,:,:), o5(:,:,:,:,:)
      real(c_float), allocatable :: o6(:,:,:,:,:,:), o7(:,:,:,:,:,:,:)
      integer(c_int8_t), allocatable :: w(:)
      type(tensogram_buffer)  :: b
      type(tensogram_message) :: m
      integer(c_int) :: e

      s0 = 42.5_c_float
      call tensogram_encode(s0, b, e); call assert(e == TGM_ERROR_OK, 'f32 r0 enc')
      call b%as_array(w); call tensogram_decode(w, m, e); call assert(e == TGM_ERROR_OK, 'f32 r0 dec')
      call tensogram_to_array(m, 1, o0, e); call assert(e == TGM_ERROR_OK, 'f32 r0 ta')
      call assert(beq(transfer(o0, [0_c_int8_t]), transfer(s0, [0_c_int8_t])), 'f32 r0 bytes')

      a3 = reshape([1.0_c_float, 2.0_c_float], shape(a3))
      call tensogram_encode(a3, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o3, e); call assert(e == TGM_ERROR_OK, 'f32 r3 ta')
      call assert(all(shape(o3) == shape(a3)) .and. beq(transfer(o3, [0_c_int8_t]), transfer(a3, [0_c_int8_t])), 'f32 r3')

      a4 = reshape([1.0_c_float, 2.0_c_float], shape(a4))
      call tensogram_encode(a4, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o4, e); call assert(e == TGM_ERROR_OK, 'f32 r4 ta')
      call assert(all(shape(o4) == shape(a4)) .and. beq(transfer(o4, [0_c_int8_t]), transfer(a4, [0_c_int8_t])), 'f32 r4')

      a5 = reshape([1.0_c_float, 2.0_c_float], shape(a5))
      call tensogram_encode(a5, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o5, e); call assert(e == TGM_ERROR_OK, 'f32 r5 ta')
      call assert(all(shape(o5) == shape(a5)) .and. beq(transfer(o5, [0_c_int8_t]), transfer(a5, [0_c_int8_t])), 'f32 r5')

      a6 = reshape([1.0_c_float, 2.0_c_float], shape(a6))
      call tensogram_encode(a6, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o6, e); call assert(e == TGM_ERROR_OK, 'f32 r6 ta')
      call assert(all(shape(o6) == shape(a6)) .and. beq(transfer(o6, [0_c_int8_t]), transfer(a6, [0_c_int8_t])), 'f32 r6')

      a7 = reshape([1.0_c_float, 2.0_c_float], shape(a7))
      call tensogram_encode(a7, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o7, e); call assert(e == TGM_ERROR_OK, 'f32 r7 ta')
      call assert(all(shape(o7) == shape(a7)) .and. beq(transfer(o7, [0_c_int8_t]), transfer(a7, [0_c_int8_t])), 'f32 r7')
   end subroutine ranks_f32

   subroutine ranks_i64()
      integer(c_int64_t)              :: a0
      integer(c_int64_t)              :: a3(2,1,1), a4(2,1,1,1), a5(2,1,1,1,1)
      integer(c_int64_t)              :: a6(2,1,1,1,1,1), a7(2,1,1,1,1,1,1)
      integer(c_int64_t), allocatable :: o0
      integer(c_int64_t), allocatable :: o3(:,:,:), o4(:,:,:,:), o5(:,:,:,:,:)
      integer(c_int64_t), allocatable :: o6(:,:,:,:,:,:), o7(:,:,:,:,:,:,:)
      integer(c_int8_t), allocatable :: w(:)
      type(tensogram_buffer)  :: b
      type(tensogram_message) :: m
      integer(c_int) :: e

      a0 = 123456789_c_int64_t
      call tensogram_encode(a0, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o0, e); call assert(e == TGM_ERROR_OK, 'i64 r0 ta')
      call assert(o0 == a0, 'i64 r0')

      a3 = reshape([10_c_int64_t, 20_c_int64_t], shape(a3))
      call tensogram_encode(a3, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o3, e); call assert(e == TGM_ERROR_OK, 'i64 r3 ta')
      call assert(all(shape(o3) == shape(a3)) .and. all(o3 == a3), 'i64 r3')

      a4 = reshape([10_c_int64_t, 20_c_int64_t], shape(a4))
      call tensogram_encode(a4, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o4, e); call assert(e == TGM_ERROR_OK, 'i64 r4 ta')
      call assert(all(shape(o4) == shape(a4)) .and. all(o4 == a4), 'i64 r4')

      a5 = reshape([10_c_int64_t, 20_c_int64_t], shape(a5))
      call tensogram_encode(a5, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o5, e); call assert(e == TGM_ERROR_OK, 'i64 r5 ta')
      call assert(all(shape(o5) == shape(a5)) .and. all(o5 == a5), 'i64 r5')

      a6 = reshape([10_c_int64_t, 20_c_int64_t], shape(a6))
      call tensogram_encode(a6, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o6, e); call assert(e == TGM_ERROR_OK, 'i64 r6 ta')
      call assert(all(shape(o6) == shape(a6)) .and. all(o6 == a6), 'i64 r6')

      a7 = reshape([10_c_int64_t, 20_c_int64_t], shape(a7))
      call tensogram_encode(a7, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o7, e); call assert(e == TGM_ERROR_OK, 'i64 r7 ta')
      call assert(all(shape(o7) == shape(a7)) .and. all(o7 == a7), 'i64 r7')
   end subroutine ranks_i64

   subroutine ranks_f64()
      real(c_double)              :: a4(2,1,1,1), a5(2,1,1,1,1)
      real(c_double)              :: a6(2,1,1,1,1,1), a7(2,1,1,1,1,1,1)
      real(c_double), allocatable :: o4(:,:,:,:), o5(:,:,:,:,:)
      real(c_double), allocatable :: o6(:,:,:,:,:,:), o7(:,:,:,:,:,:,:)
      integer(c_int8_t), allocatable :: w(:)
      type(tensogram_buffer)  :: b
      type(tensogram_message) :: m
      integer(c_int) :: e

      a4 = reshape([1.5_c_double, 2.5_c_double], shape(a4))
      call tensogram_encode(a4, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o4, e); call assert(e == TGM_ERROR_OK, 'f64 r4 ta')
      call assert(all(shape(o4) == shape(a4)) .and. beq(transfer(o4, [0_c_int8_t]), transfer(a4, [0_c_int8_t])), 'f64 r4')

      a5 = reshape([1.5_c_double, 2.5_c_double], shape(a5))
      call tensogram_encode(a5, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o5, e); call assert(e == TGM_ERROR_OK, 'f64 r5 ta')
      call assert(all(shape(o5) == shape(a5)) .and. beq(transfer(o5, [0_c_int8_t]), transfer(a5, [0_c_int8_t])), 'f64 r5')

      a6 = reshape([1.5_c_double, 2.5_c_double], shape(a6))
      call tensogram_encode(a6, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o6, e); call assert(e == TGM_ERROR_OK, 'f64 r6 ta')
      call assert(all(shape(o6) == shape(a6)) .and. beq(transfer(o6, [0_c_int8_t]), transfer(a6, [0_c_int8_t])), 'f64 r6')

      a7 = reshape([1.5_c_double, 2.5_c_double], shape(a7))
      call tensogram_encode(a7, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o7, e); call assert(e == TGM_ERROR_OK, 'f64 r7 ta')
      call assert(all(shape(o7) == shape(a7)) .and. beq(transfer(o7, [0_c_int8_t]), transfer(a7, [0_c_int8_t])), 'f64 r7')
   end subroutine ranks_f64

   subroutine ranks_i32()
      integer(c_int32_t)              :: a0
      integer(c_int32_t)              :: a4(2,1,1,1), a5(2,1,1,1,1)
      integer(c_int32_t)              :: a6(2,1,1,1,1,1), a7(2,1,1,1,1,1,1)
      integer(c_int32_t), allocatable :: o0
      integer(c_int32_t), allocatable :: o4(:,:,:,:), o5(:,:,:,:,:)
      integer(c_int32_t), allocatable :: o6(:,:,:,:,:,:), o7(:,:,:,:,:,:,:)
      integer(c_int8_t), allocatable :: w(:)
      type(tensogram_buffer)  :: b
      type(tensogram_message) :: m
      integer(c_int) :: e

      a0 = 7777_c_int32_t
      call tensogram_encode(a0, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o0, e); call assert(e == TGM_ERROR_OK, 'i32 r0 ta')
      call assert(o0 == a0, 'i32 r0')

      a4 = reshape([3_c_int32_t, 6_c_int32_t], shape(a4))
      call tensogram_encode(a4, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o4, e); call assert(e == TGM_ERROR_OK, 'i32 r4 ta')
      call assert(all(shape(o4) == shape(a4)) .and. all(o4 == a4), 'i32 r4')

      a5 = reshape([3_c_int32_t, 6_c_int32_t], shape(a5))
      call tensogram_encode(a5, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o5, e); call assert(e == TGM_ERROR_OK, 'i32 r5 ta')
      call assert(all(shape(o5) == shape(a5)) .and. all(o5 == a5), 'i32 r5')

      a6 = reshape([3_c_int32_t, 6_c_int32_t], shape(a6))
      call tensogram_encode(a6, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o6, e); call assert(e == TGM_ERROR_OK, 'i32 r6 ta')
      call assert(all(shape(o6) == shape(a6)) .and. all(o6 == a6), 'i32 r6')

      a7 = reshape([3_c_int32_t, 6_c_int32_t], shape(a7))
      call tensogram_encode(a7, b, e); call b%as_array(w); call tensogram_decode(w, m, e)
      call tensogram_to_array(m, 1, o7, e); call assert(e == TGM_ERROR_OK, 'i32 r7 ta')
      call assert(all(shape(o7) == shape(a7)) .and. all(o7 == a7), 'i32 r7')
   end subroutine ranks_i32

end program test_ranks

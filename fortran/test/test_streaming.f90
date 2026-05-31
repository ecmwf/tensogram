! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Streaming encoder: write a single multi-object message progressively (one
!> data object at a time, mixed dtype/rank, with lossless compression on one),
!> finish, then reopen and decode all objects and assert a bit-identical
!> round-trip. PLAN_FORTRAN.md streaming milestone.
program test_streaming
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'test_streaming_tmp.tgm'
   type(tensogram_streaming_encoder) :: enc
   type(tensogram_file)              :: f
   type(tensogram_message)           :: msg
   integer(c_int) :: err
   integer :: i, j, k, n, ios

   real(c_float)      :: a1(3, 4)
   real(c_double)     :: a2(5)
   integer(c_int32_t) :: a3(2, 2, 2)
   real(c_float),      allocatable :: o1(:,:)
   real(c_double),     allocatable :: o2(:)
   integer(c_int32_t), allocatable :: o3(:,:,:)

   do j = 1, 4; do i = 1, 3; a1(i, j) = real(i, c_float) + 0.1_c_float * real(j, c_float); end do; end do
   do i = 1, 5; a2(i) = real(i, c_double) * 2.5_c_double; end do
   do k = 1, 2; do j = 1, 2; do i = 1, 2; a3(i, j, k) = i * 100 + j * 10 + k; end do; end do; end do

   ! Stream three objects of different dtype/rank, one at a time.
   call tensogram_streaming_encoder_create(path, enc, err); call assert(err == TGM_ERROR_OK, 'create')
   call tensogram_streaming_encoder_write(enc, a1, err, compression='zstd')  ! lossless
   call assert(err == TGM_ERROR_OK, 'write obj1 (zstd)')
   call assert(tensogram_streaming_encoder_count(enc) == 1, 'count == 1')
   call tensogram_streaming_encoder_write(enc, a2, err); call assert(err == TGM_ERROR_OK, 'write obj2')
   call tensogram_streaming_encoder_write(enc, a3, err); call assert(err == TGM_ERROR_OK, 'write obj3')
   call assert(tensogram_streaming_encoder_count(enc) == 3, 'count == 3')
   call tensogram_streaming_encoder_finish(enc, err);    call assert(err == TGM_ERROR_OK, 'finish')
   call enc%free()

   ! Reopen the finished file: one message, three objects, all round-trip.
   call tensogram_file_open(path, f, err);            call assert(err == TGM_ERROR_OK, 'open')
   call tensogram_file_message_count(f, n, err);      call assert(err == TGM_ERROR_OK, 'count err')
   call assert(n == 1, 'one message')
   call tensogram_file_decode_message(f, 1, msg, err); call assert(err == TGM_ERROR_OK, 'decode')
   call assert(tensogram_num_objects(msg) == 3, 'three objects')

   call tensogram_to_array(msg, 1, o1, err); call assert(err == TGM_ERROR_OK, 'to_array 1')
   call assert(all(shape(o1) == [3, 4]), 'obj1 shape')
   call assert(bytes_eq(transfer(o1, [0_c_int8_t]), transfer(a1, [0_c_int8_t])), 'obj1 values (zstd lossless)')

   call tensogram_to_array(msg, 2, o2, err); call assert(err == TGM_ERROR_OK, 'to_array 2')
   call assert(size(o2) == 5, 'obj2 size')
   call assert(bytes_eq(transfer(o2, [0_c_int8_t]), transfer(a2, [0_c_int8_t])), 'obj2 values')

   call tensogram_to_array(msg, 3, o3, err); call assert(err == TGM_ERROR_OK, 'to_array 3')
   call assert(all(shape(o3) == [2, 2, 2]), 'obj3 shape')
   call assert(all(o3 == a3), 'obj3 values')

   call f%close()
   open(newunit=ios, file=path, status='old', iostat=err)
   if (err == 0) close(ios, status='delete')

   print '(a)', 'test_streaming: PASS'

contains

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_streaming: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   logical function bytes_eq(x, y)
      integer(c_int8_t), intent(in) :: x(:), y(:)
      bytes_eq = size(x) == size(y)
      if (bytes_eq) bytes_eq = all(x == y)
   end function bytes_eq

end program test_streaming

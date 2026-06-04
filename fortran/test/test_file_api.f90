! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> File API: create a multi-message .tgm, append messages of different
!> dtypes/ranks, reopen, count them, and decode each by index (random access).
!> Also exercises read_message (raw bytes) -> decode.
program test_file_api
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'test_file_api_tmp.tgm'
   type(tensogram_file)    :: f
   type(tensogram_message) :: msg
   integer(c_int) :: err
   integer :: n, ios, i, j

   real(c_float)      :: a1(5)
   real(c_double)     :: a2(3, 2)
   integer(c_int32_t) :: a3(4)
   integer(c_int64_t) :: a4(3)
   real(c_float),      allocatable :: o1(:)
   real(c_double),     allocatable :: o2(:,:)
   integer(c_int32_t), allocatable :: o3(:)
   integer(c_int64_t), allocatable :: o4(:)

   do i = 1, 5; a1(i) = real(i, c_float) * 0.5_c_float; end do
   do j = 1, 2; do i = 1, 3; a2(i, j) = real(i * 10 + j, c_double); end do; end do
   a3 = [11_c_int32_t, 22_c_int32_t, 33_c_int32_t, 44_c_int32_t]
   a4 = [100000_c_int64_t, 200000_c_int64_t, 300000_c_int64_t]

   ! Create and append four messages (mixed dtype/rank).
   call tensogram_file_create(path, f, err); call assert(err == TGM_ERROR_OK, 'create')
   call tensogram_file_append(f, a1, err);   call assert(err == TGM_ERROR_OK, 'append a1')
   call tensogram_file_append(f, a2, err);   call assert(err == TGM_ERROR_OK, 'append a2')
   call tensogram_file_append(f, a3, err);   call assert(err == TGM_ERROR_OK, 'append a3')
   call tensogram_file_append(f, a4, err);   call assert(err == TGM_ERROR_OK, 'append a4 (int64)')
   call f%close()

   ! Reopen and count.
   call tensogram_file_open(path, f, err);          call assert(err == TGM_ERROR_OK, 'open')
   call tensogram_file_message_count(f, n, err);    call assert(err == TGM_ERROR_OK, 'count')
   call assert(n == 4, 'message_count == 4')

   ! Random-access decode by index (1-based; explicit decode options here).
   call tensogram_file_decode_message(f, 1, msg, err, verify_hash=.false., native_byte_order=.true.)
   call assert(err == TGM_ERROR_OK, 'decode 1')
   call tensogram_to_array(msg, 1, o1, err);           call assert(err == TGM_ERROR_OK, 'to_array 1')
   call assert(size(o1) == 5, 'msg1 size')
   call assert(bytes_eq(transfer(o1, [0_c_int8_t]), transfer(a1, [0_c_int8_t])), 'msg1 values')

   call tensogram_file_decode_message(f, 2, msg, err); call assert(err == TGM_ERROR_OK, 'decode 2')
   call tensogram_to_array(msg, 1, o2, err);           call assert(err == TGM_ERROR_OK, 'to_array 2')
   call assert(all(shape(o2) == [3, 2]), 'msg2 shape')
   call assert(bytes_eq(transfer(o2, [0_c_int8_t]), transfer(a2, [0_c_int8_t])), 'msg2 values')

   call tensogram_file_decode_message(f, 3, msg, err); call assert(err == TGM_ERROR_OK, 'decode 3')
   call tensogram_to_array(msg, 1, o3, err);           call assert(err == TGM_ERROR_OK, 'to_array 3')
   call assert(all(o3 == a3), 'msg3 values')

   call tensogram_file_decode_message(f, 4, msg, err); call assert(err == TGM_ERROR_OK, 'decode 4')
   call tensogram_to_array(msg, 1, o4, err);           call assert(err == TGM_ERROR_OK, 'to_array 4')
   call assert(all(o4 == a4), 'msg4 values (int64)')

   ! Raw-bytes path: read_message -> decode.
   call read_back_first()

   call f%close()

   ! Clean up the temp file.
   open(newunit=ios, file=path, status='old', iostat=err)
   if (err == 0) close(ios, status='delete')

   print '(a)', 'test_file_api: PASS'

contains

   subroutine read_back_first()
      type(tensogram_buffer)  :: raw
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_message) :: m2
      real(c_float), allocatable :: r1(:)
      integer(c_int) :: e
      call tensogram_file_read_message(f, 1, raw, e); call assert(e == TGM_ERROR_OK, 'read_message 1')
      call assert(raw%size() > 0_c_size_t, 'read bytes non-empty')
      call raw%as_array(wire)
      call tensogram_decode(wire, m2, e);             call assert(e == TGM_ERROR_OK, 'decode read bytes')
      call tensogram_to_array(m2, 1, r1, e);          call assert(e == TGM_ERROR_OK, 'to_array read')
      call assert(bytes_eq(transfer(r1, [0_c_int8_t]), transfer(a1, [0_c_int8_t])), 'read_message values')
   end subroutine read_back_first

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_file_api: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   logical function bytes_eq(x, y)
      integer(c_int8_t), intent(in) :: x(:), y(:)
      bytes_eq = size(x) == size(y)
      if (bytes_eq) bytes_eq = all(x == y)
   end function bytes_eq

end program test_file_api

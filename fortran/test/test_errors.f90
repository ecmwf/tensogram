! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Error-path behaviour: failures return a specific non-OK `tgm_error` code
!> (never abort) and populate the thread-local `tensogram_last_error()`
!> string. Covers framing (garbage decode), encoding (non-finite reject), I/O
!> (missing file), and object (dtype mismatch) categories, plus the static
!> `tensogram_strerror` descriptions.
program test_errors
   use, intrinsic :: iso_c_binding, only : c_int8_t, c_int, c_int64_t, c_float
   use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan, ieee_positive_inf
   use tensogram
   implicit none

   integer(c_int8_t) :: garbage(32)
   real(c_float)     :: a(3)
   integer(c_int8_t), allocatable :: wire(:)
   type(tensogram_buffer)  :: buf
   type(tensogram_message) :: msg
   type(tensogram_file)    :: f
   integer(c_int) :: err
   integer :: i

   ! Framing: not a TENSOGRM message -> decode fails gracefully.
   do i = 1, size(garbage)
      garbage(i) = int(mod(i * 37, 251), c_int8_t)
   end do
   call tensogram_decode(garbage, msg, err)
   call assert(err == TGM_ERROR_FRAMING, 'garbage decode -> FRAMING')
   call assert(len(tensogram_last_error()) > 0, 'framing sets last_error')

   ! Encoding: non-finite values are rejected by default.
   a = [1.0_c_float, ieee_value(0.0_c_float, ieee_quiet_nan), 3.0_c_float]
   call tensogram_encode(a, buf, err)
   call assert(err == TGM_ERROR_ENCODING, 'NaN encode -> ENCODING')
   call assert(len(tensogram_last_error()) > 0, 'NaN sets last_error')
   a = [1.0_c_float, ieee_value(0.0_c_float, ieee_positive_inf), 3.0_c_float]
   call tensogram_encode(a, buf, err)
   call assert(err == TGM_ERROR_ENCODING, '+Inf encode -> ENCODING')

   ! I/O: opening a non-existent file reports an I/O error.
   call tensogram_file_open('/nonexistent/tensogram_missing_xyz.tgm', f, err)
   call assert(err == TGM_ERROR_IO, 'open missing file -> IO')
   call assert(len(tensogram_last_error()) > 0, 'IO sets last_error')

   ! Object: decoding into the wrong dtype reports an object error.
   a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
   call tensogram_encode(a, buf, err);                call assert(err == TGM_ERROR_OK, 'encode for mismatch')
   call buf%as_array(wire)
   call tensogram_decode(wire, msg, err);             call assert(err == TGM_ERROR_OK, 'decode for mismatch')
   call wrong_dtype(msg, err)
   call assert(err == TGM_ERROR_OBJECT, 'wrong-dtype to_array -> OBJECT')

   ! Static descriptions exist for every known code.
   call assert(len(tensogram_strerror(TGM_ERROR_FRAMING))  > 0, 'strerror FRAMING')
   call assert(len(tensogram_strerror(TGM_ERROR_ENCODING)) > 0, 'strerror ENCODING')
   call assert(len(tensogram_strerror(TGM_ERROR_IO))       > 0, 'strerror IO')

   print '(a)', 'test_errors: PASS'

contains

   subroutine wrong_dtype(m, e)
      type(tensogram_message), intent(in)  :: m
      integer(c_int),          intent(out) :: e
      integer(c_int64_t), allocatable :: i64out(:)
      call tensogram_to_array(m, 1, i64out, e)   ! int64 out for a float32 object
   end subroutine wrong_dtype

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_errors: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

end program test_errors

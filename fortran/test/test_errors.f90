! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Error-path behaviour: decoding garbage must return a non-OK code (not abort)
!> and populate the thread-local last-error string; tensogram_strerror must
!> return a non-empty description for a known code.
program test_errors
   use, intrinsic :: iso_c_binding, only : c_int8_t, c_int
   use tensogram
   implicit none

   integer(c_int8_t) :: garbage(32)
   type(tensogram_message) :: msg
   integer(c_int) :: err
   integer :: i

   ! Not a TENSOGRM message — decode must fail gracefully.
   do i = 1, size(garbage)
      garbage(i) = int(mod(i * 37, 251), c_int8_t)
   end do

   call tensogram_decode(garbage, msg, err)
   call assert(err /= TGM_ERROR_OK, 'decode of garbage returns non-OK')
   call assert(len(tensogram_last_error()) > 0, 'last_error is populated')
   call assert(len(tensogram_strerror(TGM_ERROR_FRAMING)) > 0, 'strerror non-empty')

   print '(a)', 'test_errors: PASS'

contains
   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_errors: FAIL: ', what
         error stop 1
      end if
   end subroutine assert
end program test_errors

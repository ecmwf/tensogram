! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Negative test for the non-copyable handle guard (PLAN_FORTRAN.md §5.4).
!> A whole-type assignment of a handle must invoke the defined `assignment(=)`
!> and `error stop` — aborting with a non-zero exit code. This program is
!> registered in CTest with WILL_FAIL TRUE: if the guard ever stops firing
!> (e.g. the defined assignment becomes inaccessible and intrinsic shallow
!> copy silently takes over), control reaches the final print, the program
!> exits 0, and CTest flags the regression.
program test_copy_guard
   use tensogram
   implicit none
   type(tensogram_message) :: a, b

   ! Must abort here via the non-copyable guard.
   b = a

   ! Unreachable if the guard works.
   print '(a)', 'test_copy_guard: FAIL: handle assignment did not abort'
   call use_handle(b)
contains
   subroutine use_handle(h)
      type(tensogram_message), intent(in) :: h
      ! Touch `h` so the compiler cannot optimise the copy away.
      if (tensogram_num_objects(h) < 0) print '(a)', 'unreachable'
   end subroutine use_handle
end program test_copy_guard

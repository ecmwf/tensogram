! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> examples/fortran/file_api.f90
!>
!> Multi-message .tgm file workflow — the common "append forecast steps"
!> pattern: create a file, append several fields one message at a time, then
!> reopen, count, and decode each by index (random access).
program file_api
   use, intrinsic :: iso_c_binding, only : c_float, c_int
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'forecast.tgm'
   integer, parameter :: NI = 32, NJ = 16, NSTEPS = 5
   type(tensogram_file)    :: f
   type(tensogram_message) :: msg
   real(c_float)              :: field(NI, NJ)
   real(c_float), allocatable :: out(:,:)
   integer(c_int) :: err
   integer :: step, n, i, j

   ! Produce and append NSTEPS fields, one message per step.
   call tensogram_file_create(path, f, err)
   call tensogram_check(err, 'create')
   do step = 1, NSTEPS
      do j = 1, NJ
         do i = 1, NI
            field(i, j) = real(step, c_float) + 0.01_c_float * real(i + j, c_float)
         end do
      end do
      call tensogram_file_append(f, field, err)
      call tensogram_check(err, 'append')
   end do
   call f%close()
   print '(a,i0,a,a)', 'Wrote ', NSTEPS, ' messages to ', path

   ! Reopen, count, and decode each step back.
   call tensogram_file_open(path, f, err)
   call tensogram_check(err, 'open')
   call tensogram_file_message_count(f, n, err)
   call tensogram_check(err, 'message_count')
   print '(a,i0,a)', 'Reopened: ', n, ' messages'

   do step = 1, n
      call tensogram_file_decode_message(f, step, msg, err)
      call tensogram_check(err, 'decode_message')
      call tensogram_to_array(msg, 1, out, err)
      call tensogram_check(err, 'to_array')
      print '(a,i0,a,i0,a,i0,a,f6.3)', '  step ', step, ': shape (', &
         size(out, 1), ',', size(out, 2), ')  field(1,1)=', out(1, 1)
   end do
   call f%close()
end program file_api

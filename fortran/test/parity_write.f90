! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Cross-language parity producer: encode a known rank-2 float32 field in
!> Fortran and write the raw Tensogram message bytes to the file named by the
!> first command-line argument. The companion `parity_check.py` decodes it and
!> asserts the column-major contract from the Python side (PLAN_FORTRAN.md §8):
!> a Fortran a(ni,nj) is seen by NumPy as the transpose (nj, ni), with
!> arr[j-1, i-1] == field(i, j).
program parity_write
   use, intrinsic :: iso_c_binding, only : c_float, c_int8_t, c_int
   use tensogram
   implicit none
   integer, parameter :: NI = 5, NJ = 3
   real(c_float)                  :: field(NI, NJ)
   integer(c_int8_t), allocatable :: wire(:)
   type(tensogram_buffer) :: buf
   integer(c_int) :: err
   integer :: i, j, u
   character(len=4096) :: path

   if (command_argument_count() < 1) then
      print '(a)', 'usage: parity_write <out.tgm>'
      error stop 2
   end if
   call get_command_argument(1, path)

   do j = 1, NJ
      do i = 1, NI
         field(i, j) = real(i * 1000 + j, c_float)
      end do
   end do

   call tensogram_encode(field, buf, err)
   call tensogram_check(err, 'encode')
   call buf%as_array(wire)

   open(newunit=u, file=trim(path), access='stream', form='unformatted', status='replace')
   write(u) wire
   close(u)
   print '(a,i0,a,a)', 'parity_write: wrote ', size(wire), ' bytes to ', trim(path)
end program parity_write

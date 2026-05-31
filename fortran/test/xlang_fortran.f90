! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Cross-language parity harness, Fortran (column-major) side. Pairs with
!> xlang_c.cpp. Both agree on a logical field F(i,j) = i*1000 + j (1-based).
!> Lossless zstd compression; values are exact integers-as-floats so equality
!> is exact (compared as integers to stay clear of real-equality pitfalls).
!>
!> Modes: `fwrite <path>` (encode a(ni,nj) -> file), `fread <path>` (decode a
!> C-written [nj,ni] tensor into out(ni,nj) and check it equals F).
program xlang_fortran
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none
   integer, parameter :: NI = 5, NJ = 3
   character(len=16)   :: mode
   character(len=4096) :: path

   if (command_argument_count() < 2) then
      print '(a)', 'usage: xlang_fortran <fwrite|fread> <path>'
      error stop 2
   end if
   call get_command_argument(1, mode)
   call get_command_argument(2, path)

   select case (trim(mode))
   case ('fwrite'); call fwrite(trim(path))
   case ('fread');  call fread(trim(path))
   case default
      print '(a,a)', 'xlang_fortran: unknown mode ', trim(mode)
      error stop 2
   end select

contains

   pure integer function F(i, j)
      integer, intent(in) :: i, j
      F = i * 1000 + j
   end function F

   subroutine fwrite(p)
      character(len=*), intent(in) :: p
      real(c_float)              :: a(NI, NJ)
      type(tensogram_buffer)     :: buf
      integer(c_int8_t), allocatable :: wire(:)
      integer(c_int) :: err
      integer :: i, j, u
      do j = 1, NJ; do i = 1, NI; a(i, j) = real(F(i, j), c_float); end do; end do
      call tensogram_encode(a, buf, err, compression='zstd')
      call tensogram_check(err, 'fwrite encode')
      call buf%as_array(wire)
      open(newunit=u, file=p, access='stream', form='unformatted', status='replace')
      write(u) wire
      close(u)
      print '(a,i0,a)', 'fwrite: wrote ', size(wire), ' bytes'
   end subroutine fwrite

   subroutine fread(p)
      character(len=*), intent(in) :: p
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_message)    :: msg
      real(c_float), allocatable :: out(:,:)
      integer(c_int) :: err
      integer :: i, j, u, sz

      open(newunit=u, file=p, access='stream', form='unformatted', status='old')
      inquire(unit=u, size=sz)
      allocate(wire(sz))
      read(u) wire
      close(u)

      call tensogram_decode(wire, msg, err);    call tensogram_check(err, 'fread decode')
      call tensogram_to_array(msg, 1, out, err); call tensogram_check(err, 'fread to_array')

      if (.not. all(shape(out) == [NI, NJ])) then
         print '(a,2i0)', 'fread: shape mismatch ', shape(out)
         error stop 1
      end if
      do j = 1, NJ; do i = 1, NI
         if (nint(out(i, j)) /= F(i, j)) then     ! integer compare: exact + lint-clean
            print '(a,i0,a,i0,a,i0,a,i0)', 'fread: out(', i, ',', j, ')=', &
               nint(out(i, j)), ' != ', F(i, j)
            error stop 1
         end if
      end do; end do
      print '(a)', 'fread: PASS (C [nj,ni] read into Fortran (ni,nj))'
   end subroutine fread

end program xlang_fortran

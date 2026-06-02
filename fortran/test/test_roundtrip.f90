! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Round-trip correctness: encode a rank-2 float32 field, decode it back, and
!> assert the result is bit-identical and the descriptor metadata survived.
!> Also exercises the column-major contract: the decoded Fortran shape must
!> equal the input (ni, nj), while the on-wire object shape is the transpose.
program test_roundtrip
   use, intrinsic :: iso_c_binding, only : c_float, c_int8_t, c_int, c_int32_t, c_int64_t
   use tensogram
   implicit none

   integer, parameter :: NI = 17, NJ = 29
   real(c_float)                  :: field(NI, NJ)
   real(c_float),    allocatable  :: out(:,:)
   integer(c_int8_t), allocatable :: wire(:)
   integer(c_int64_t), allocatable :: ext(:)
   type(tensogram_buffer)  :: buf
   type(tensogram_message) :: msg
   integer(c_int) :: err
   integer :: i, j

   do j = 1, NJ
      do i = 1, NI
         field(i, j) = real(i, c_float) * 7.0_c_float - real(j, c_float) * 0.5_c_float
      end do
   end do

   call tensogram_encode(field, buf, err)
   call assert(err == TGM_ERROR_OK, 'encode returned OK')
   call assert(buf%size() > 0_c_int8_t, 'encoded buffer is non-empty')

   call buf%as_array(wire)
   call tensogram_decode(wire, msg, err)
   call assert(err == TGM_ERROR_OK, 'decode returned OK')

   call assert(tensogram_num_objects(msg) == 1, 'one object decoded')
   call assert(tensogram_object_dtype(msg, 1) == 'float32', 'dtype is float32')
   call assert(tensogram_object_ndim(msg, 1) == 2, 'ndim is 2')

   ! Column-major contract: Fortran extents come back as (ni, nj).
   ext = tensogram_object_shape(msg, 1)
   call assert(size(ext) == 2, 'shape has rank 2')
   call assert(int(ext(1)) == NI, 'fastest extent is NI')
   call assert(int(ext(2)) == NJ, 'slowest extent is NJ')

   call tensogram_to_array(msg, 1, out, err)
   call assert(err == TGM_ERROR_OK, 'to_array returned OK')
   call assert(size(out, 1) == NI .and. size(out, 2) == NJ, 'decoded shape (NI, NJ)')
   call assert(bit_identical(out, field), 'round-trip bit-identical')

   print '(a)', 'test_roundtrip: PASS'

contains
   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_roundtrip: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   !> Byte-exact equality via bit patterns — the correct test for a lossless
   !> round-trip (avoids real `==` and also distinguishes ±0.0 / NaN payloads).
   logical function bit_identical(x, y)
      real(c_float), intent(in) :: x(:,:), y(:,:)
      integer(c_int32_t), allocatable :: bx(:), by(:)
      bit_identical = .false.
      if (size(x, 1) /= size(y, 1) .or. size(x, 2) /= size(y, 2)) return
      bx = transfer(x, 0_c_int32_t, size(x))
      by = transfer(y, 0_c_int32_t, size(y))
      bit_identical = all(bx == by)
   end function bit_identical
end program test_roundtrip

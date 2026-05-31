! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> examples/fortran/encode_decode.f90
!>
!> End-to-end Tensogram round-trip from Fortran, mirroring
!> examples/python/01_encode_decode.py:
!>
!>   field(ni,nj)  --encode-->  tensogram_buffer (Rust-owned bytes)
!>                 --copy out-->  wire(:)  (Fortran int8 array)
!>                 --decode-->   tensogram_message handle
!>                 --extract-->  out(ni,nj)
!>
!> and asserts the round-trip is bit-identical.
program encode_decode
   use, intrinsic :: iso_c_binding, only : c_float, c_int8_t, c_int
   use tensogram
   implicit none

   integer, parameter :: NI = 100, NJ = 200
   real(c_float)                  :: field(NI, NJ)
   real(c_float),    allocatable  :: out(:,:)
   integer(c_int8_t), allocatable :: wire(:)
   type(tensogram_buffer)  :: buf
   type(tensogram_message) :: msg
   integer(c_int)    :: err
   integer           :: i, j
   real(c_float)     :: maxdiff

   ! 1. Synthesise a distinct value per element.
   do j = 1, NJ
      do i = 1, NI
         field(i, j) = real(i, c_float) + 0.001_c_float * real(j, c_float)
      end do
   end do
   print '(a,i0,a,i0,a)', 'Input:   shape=(', NI, ',', NJ, ')  dtype=float32'

   ! 2. Encode -> Rust-owned byte buffer (lossless: encoding/compression "none").
   call tensogram_encode(field, buf, err)
   call tensogram_check(err, 'encode')
   print '(a,i0,a)',      'Message: ', buf%size(), ' bytes'

   ! 3. tgm_bytes_t round-trip: copy the wire bytes out, then release the buffer.
   call buf%as_array(wire)
   call buf%free()                       ! optional: free Rust memory early

   ! 4. Decode the wire bytes -> message handle.
   call tensogram_decode(wire, msg, err)
   call tensogram_check(err, 'decode')
   print '(a,i0,a)',      'Decoded: ', tensogram_num_objects(msg), ' object(s)'
   print '(a,a)',         '  dtype = ', tensogram_object_dtype(msg, 1)

   ! 5. Extract object 1 back into a Fortran array shaped (ni, nj).
   call tensogram_to_array(msg, 1, out, err)
   call tensogram_check(err, 'to_array')
   print '(a,i0,a,i0,a)', '  shape = (', size(out, 1), ',', size(out, 2), ')'

   ! 6. Verify the Fortran <-> Fortran round-trip is identity.
   maxdiff = maxval(abs(out - field))
   print '(a,es12.5)',    '  max |out - field| = ', maxdiff
   if (maxdiff > 0.0_c_float) then
      print '(a)', 'Round-trip MISMATCH.'
      error stop 1
   else
      print '(a)', 'Round-trip OK (bit-identical).'
   end if
end program encode_decode

! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> examples/fortran/streaming.f90
!>
!> Progressive (streaming) encode: write a single multi-object message one
!> tensor at a time without buffering the whole message, then reopen it and
!> decode every object. Useful for producers that emit objects incrementally.
program streaming
   use, intrinsic :: iso_c_binding, only : c_float, c_int
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'streamed.tgm'
   integer, parameter :: NOBJ = 4, NI = 16, NJ = 8
   type(tensogram_streaming_encoder) :: enc
   type(tensogram_file)              :: f
   type(tensogram_message)           :: msg
   real(c_float)              :: field(NI, NJ)
   real(c_float), allocatable :: out(:,:)
   integer(c_int) :: err
   integer :: obj, i, j

   ! Open the stream and write NOBJ objects one at a time (lossless zstd).
   call tensogram_streaming_encoder_create(path, enc, err)
   call tensogram_check(err, 'create')
   do obj = 1, NOBJ
      do j = 1, NJ
         do i = 1, NI
            field(i, j) = real(obj, c_float) + 0.001_c_float * real(i * NJ + j, c_float)
         end do
      end do
      call tensogram_streaming_encoder_write(enc, field, err, compression='zstd')
      call tensogram_check(err, 'write')
      print '(a,i0,a,i0)', '  wrote object ', obj, '; count = ', &
         tensogram_streaming_encoder_count(enc)
   end do
   call tensogram_streaming_encoder_finish(enc, err)
   call tensogram_check(err, 'finish')
   call enc%free()
   print '(a,i0,a,a)', 'Streamed ', NOBJ, ' objects into ', path

   ! Reopen and decode every object from the single message.
   call tensogram_file_open(path, f, err)
   call tensogram_check(err, 'open')
   call tensogram_file_decode_message(f, 1, msg, err)
   call tensogram_check(err, 'decode_message')
   print '(a,i0,a)', 'Decoded message with ', tensogram_num_objects(msg), ' object(s)'
   do obj = 1, tensogram_num_objects(msg)
      call tensogram_to_array(msg, obj, out, err)
      call tensogram_check(err, 'to_array')
      print '(a,i0,a,i0,a,i0,a,f7.3)', '  object ', obj, ': shape (', &
         size(out, 1), ',', size(out, 2), ')  field(1,1)=', out(1, 1)
   end do
   call f%close()
end program streaming

! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Negative tests for the deliberate abort paths: every handle type's
!> non-copyable guard must `error stop` when a LIVE handle is copied, and
!> tensogram_check must `error stop` on a non-OK code. One program selected by
!> a command-line argument; each mode is registered as a WILL_FAIL CTest, so a
!> clean exit (guard silently gone) is reported as a regression.
program test_guards
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none
   character(len=32) :: which

   if (command_argument_count() < 1) error stop 2
   call get_command_argument(1, which)

   select case (trim(which))
   case ('buffer');      call g_buffer()
   case ('message');     call g_message()
   case ('file');        call g_file()
   case ('metadata');    call g_metadata()
   case ('stream');      call g_stream()
   case ('check_ctx');   call tensogram_check(TGM_ERROR_OBJECT, 'guard-context')
   case ('check_noctx'); call tensogram_check(TGM_ERROR_OBJECT)
   case default;         print '(a)', 'unknown guard mode'; error stop 2
   end select

   ! Reached only if no guard fired — WILL_FAIL turns this clean exit into a
   ! reported regression.
   print '(a)', 'test_guards: GUARD DID NOT FIRE'

contains

   subroutine live_buffer(buf)
      type(tensogram_buffer), intent(out) :: buf
      real(c_float)  :: a(2)
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float]
      call tensogram_encode(a, buf, err)
   end subroutine live_buffer

   subroutine g_buffer()
      type(tensogram_buffer) :: a, b
      call live_buffer(a)
      b = a                                   ! non-copyable guard -> error stop
      if (b%size() < 0_c_size_t) print '(a)', 'x'
   end subroutine g_buffer

   subroutine g_message()
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: a, b
      integer(c_int8_t), allocatable :: w(:)
      integer(c_int) :: err
      call live_buffer(buf); call buf%as_array(w)
      call tensogram_decode(w, a, err)
      b = a
      if (tensogram_num_objects(b) < 0) print '(a)', 'x'
   end subroutine g_message

   subroutine g_file()
      type(tensogram_file) :: a, b
      integer(c_int) :: err, n
      call tensogram_file_create('test_guards_tmp.tgm', a, err)
      b = a
      call tensogram_file_message_count(b, n, err)
   end subroutine g_file

   subroutine g_metadata()
      type(tensogram_buffer)   :: buf
      type(tensogram_message)  :: msg
      type(tensogram_metadata) :: a, b
      integer(c_int8_t), allocatable :: w(:)
      integer(c_int) :: err
      call live_buffer(buf); call buf%as_array(w)
      call tensogram_decode(w, msg, err)
      call tensogram_message_metadata(msg, a, err)
      b = a
      if (len(tensogram_metadata_get_string(b, 'none')) < 0) print '(a)', 'x'
   end subroutine g_metadata

   subroutine g_stream()
      type(tensogram_streaming_encoder) :: a, b
      integer(c_int) :: err
      call tensogram_streaming_encoder_create('test_guards_stream_tmp.tgm', a, err)
      b = a
      if (tensogram_streaming_encoder_count(b) < 0) print '(a)', 'x'
   end subroutine g_stream

end program test_guards

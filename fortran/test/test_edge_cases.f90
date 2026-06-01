! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Edge-case behaviour of the Fortran binding: zero-size tensors, out-of-range
!> object indices, too-short / empty decode buffers, zero-object streaming
!> messages, and Unicode / long-string metadata. Each must fail gracefully
!> (clean error or empty result) rather than crash, or round-trip exactly.
program test_edge_cases
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   call empty_array_roundtrip()
   call out_of_range_object()
   call short_buffer_decode()
   call zero_object_stream()
   call unicode_and_long_metadata()

   print '(a)', 'test_edge_cases: PASS'

contains

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_edge_cases: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   !> A zero-element tensor encodes and round-trips to a zero-size array.
   subroutine empty_array_roundtrip()
      real(c_float)              :: a(0)
      real(c_float), allocatable :: out(:)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      call tensogram_encode(a, buf, err);    call assert(err == TGM_ERROR_OK, 'empty encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'empty decode')
      call assert(tensogram_num_objects(msg) == 1, 'empty: one object')
      call tensogram_to_array(msg, 1, out, err); call assert(err == TGM_ERROR_OK, 'empty to_array')
      call assert(size(out) == 0, 'empty: decoded size 0')
   end subroutine empty_array_roundtrip

   !> Object indices outside [1, num_objects] fail cleanly, never crash.
   subroutine out_of_range_object()
      real(c_float)              :: a(3)
      real(c_float), allocatable :: out(:)
      integer(c_int8_t), allocatable :: wire(:)
      type(tensogram_buffer)  :: buf
      type(tensogram_message) :: msg
      integer(c_int) :: err
      a = [1.0_c_float, 2.0_c_float, 3.0_c_float]
      call tensogram_encode(a, buf, err);    call assert(err == TGM_ERROR_OK, 'oor encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err); call assert(err == TGM_ERROR_OK, 'oor decode')
      call tensogram_to_array(msg, 2, out, err)   ! beyond count
      call assert(err == TGM_ERROR_OBJECT, 'to_array beyond count -> OBJECT')
      call tensogram_to_array(msg, 0, out, err)   ! zero / underflow index
      call assert(err == TGM_ERROR_OBJECT, 'to_array index 0 -> OBJECT')
      call assert(len(tensogram_object_dtype(msg, 9)) == 0, 'oor dtype -> empty string')
   end subroutine out_of_range_object

   !> Empty and too-short wire buffers decode to a clean error, not a crash.
   subroutine short_buffer_decode()
      integer(c_int8_t), allocatable :: wire(:)
      integer(c_int8_t) :: tiny(4)
      type(tensogram_message) :: msg
      integer(c_int) :: err
      allocate(wire(0))
      call tensogram_decode(wire, msg, err); call assert(err /= TGM_ERROR_OK, 'empty buffer -> error')
      tiny = 0_c_int8_t
      call tensogram_decode(tiny, msg, err); call assert(err /= TGM_ERROR_OK, 'tiny buffer -> error')
      call assert(len(tensogram_last_error()) > 0, 'short buffer sets last_error')
   end subroutine short_buffer_decode

   !> A streamed message with no objects (create then finish) is valid and
   !> decodes to zero objects.
   subroutine zero_object_stream()
      character(len=*), parameter :: p = 'test_edge_zero_tmp.tgm'
      type(tensogram_streaming_encoder) :: enc
      type(tensogram_file)              :: f
      type(tensogram_message)           :: msg
      integer(c_int) :: err
      integer :: n, ios
      call tensogram_streaming_encoder_create(p, enc, err); call assert(err == TGM_ERROR_OK, 'zero-stream create')
      call tensogram_streaming_encoder_finish(enc, err);    call assert(err == TGM_ERROR_OK, 'zero-stream finish')
      call enc%free()
      call tensogram_file_open(p, f, err);             call assert(err == TGM_ERROR_OK, 'zero-stream open')
      call tensogram_file_message_count(f, n, err);    call assert(err == TGM_ERROR_OK, 'zero-stream count')
      call assert(n == 1, 'zero-stream: one message')
      call tensogram_file_decode_message(f, 1, msg, err); call assert(err == TGM_ERROR_OK, 'zero-stream decode')
      call assert(tensogram_num_objects(msg) == 0, 'zero-stream: zero objects')
      call f%close()
      open(newunit=ios, file=p, status='old', iostat=err)
      if (err == 0) close(ios, status='delete')
   end subroutine zero_object_stream

   !> Unicode (raw UTF-8) and long strings survive the metadata builder /
   !> JSON escaper / decode / getter round-trip byte-for-byte.
   subroutine unicode_and_long_metadata()
      type(tensogram_meta)     :: m
      type(tensogram_buffer)   :: buf
      type(tensogram_message)  :: msg
      type(tensogram_metadata) :: meta
      integer(c_int8_t), allocatable :: wire(:)
      real(c_float)  :: a(2)
      integer(c_int) :: err
      character(len=:), allocatable :: uni, longstr
      a = [1.0_c_float, 2.0_c_float]
      uni = 'caf' // char(195) // char(169)    ! "café" as raw UTF-8 bytes
      longstr = repeat('x', 500)
      call m%add_string('city', uni)
      call m%add_string('long', longstr)
      call tensogram_encode(a, buf, err, metadata_json=m%base_json())
      call assert(err == TGM_ERROR_OK, 'unicode encode')
      call buf%as_array(wire)
      call tensogram_decode(wire, msg, err);            call assert(err == TGM_ERROR_OK, 'unicode decode')
      call tensogram_message_metadata(msg, meta, err);  call assert(err == TGM_ERROR_OK, 'unicode metadata')
      call assert(tensogram_metadata_get_string(meta, 'city') == uni, 'unicode round-trip')
      call assert(tensogram_metadata_get_string(meta, 'long') == longstr, 'long string round-trip')
   end subroutine unicode_and_long_metadata

end program test_edge_cases

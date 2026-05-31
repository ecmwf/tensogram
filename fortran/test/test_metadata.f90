! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Metadata ergonomics: build per-object application metadata with the
!> tensogram_meta builder (incl. characters needing JSON escaping), encode with
!> a lossless zstd pipeline, decode, assert the round-trip is bit-identical, and
!> read the metadata back through the dot-notation getters.
program test_metadata
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   ! Contains a quote, a backslash, and a newline -> exercises json_escape.
   character(len=*), parameter :: NOTE = 'a' // char(34) // 'b' // char(92) // &
                                          'c' // char(10) // 'd'

   type(tensogram_meta)     :: m
   type(tensogram_buffer)   :: buf
   type(tensogram_message)  :: msg
   type(tensogram_metadata) :: meta
   integer(c_int8_t), allocatable :: wire(:)
   real(c_float)              :: field(8, 8)
   real(c_float), allocatable :: out(:,:)
   integer(c_int) :: err
   integer :: i, j

   do j = 1, 8; do i = 1, 8
      field(i, j) = real(i, c_float) + 0.5_c_float * real(j, c_float)
   end do; end do

   call m%add_string('name', 'temperature')
   call m%add_string('units', 'K')
   call m%add_int('level', 850_c_int64_t)
   call m%add_real('scale', 1.25_c_double)
   call m%add_string('note', NOTE)

   ! Encode with a lossless compressor (zstd) plus the metadata.
   call tensogram_encode(field, buf, err, metadata_json=m%base_json(), compression='zstd')
   call assert(err == TGM_ERROR_OK, 'encode (zstd + metadata)')

   call buf%as_array(wire)
   call tensogram_decode(wire, msg, err)
   call assert(err == TGM_ERROR_OK, 'decode')

   ! Lossless: the compressed round-trip is bit-identical.
   call tensogram_to_array(msg, 1, out, err)
   call assert(err == TGM_ERROR_OK, 'to_array')
   call assert(bit_eq(out, field), 'zstd round-trip bit-identical')

   ! Read the application metadata back via dot-notation getters.
   call tensogram_message_metadata(msg, meta, err)
   call assert(err == TGM_ERROR_OK, 'message_metadata')
   call assert(tensogram_metadata_get_string(meta, 'name') == 'temperature', 'name')
   call assert(tensogram_metadata_get_string(meta, 'units') == 'K', 'units')
   call assert(tensogram_metadata_get_int(meta, 'level', -1_c_int64_t) == 850_c_int64_t, 'level')
   call assert(abs(tensogram_metadata_get_float(meta, 'scale', 0.0_c_double) - 1.25_c_double) &
               < 1.0e-12_c_double, 'scale')
   call assert(tensogram_metadata_get_string(meta, 'note') == NOTE, 'note (escaped round-trip)')

   ! Absent keys return the supplied default / empty string.
   call assert(len(tensogram_metadata_get_string(meta, 'missing')) == 0, 'missing string -> empty')
   call assert(tensogram_metadata_get_int(meta, 'missing', 42_c_int64_t) == 42_c_int64_t, &
               'missing int -> default')

   print '(a)', 'test_metadata: PASS'

contains

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_metadata: FAIL: ', what
         error stop 1
      end if
   end subroutine assert

   logical function bit_eq(x, y)
      real(c_float), intent(in) :: x(:,:), y(:,:)
      bit_eq = all(shape(x) == shape(y))
      if (bit_eq) bit_eq = all(transfer(x, [0_c_int8_t]) == transfer(y, [0_c_int8_t]))
   end function bit_eq

end program test_metadata

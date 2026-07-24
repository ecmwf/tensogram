! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> examples/fortran/validate.f90
!>
!> Inspect a multi-message buffer without a full decode: SCAN it for message
!> boundaries, read each message's wire VERSION, and VALIDATE it (JSON report).
!> Then VALIDATE a `.tgm` file on disk. Mirrors the whole-buffer utilities in
!> the Python `tensogram.validate` / `tensogram.scan` and the C++ `validate` /
!> `scan` surfaces.
program validate
   use, intrinsic :: iso_c_binding, only : c_float, c_int8_t, c_int, c_size_t
   use tensogram
   implicit none

   character(len=*), parameter :: path = 'validated.tgm'
   integer, parameter :: NI = 8, NJ = 4
   real(c_float)                  :: field(NI, NJ)
   integer(c_int8_t), allocatable :: wa(:), wb(:), buffer(:), seg(:)
   integer(c_size_t), allocatable :: offsets(:), lengths(:)
   type(tensogram_buffer)  :: bufa, bufb
   type(tensogram_file)    :: f
   type(tensogram_message) :: msg
   character(len=:), allocatable :: report
   integer(c_int) :: err
   integer :: k, i, j

   print '(a,i0)', 'Library wire-format version: TGM_WIRE_VERSION = ', TGM_WIRE_VERSION

   ! Build two encoded messages and concatenate them into one buffer.
   do j = 1, NJ; do i = 1, NI; field(i, j) = real(i + j, c_float); end do; end do
   call tensogram_encode(field, bufa, err); call tensogram_check(err, 'encode A')
   field = field + 100.0_c_float
   call tensogram_encode(field, bufb, err); call tensogram_check(err, 'encode B')
   call bufa%as_array(wa)
   call bufb%as_array(wb)
   buffer = [wa, wb]
   print '(a,i0,a)', 'Concatenated buffer: ', size(buffer), ' bytes (2 messages)'

   ! 1. Scan for message boundaries (offsets are 1-based indices into buffer).
   call tensogram_scan(buffer, offsets, lengths, err)
   call tensogram_check(err, 'scan')
   print '(a,i0,a)', 'Scan found ', size(offsets), ' message(s):'
   do k = 1, size(offsets)
      ! 2. Slice each message out, read its wire version and validate it.
      seg = buffer(offsets(k) : offsets(k) + lengths(k) - 1_c_size_t)
      call tensogram_decode(seg, msg, err)
      call tensogram_check(err, 'decode')
      report = tensogram_validate(seg)
      print '(a,i0,a,i0,a,i0,a,i0,a,l1)',                                    &
         '  msg ', k, ': offset=', offsets(k), ' length=', lengths(k),       &
         ' version=', tensogram_message_version(msg),                        &
         ' clean=', index(report, '"issues":[]') > 0
   end do

   ! 3. Validate a `.tgm` file on disk (all messages, full level).
   call tensogram_file_create(path, f, err); call tensogram_check(err, 'create')
   call tensogram_file_append(f, field, err); call tensogram_check(err, 'append')
   call f%close()
   report = tensogram_validate_file(path, level='full', err=err)
   call tensogram_check(err, 'validate_file')
   print '(a,l1,a,l1)', 'File report: has messages=', index(report, '"messages"') > 0, &
      '  file_issues empty=', index(report, '"file_issues":[]') > 0
end program validate

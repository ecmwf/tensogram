! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Deferred Wave-B, component B — the SYNCHRONOUS remote surface:
!>   * tensogram_is_remote_url    (tgm_is_remote_url)
!>   * tensogram_file_open_remote (tgm_file_open_remote)
!>
!> The remote backend is an opt-in Cargo feature of the C library, but its
!> symbols always link: a build without `--features=remote` answers "not remote
!> for me" to every URL and returns TGM_ERROR_REMOTE (with an explanatory
!> message) from open_remote. Both flavours are covered here — the build's own
!> answer for one object-store URL selects which expectations apply, so the
!> test is honest on a feature-off build and still exercises the real backend
!> when one is present.
program test_remote
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   integer :: npass
   logical :: remote_capable

   npass = 0
   ! One probe decides this build's flavour; every other scheme must agree.
   remote_capable = tensogram_is_remote_url('s3://bucket/key.tgm')

   call local_sources_are_never_remote()
   call object_store_schemes_match_the_build()
   call open_remote_validates_arguments_first()
   call open_remote_behaviour()

   print '(a,i0,a,l1,a)', 'test_remote: PASS (', npass, ' checks, remote feature = ', remote_capable, ')'

contains

   ! ---- Local sources belong to the local backend, in every build ----------
   subroutine local_sources_are_never_remote()
      call assert(.not. tensogram_is_remote_url('/tmp/data.tgm'), 'absolute path is not remote')
      call assert(.not. tensogram_is_remote_url('data.tgm'), 'bare filename is not remote')
      call assert(.not. tensogram_is_remote_url('./relative/data.tgm'), 'relative path is not remote')
      call assert(.not. tensogram_is_remote_url('file:///tmp/data.tgm'), 'file:// is not remote')
      call assert(.not. tensogram_is_remote_url(''), 'empty source is not remote')
      call assert(.not. tensogram_is_remote_url('ftp://host/data.tgm'), 'ftp:// is not a supported scheme')
   end subroutine local_sources_are_never_remote

   ! ---- Every object-store scheme answers with this build's capability ----
   subroutine object_store_schemes_match_the_build()
      call assert(tensogram_is_remote_url('s3://bucket/key.tgm')      .eqv. remote_capable, 's3://')
      call assert(tensogram_is_remote_url('s3a://bucket/key.tgm')     .eqv. remote_capable, 's3a://')
      call assert(tensogram_is_remote_url('gs://bucket/key.tgm')      .eqv. remote_capable, 'gs://')
      call assert(tensogram_is_remote_url('az://container/key.tgm')   .eqv. remote_capable, 'az://')
      call assert(tensogram_is_remote_url('azure://container/key.tgm').eqv. remote_capable, 'azure://')
      call assert(tensogram_is_remote_url('http://host/key.tgm')      .eqv. remote_capable, 'http://')
      call assert(tensogram_is_remote_url('https://host/key.tgm')     .eqv. remote_capable, 'https://')
      ! Schemes are compared case-insensitively.
      call assert(tensogram_is_remote_url('S3://bucket/key.tgm')  .eqv. remote_capable, 'S3:// (upper case)')
      call assert(tensogram_is_remote_url('HTTPS://host/key.tgm') .eqv. remote_capable, 'HTTPS:// (upper case)')
      ! Trailing blanks of a fixed-length actual argument are not part of the URL.
      call assert(tensogram_is_remote_url(padded()) .eqv. remote_capable, 'blank-padded source is trimmed')
   end subroutine object_store_schemes_match_the_build

   ! ---- Argument validation is identical in every build --------------------
   !  Parallel arrays are only well-formed when they are the same length; the
   !  C ABI takes a single n_options, so the mismatch is caught here rather
   !  than silently reading past the shorter array.
   subroutine open_remote_validates_arguments_first()
      type(tensogram_file) :: f
      integer(c_int) :: err
      character(len=16) :: keys2(2), values1(1), keys1(1), values2(2), none(0)
      keys2    = ['aws_region      ', 'aws_endpoint    ']
      values1  = ['eu-west-1       ']
      keys1    = ['aws_region      ']
      values2  = ['eu-west-1       ', 'http://localhost']

      ! Rejected before the C call, so no build ever touches the network here.
      call tensogram_file_open_remote('s3://bucket/key.tgm', keys2, values1, f, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'more keys than values -> INVALID_ARG')
      call tensogram_file_open_remote('s3://bucket/key.tgm', keys1, values2, f, err)
      call assert(err == TGM_ERROR_INVALID_ARG, 'more values than keys -> INVALID_ARG')
      ! Zero-size arrays mean "no storage options" — never an argument error.
      ! A `file://` URL keeps this hermetic: it reaches the backend (or the
      ! feature-off stub) without a network round-trip in either build.
      call tensogram_file_open_remote('file:///nonexistent/missing.tgm', none, none, f, err)
      call assert(err /= TGM_ERROR_INVALID_ARG, 'empty option arrays are not an argument error')
   end subroutine open_remote_validates_arguments_first

   ! ---- open_remote: feature-off explains itself, feature-on reads ---------
   subroutine open_remote_behaviour()
      type(tensogram_file) :: f
      integer(c_int) :: err
      character(len=16) :: keys(1), values(1)
      character(len=:), allocatable :: detail
      keys   = ['aws_region      ']
      values = ['eu-west-1       ']

      if (.not. remote_capable) then
         ! Documented feature-off contract: TGM_ERROR_REMOTE + how to fix it.
         call tensogram_file_open_remote('s3://bucket/key.tgm', f, err)
         call assert(err == TGM_ERROR_REMOTE, 'feature-off open_remote -> TGM_ERROR_REMOTE')
         detail = tensogram_last_error()
         call assert(index(detail, 'remote') > 0, 'feature-off open_remote explains the missing feature')
         call assert(index(detail, 'features=remote') > 0, 'feature-off open_remote says how to enable it')

         ! Storage options and scan options are marshalled before the feature
         ! check, so a well-formed call still reaches the same honest answer.
         call tensogram_file_open_remote('s3://bucket/key.tgm', keys, values, f, err)
         call assert(err == TGM_ERROR_REMOTE, 'feature-off open_remote with options -> TGM_ERROR_REMOTE')
         call tensogram_file_open_remote('s3://bucket/key.tgm', f, err, bidirectional=.false.)
         call assert(err == TGM_ERROR_REMOTE, 'feature-off open_remote forward-only -> TGM_ERROR_REMOTE')
      else
         call open_remote_round_trip()
      end if
   end subroutine open_remote_behaviour

   !> A remote-capable build: `file://` travels the same object-store code
   !> path as `s3://` & friends, so the round-trip needs no network.
   subroutine open_remote_round_trip()
      character(len=*), parameter :: path = 'test_remote_fixture.tgm'
      character(len=4096) :: cwd
      character(len=:), allocatable :: url
      type(tensogram_file)    :: f, remote
      type(tensogram_message) :: msg
      real(c_float), allocatable :: got(:)
      integer(c_int) :: err
      integer :: n, unit, ios
      call get_environment_variable('PWD', cwd, status=ios)
      if (ios /= 0 .or. len_trim(cwd) == 0) return       ! cannot build an absolute URL
      url = 'file://' // trim(cwd) // '/' // path

      call tensogram_file_create(path, f, err); call assert(err == TGM_ERROR_OK, 'fixture: create')
      call tensogram_file_append(f, [1.0_c_float, 2.0_c_float, 3.0_c_float], err)
      call assert(err == TGM_ERROR_OK, 'fixture: append A')
      call tensogram_file_append(f, [5.0_c_float, 6.0_c_float], err)
      call assert(err == TGM_ERROR_OK, 'fixture: append B')
      call f%close()

      ! The returned handle is an ordinary tensogram_file: the whole file API works.
      call tensogram_file_open_remote(url, remote, err)
      call assert(err == TGM_ERROR_OK, 'open_remote file:// -> OK')
      call tensogram_file_message_count(remote, n, err)
      call assert(err == TGM_ERROR_OK .and. n == 2, 'open_remote: two messages')
      call tensogram_file_decode_message(remote, 2, msg, err)
      call assert(err == TGM_ERROR_OK, 'open_remote: decode message 2')
      call tensogram_to_array(msg, 1, got, err)
      call assert(err == TGM_ERROR_OK .and. size(got) == 2, 'open_remote: object round-trips')
      call remote%close()

      ! Forward-only walk is accepted too.
      call tensogram_file_open_remote(url, remote, err, bidirectional=.false.)
      call assert(err == TGM_ERROR_OK, 'open_remote forward-only -> OK')
      call remote%close()

      ! A missing object is a remote error, not a local I/O error.
      call tensogram_file_open_remote('file:///nonexistent/missing.tgm', remote, err)
      call assert(err /= TGM_ERROR_OK, 'open_remote missing object -> error')

      open(newunit=unit, file=path, status='old', iostat=ios)
      if (ios == 0) close(unit, status='delete')
   end subroutine open_remote_round_trip

   ! ---- helpers ------------------------------------------------------------

   !> A blank-padded fixed-length source, as an application would pass one.
   function padded() result(s)
      character(len=32) :: s
      s = 's3://bucket/key.tgm'
   end function padded

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_remote: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

end program test_remote

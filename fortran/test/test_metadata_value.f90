! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> Precise metadata access (metadata-access parity).
!>
!> Builds a genuine TWO-object message via the streaming encoder (create-time
!> `base:[{...},{...}]` is preserved 1:1 with the written objects, plus free-form
!> top-level keys that flow into `_extra_`), then exercises the full value-cursor
!> surface: existence (message-level + per-object), precise typed try-get
!> (right-type success, wrong-type -> .false., absent-vs-empty), the value cursor
!> (is_present / kind / as_int / as_uint / as_float / as_bool / as_string /
!> as_bytes),
!> nested-map navigation, array navigation, map enumeration via object(i)
!> (incl. `_reserved_` visibility), the `_extra_` / `_reserved_` section views,
!> per-object scoping, and num_objects.
!>
!> Note on integer range: values are supplied as JSON, and the encode path maps
!> a JSON number > i64::MAX to a CBOR *float* (not a u64 integer). So a genuine
!> "u64-only" integer is not reachable here; as_uint is exercised on a
!> non-negative i64 integer (-> .true.) and on a negative one (-> .false.).
program test_metadata_value
   use, intrinsic :: iso_c_binding
   use tensogram
   implicit none

   character(len=*), parameter :: PATH = 'test_metadata_value.tgm'

   ! base[0] carries the rich payload; base[1] shares some keys and adds one of
   ! its own; `project` / `count` are free-form -> _extra_.
   character(len=*), parameter :: META_JSON = &
      '{"base":[{' //                                                       &
         '"name":"alpha","units":"K","level":850,"neg":-42,' //            &
         '"big":9223372036854775807,"ratio":0.25,' //                      &
         '"flag":true,"flag_off":false,"empty":"","zero":0,' //            &
         '"mars":{"class":"od","stream":"oper"},' //                       &
         '"levels":[1000,850,500],"only0":"present"' //                    &
      '},{' //                                                             &
         '"name":"beta","units":"m","only1":"here"' //                     &
      '}],' //                                                             &
      '"project":"tgm-parity","count":7}'

   type(tensogram_streaming_encoder) :: enc
   type(tensogram_file)              :: f
   type(tensogram_message)           :: msg
   type(tensogram_metadata)          :: meta

   ! Value cursors (non-owning views; borrowed until `meta` is freed).
   type(tensogram_value) :: v, w, w2, mars, cls, lv, e1, o0, o1, ex, rv, resv, tns

   real(c_float)      :: d0(2,2), d1(3,2)
   character(len=:), allocatable :: s, k
   integer(c_int8_t), allocatable :: raw(:)
   integer(c_int64_t) :: i64, u64
   real(c_double)     :: r
   logical            :: b, ok
   integer(c_int)     :: err
   integer            :: i, j, nkeys, npass
   logical            :: found_name, found_reserved, found_only0, found_only1

   npass = 0

   do j = 1, 2; do i = 1, 2; d0(i, j) = real(i + j, c_float); end do; end do
   do j = 1, 2; do i = 1, 3; d1(i, j) = real(i * j, c_float); end do; end do

   ! ---- Build the two-object fixture via the streaming encoder ------------
   call tensogram_streaming_encoder_create(PATH, enc, err, metadata_json=META_JSON)
   call assert(err == TGM_ERROR_OK, 'streaming create')
   call tensogram_streaming_encoder_write(enc, d0, err)
   call assert(err == TGM_ERROR_OK, 'streaming write object 0')
   call tensogram_streaming_encoder_write(enc, d1, err)
   call assert(err == TGM_ERROR_OK, 'streaming write object 1')
   call tensogram_streaming_encoder_finish(enc, err)
   call assert(err == TGM_ERROR_OK, 'streaming finish')
   call enc%free()

   call tensogram_file_open(PATH, f, err)
   call assert(err == TGM_ERROR_OK, 'file open')
   call tensogram_file_decode_message(f, 1, msg, err)
   call assert(err == TGM_ERROR_OK, 'decode message')
   call tensogram_message_metadata(msg, meta, err)
   call assert(err == TGM_ERROR_OK, 'message_metadata')

   ! ---- num_objects -------------------------------------------------------
   call assert(tensogram_metadata_num_objects(meta) == 2, 'num_objects == 2')

   ! ---- has: present / absent, message-level + per-object -----------------
   call assert(tensogram_metadata_has(meta, 'name'), 'has(name) [base]')
   call assert(tensogram_metadata_has(meta, 'project'), 'has(project) [_extra_ fallback]')
   call assert(.not. tensogram_metadata_has(meta, 'missing'), 'has(missing) -> .false.')
   call assert(tensogram_metadata_has(meta, 'name', obj_index=1), 'has_at(1, name)')
   call assert(tensogram_metadata_has(meta, 'only0', obj_index=1), 'has_at(1, only0)')
   call assert(.not. tensogram_metadata_has(meta, 'only0', obj_index=2), 'has_at(2, only0) -> .false.')
   call assert(tensogram_metadata_has(meta, 'only1', obj_index=2), 'has_at(2, only1)')
   call assert(.not. tensogram_metadata_has(meta, 'only1', obj_index=1), 'has_at(1, only1) -> .false.')
   ! per-object has does NOT fall back to _extra_:
   call assert(.not. tensogram_metadata_has(meta, 'project', obj_index=1), 'has_at(1, project) -> .false. (no extra)')
   ! stored empty / zero are present (not absent):
   call assert(tensogram_metadata_has(meta, 'empty', obj_index=1), 'has_at(1, empty) present')
   call assert(tensogram_metadata_has(meta, 'zero', obj_index=1),  'has_at(1, zero) present')
   ! _reserved_ hidden from path getters at the first segment:
   call assert(.not. tensogram_metadata_has(meta, '_reserved_'), 'has(_reserved_) hidden from path')

   ! ---- try_get_* : right-type success ------------------------------------
   ok = tensogram_metadata_try_get_string(meta, 'name', s)
   call assert(ok .and. s == 'alpha', 'try_get_string(name) == alpha')
   ok = tensogram_metadata_try_get_int(meta, 'level', i64)
   call assert(ok .and. i64 == 850_c_int64_t, 'try_get_int(level) == 850')
   ok = tensogram_metadata_try_get_float(meta, 'ratio', r)
   call assert(ok .and. abs(r - 0.25_c_double) < 1.0e-12_c_double, 'try_get_float(ratio) == 0.25')
   ok = tensogram_metadata_try_get_bool(meta, 'flag', b)
   call assert(ok .and. b, 'try_get_bool(flag) == .true.')
   ok = tensogram_metadata_try_get_bool(meta, 'flag_off', b)
   call assert(ok .and. (.not. b), 'try_get_bool(flag_off) == .false. (found, not absent)')
   ! integer widened to float:
   ok = tensogram_metadata_try_get_float(meta, 'level', r)
   call assert(ok .and. abs(r - 850.0_c_double) < 1.0e-9_c_double, 'try_get_float(level) widens int')

   ! ---- try_get_* : wrong type -> .false. ---------------------------------
   call assert(.not. tensogram_metadata_try_get_int(meta, 'name', i64), 'try_get_int(string) -> .false.')
   call assert(.not. tensogram_metadata_try_get_string(meta, 'level', s), 'try_get_string(int) -> .false.')
   call assert(.not. tensogram_metadata_try_get_bool(meta, 'level', b), 'try_get_bool(int) -> .false.')
   call assert(.not. tensogram_metadata_try_get_float(meta, 'name', r), 'try_get_float(string) -> .false.')
   call assert(.not. tensogram_metadata_try_get_int(meta, 'ratio', i64), 'try_get_int(float) -> .false. (no coercion)')

   ! ---- absent vs empty (the default-hiding bug) --------------------------
   ok = tensogram_metadata_try_get_string(meta, 'empty', s)
   call assert(ok .and. len(s) == 0, 'try_get_string(empty) -> found, ""')
   ok = tensogram_metadata_try_get_int(meta, 'zero', i64)
   call assert(ok .and. i64 == 0_c_int64_t, 'try_get_int(zero) -> found, 0')
   call assert(.not. tensogram_metadata_try_get_string(meta, 'missing', s), 'try_get_string(missing) -> .false.')
   call assert(.not. tensogram_metadata_try_get_int(meta, 'missing', i64), 'try_get_int(missing) -> .false.')

   ! ---- get() cursor: presence + kind() -----------------------------------
   v = tensogram_metadata_get(meta, 'name')
   call assert(v%is_present(), 'get(name) present')
   call assert(v%kind() == TGM_VALUE_TYPE_STRING, 'kind(name) == STRING')
   v = tensogram_metadata_get(meta, 'missing')
   call assert(.not. v%is_present(), 'get(missing) not present')
   v = tensogram_metadata_get(meta, 'level')
   call assert(v%kind() == TGM_VALUE_TYPE_INT, 'kind(level) == INT')
   v = tensogram_metadata_get(meta, 'ratio')
   call assert(v%kind() == TGM_VALUE_TYPE_FLOAT, 'kind(ratio) == FLOAT')
   v = tensogram_metadata_get(meta, 'flag')
   call assert(v%kind() == TGM_VALUE_TYPE_BOOL, 'kind(flag) == BOOL')
   v = tensogram_metadata_get(meta, 'mars')
   call assert(v%kind() == TGM_VALUE_TYPE_MAP, 'kind(mars) == MAP')
    v = tensogram_metadata_get(meta, 'levels')
    call assert(v%kind() == TGM_VALUE_TYPE_ARRAY, 'kind(levels) == ARRAY')

    ! ---- as_bytes: wrong-type contract. (Byte-string metadata is produced by
    ! the Rust side, e.g. GRIB local sections; the Fortran/JSON encode path
    ! cannot emit it, so the found=.true. path is covered by the Rust oracle.)
    v = tensogram_metadata_get(meta, 'name')
    call assert(.not. v%as_bytes(raw), 'as_bytes(string) -> .false. (wrong type)')

   ! ---- as_int / as_uint range semantics ----------------------------------
   v = tensogram_metadata_get(meta, 'neg')
   ok = v%as_int(i64)
   call assert(ok .and. i64 == -42_c_int64_t, 'neg as_int == -42')
   call assert(.not. v%as_uint(u64), 'neg as_uint -> .false. (negative)')
   v = tensogram_metadata_get(meta, 'big')
   ok = v%as_uint(u64)
   call assert(ok .and. u64 == 9223372036854775807_c_int64_t, 'big as_uint == i64::MAX')
   ok = v%as_int(i64)
   call assert(ok .and. i64 == 9223372036854775807_c_int64_t, 'big as_int == i64::MAX')

   ! ---- nested map navigation ---------------------------------------------
   mars = tensogram_metadata_get(meta, 'mars')
   cls  = mars%get('class')
   ok = cls%as_string(s)
   call assert(ok .and. s == 'od', 'mars -> get(class) -> as_string == od')
   w = mars%get('nope')
   call assert(.not. w%is_present(), 'mars -> get(nope) absent')
   ! dot-path equivalent (message-level path walker):
   v = tensogram_metadata_get(meta, 'mars.class')
   ok = v%as_string(s)
   call assert(ok .and. s == 'od', 'get(mars.class) dot-path == od')

   ! ---- array navigation ---------------------------------------------------
   lv = tensogram_metadata_get(meta, 'levels')
   call assert(lv%len() == 3, 'levels len() == 3')
   e1 = lv%elem(1); ok = e1%as_int(i64)
   call assert(ok .and. i64 == 1000_c_int64_t, 'levels(1) == 1000')
   e1 = lv%elem(2); ok = e1%as_int(i64)
   call assert(ok .and. i64 == 850_c_int64_t, 'levels(2) == 850')
   e1 = lv%elem(3); ok = e1%as_int(i64)
   call assert(ok .and. i64 == 500_c_int64_t, 'levels(3) == 500')
   w = lv%elem(4)
   call assert(.not. w%is_present(), 'levels(4) out of range absent')

   ! ---- map enumeration via object(i) (incl. _reserved_ visibility) -------
   o0 = tensogram_metadata_object(meta, 1)
   call assert(o0%is_present(), 'object(1) present')
   nkeys = o0%len()
   call assert(nkeys > 0, 'object(1) len() > 0')
   found_name = .false.; found_reserved = .false.
   found_only0 = .false.; found_only1 = .false.
   do i = 1, nkeys
      k = o0%key(i)
      select case (k)
      case ('name');       found_name = .true.
      case ('_reserved_'); found_reserved = .true.
      case ('only0');      found_only0 = .true.
      case ('only1');      found_only1 = .true.
      end select
   end do
   call assert(found_name,     'object(1) enumerates key: name')
   call assert(found_reserved, 'object(1) enumerates key: _reserved_ (parity)')
    call assert(found_only0,    'object(1) enumerates key: only0')
    call assert(.not. found_only1, 'object(1) does NOT contain only1 (per-object)')
    ! An out-of-range key reports found=.false. (distinct from a present
    ! empty-string key, which would be '' with found=.true.).
    k = o0%key(nkeys + 1, found=ok)
    call assert(.not. ok, 'key(out-of-range) reports found=.false. (absent)')

   ! _reserved_ reachable via the section map (nav to tensor.dtype):
   resv = o0%get('_reserved_')
   call assert(resv%is_present(), 'object(1) get(_reserved_) present')
   tns = resv%get('tensor')
   w = tns%get('dtype')
   ok = w%as_string(s)
   call assert(ok .and. s == 'float32', 'object(1) _reserved_.tensor.dtype == float32')

   ! ---- per-object scoping via object(i) and get(...,obj_index) -----------
   o1 = tensogram_metadata_object(meta, 2)
   w = o1%get('name'); ok = w%as_string(s)
   call assert(ok .and. s == 'beta', 'object(2) name == beta')
   v = tensogram_metadata_get(meta, 'only0', obj_index=1); ok = v%as_string(s)
   call assert(ok .and. s == 'present', 'get_at(1, only0) == present')
   v = tensogram_metadata_get(meta, 'only0', obj_index=2)
   call assert(.not. v%is_present(), 'get_at(2, only0) absent')
   v = tensogram_metadata_get(meta, 'only1', obj_index=2); ok = v%as_string(s)
   call assert(ok .and. s == 'here', 'get_at(2, only1) == here')

   ! ---- _extra_ / _reserved_ section views --------------------------------
   ex = tensogram_metadata_extra(meta)
   call assert(ex%is_present(), 'extra() present')
   call assert(ex%len() >= 2, 'extra() len() >= 2')
   w = ex%get('project'); ok = w%as_string(s)
   call assert(ok .and. s == 'tgm-parity', 'extra() project == tgm-parity')
   w = ex%get('count'); ok = w%as_int(i64)
   call assert(ok .and. i64 == 7_c_int64_t, 'extra() count == 7')

   rv = tensogram_metadata_reserved(meta)
   call assert(rv%is_present(), 'reserved() present')
   w  = rv%get('encoder')
   w2 = w%get('name'); ok = w2%as_string(s)
   call assert(ok .and. s == 'tensogram', 'reserved() encoder.name == tensogram')

   call f%close()

   print '(a,i0,a)', 'test_metadata_value: PASS (', npass, ' checks)'

contains

   subroutine assert(cond, what)
      logical,          intent(in) :: cond
      character(len=*), intent(in) :: what
      if (.not. cond) then
         print '(a,a)', 'test_metadata_value: FAIL: ', what
         error stop 1
      end if
      npass = npass + 1
   end subroutine assert

end program test_metadata_value

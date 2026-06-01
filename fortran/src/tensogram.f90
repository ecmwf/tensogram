! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> tensogram — Fortran 2018 interface to the Tensogram C ABI (libtensogram).
!> (Requires F2018 for assumed-rank dummies + select rank; the contiguous
!> attribute it leans on is F2008.)
!>
!> Shallow binding: a private `bind(C)` interface block mirrors the synchronous
!> subset of `tensogram.h`; the public `tensogram_*` procedures add idiomatic
!> Fortran strings, ownership (finalizers), the memory-order contract, and a
!> non-copyable handle guard. See PLAN_FORTRAN.md for the full design.
!>
!> GENERIC ENCODE / DECODE
!>   `tensogram_encode` and `tensogram_to_array` are generic over dtype
!>   (real32, real64, int32, int64) and rank (0..7), using assumed-rank
!>   dummies. (int8/int16/complex/float16 are documented follow-ups; Fortran
!>   has no native unsigned or half/complex-as-pairs scalar mapping.)
!>
!> MEMORY-ORDER CONTRACT (PLAN_FORTRAN.md §5.1)
!>   A Fortran array `a(ni, nj, ...)` is written with the on-wire descriptor
!>   shape and strides REVERSED to C/row-major order. Consequences:
!>     * Round-trips Fortran <-> Fortran are bit-identical.
!>     * A NumPy / C reader sees the reversed (transposed) shape.
!>
!> OWNERSHIP / LIFETIMES (PLAN_FORTRAN.md §5.4 — approach "A+")
!>   `tensogram_buffer`  owns Rust-allocated encoded bytes (tgm_bytes_free).
!>   `tensogram_message` owns a decoded message handle  (tgm_message_free).
!>   Both free in a `final` procedure and via an idempotent `free` binding.
!>   The types are NON-COPYABLE: a defined `assignment(=)` calls `error stop`,
!>   so an accidental `b = a` aborts loudly at the copy site instead of
!>   aliasing the handle and double-freeing later. Pass handles by reference.
module tensogram
   use, intrinsic :: iso_c_binding
   use, intrinsic :: iso_fortran_env, only : error_unit
   implicit none
   private

   ! ---- Public API ---------------------------------------------------------
   public :: tensogram_buffer, tensogram_message
   public :: tensogram_encode      !> generic: real32/64, int32/64, rank 0..7
   public :: tensogram_decode
   public :: tensogram_to_array    !> generic: real32/64, int32/64, rank 0..7
   public :: tensogram_file
   public :: tensogram_file_open, tensogram_file_create
   public :: tensogram_file_message_count
   public :: tensogram_file_append   !> generic: real32/64, int32/64, rank 0..7
   public :: tensogram_file_decode_message, tensogram_file_read_message
   public :: tensogram_meta            !> build per-object application metadata
   public :: tensogram_metadata, tensogram_message_metadata
   public :: tensogram_metadata_get_string, tensogram_metadata_get_int
   public :: tensogram_metadata_get_float
   public :: tensogram_streaming_encoder
   public :: tensogram_streaming_encoder_create, tensogram_streaming_encoder_write
   public :: tensogram_streaming_encoder_finish, tensogram_streaming_encoder_count
   public :: tensogram_num_objects, tensogram_object_ndim
   public :: tensogram_object_shape, tensogram_object_dtype
   public :: tensogram_strerror, tensogram_last_error, tensogram_check
   public :: TGM_ERROR_OK, TGM_ERROR_FRAMING, TGM_ERROR_METADATA,            &
             TGM_ERROR_ENCODING, TGM_ERROR_COMPRESSION, TGM_ERROR_OBJECT,    &
             TGM_ERROR_IO, TGM_ERROR_HASH_MISMATCH, TGM_ERROR_INVALID_ARG,   &
             TGM_ERROR_END_OF_ITER, TGM_ERROR_REMOTE, TGM_ERROR_MISSING_HASH,&
             TGM_ERROR_TIMEOUT, TGM_ERROR_CANCELLED

   ! ---- tgm_error enum — mirror of tensogram.h (CI checks they match) ------
   integer(c_int), parameter :: TGM_ERROR_OK            = 0
   integer(c_int), parameter :: TGM_ERROR_FRAMING       = 1
   integer(c_int), parameter :: TGM_ERROR_METADATA      = 2
   integer(c_int), parameter :: TGM_ERROR_ENCODING      = 3
   integer(c_int), parameter :: TGM_ERROR_COMPRESSION   = 4
   integer(c_int), parameter :: TGM_ERROR_OBJECT        = 5
   integer(c_int), parameter :: TGM_ERROR_IO            = 6
   integer(c_int), parameter :: TGM_ERROR_HASH_MISMATCH = 7
   integer(c_int), parameter :: TGM_ERROR_INVALID_ARG   = 8
   integer(c_int), parameter :: TGM_ERROR_END_OF_ITER   = 9
   integer(c_int), parameter :: TGM_ERROR_REMOTE        = 10
   integer(c_int), parameter :: TGM_ERROR_MISSING_HASH  = 11
   integer(c_int), parameter :: TGM_ERROR_TIMEOUT       = 12
   integer(c_int), parameter :: TGM_ERROR_CANCELLED     = 13

   ! ---- Interoperable POD struct: tgm_bytes_t ------------------------------
   type, bind(C) :: tgm_bytes_t
      type(c_ptr)       :: data = c_null_ptr
      integer(c_size_t) :: len  = 0_c_size_t
   end type tgm_bytes_t

   ! ---- Owned encoded buffer (RAII over tgm_bytes_t) -----------------------
   type :: tensogram_buffer
      type(tgm_bytes_t), private :: raw
   contains
      procedure :: as_array => buffer_as_array
      procedure :: size     => buffer_size
      procedure :: free     => buffer_free
      procedure, private :: buffer_assign
      generic, public :: assignment(=) => buffer_assign
      final     :: buffer_final
   end type tensogram_buffer

   ! ---- Owned decoded message (RAII over tgm_message_t*) -------------------
   type :: tensogram_message
      type(c_ptr), private :: ptr = c_null_ptr
   contains
      procedure :: free => message_free
      procedure, private :: message_assign
      generic, public :: assignment(=) => message_assign
      final     :: message_final
   end type tensogram_message

   ! ---- Owned file handle (RAII over tgm_file_t*) --------------------------
   type :: tensogram_file
      type(c_ptr), private :: ptr = c_null_ptr
   contains
      procedure :: close => file_close
      procedure, private :: file_assign
      generic, public :: assignment(=) => file_assign
      final     :: file_final
   end type tensogram_file

   ! ---- Application-metadata builder (one object's `base[i]` entry) ---------
   !  Accumulate key/value pairs (zero-dependency, JSON-escaped) and emit a
   !  `"base":[{...}]` fragment to pass as `metadata_json` on encode/append.
   type :: tensogram_meta
      character(len=:), allocatable, private :: body
   contains
      procedure :: add_string => meta_add_string
      procedure :: add_int    => meta_add_int
      procedure :: add_real   => meta_add_real
      procedure :: base_json  => meta_base_json
   end type tensogram_meta

   ! ---- Owned metadata handle (RAII over tgm_metadata_t*) ------------------
   type :: tensogram_metadata
      type(c_ptr), private :: ptr = c_null_ptr
   contains
      procedure :: free => metadata_free
      procedure, private :: metadata_assign
      generic, public :: assignment(=) => metadata_assign
      final     :: metadata_final
   end type tensogram_metadata

   ! ---- Owned streaming encoder (RAII over tgm_streaming_encoder_t*) -------
   !  Writes a single multi-object message to a file progressively: the
   !  preamble + header metadata on create, one data-object frame per write,
   !  and the footer + postamble on finish. The handle is freed via `free`
   !  (which ABANDONS an unfinished stream — call finish first for a valid
   !  file); the finalizer only releases the handle, it does not finish.
   type :: tensogram_streaming_encoder
      type(c_ptr), private :: ptr = c_null_ptr
   contains
      procedure :: free => stream_enc_free
      procedure, private :: stream_enc_assign
      generic, public :: assignment(=) => stream_enc_assign
      final     :: stream_enc_final
   end type tensogram_streaming_encoder

   ! ---- Generic encode / decode / append over dtype ------------------------
   interface tensogram_encode
      module procedure encode_f32, encode_f64, encode_i32, encode_i64
   end interface tensogram_encode

   interface tensogram_to_array
      module procedure to_array_f32, to_array_f64, to_array_i32, to_array_i64
   end interface tensogram_to_array

   interface tensogram_file_append
      module procedure append_f32, append_f64, append_i32, append_i64
   end interface tensogram_file_append

   interface tensogram_streaming_encoder_write
      module procedure stream_write_f32, stream_write_f64, stream_write_i32, stream_write_i64
   end interface tensogram_streaming_encoder_write

   ! =========================================================================
   !  Raw C ABI — synchronous subset of tensogram.h.
   ! =========================================================================
   interface
      function c_tgm_last_error() bind(C, name="tgm_last_error") result(p)
         import :: c_ptr
         type(c_ptr) :: p
      end function

      function c_tgm_error_string(err) bind(C, name="tgm_error_string") result(p)
         import :: c_ptr, c_int
         integer(c_int), value :: err
         type(c_ptr) :: p
      end function

      subroutine c_tgm_bytes_free(buf) bind(C, name="tgm_bytes_free")
         import :: tgm_bytes_t
         type(tgm_bytes_t), value :: buf     ! struct passed BY VALUE
      end subroutine

      function c_tgm_encode(meta, ptrs, lens, n, hash, threads, out) &
            bind(C, name="tgm_encode") result(err)
         import :: c_ptr, c_size_t, c_int32_t, c_int, tgm_bytes_t
         type(c_ptr),        value       :: meta
         type(c_ptr),        value       :: ptrs
         type(c_ptr),        value       :: lens
         integer(c_size_t),  value       :: n
         type(c_ptr),        value       :: hash
         integer(c_int32_t), value       :: threads
         type(tgm_bytes_t),  intent(out) :: out
         integer(c_int)                  :: err
      end function

      function c_tgm_decode(buf, buf_len, native_bo, threads, verify, out) &
            bind(C, name="tgm_decode") result(err)
         import :: c_ptr, c_size_t, c_int32_t, c_int
         type(c_ptr),        value       :: buf
         integer(c_size_t),  value       :: buf_len
         integer(c_int32_t), value       :: native_bo
         integer(c_int32_t), value       :: threads
         integer(c_int32_t), value       :: verify
         type(c_ptr),        intent(out) :: out
         integer(c_int)                  :: err
      end function

      function c_tgm_message_num_objects(msg) &
            bind(C, name="tgm_message_num_objects") result(n)
         import :: c_ptr, c_size_t
         type(c_ptr), value :: msg
         integer(c_size_t)  :: n
      end function

      function c_tgm_object_ndim(msg, idx) bind(C, name="tgm_object_ndim") result(n)
         import :: c_ptr, c_size_t, c_int64_t
         type(c_ptr),       value :: msg
         integer(c_size_t), value :: idx
         integer(c_int64_t)       :: n
      end function

      function c_tgm_object_shape(msg, idx) bind(C, name="tgm_object_shape") result(p)
         import :: c_ptr, c_size_t
         type(c_ptr),       value :: msg
         integer(c_size_t), value :: idx
         type(c_ptr)              :: p
      end function

      function c_tgm_object_dtype(msg, idx) bind(C, name="tgm_object_dtype") result(p)
         import :: c_ptr, c_size_t
         type(c_ptr),       value :: msg
         integer(c_size_t), value :: idx
         type(c_ptr)              :: p
      end function

      function c_tgm_object_data(msg, idx, out_len) &
            bind(C, name="tgm_object_data") result(p)
         import :: c_ptr, c_size_t
         type(c_ptr),       value       :: msg
         integer(c_size_t), value       :: idx
         integer(c_size_t), intent(out) :: out_len
         type(c_ptr)                    :: p
      end function

      subroutine c_tgm_message_free(msg) bind(C, name="tgm_message_free")
         import :: c_ptr
         type(c_ptr), value :: msg
      end subroutine

      function c_strlen(s) bind(C, name="strlen") result(n)
         import :: c_ptr, c_size_t
         type(c_ptr), value :: s
         integer(c_size_t)  :: n
      end function

      function c_tgm_file_open(path, out) bind(C, name="tgm_file_open") result(err)
         import :: c_ptr, c_int
         type(c_ptr), value       :: path
         type(c_ptr), intent(out) :: out
         integer(c_int)           :: err
      end function

      function c_tgm_file_create(path, out) bind(C, name="tgm_file_create") result(err)
         import :: c_ptr, c_int
         type(c_ptr), value       :: path
         type(c_ptr), intent(out) :: out
         integer(c_int)           :: err
      end function

      function c_tgm_file_message_count(file, cnt) &
            bind(C, name="tgm_file_message_count") result(err)
         import :: c_ptr, c_size_t, c_int
         type(c_ptr), value             :: file
         integer(c_size_t), intent(out) :: cnt
         integer(c_int)                 :: err
      end function

      function c_tgm_file_decode_message(file, idx, native_bo, threads, verify, out) &
            bind(C, name="tgm_file_decode_message") result(err)
         import :: c_ptr, c_size_t, c_int32_t, c_int
         type(c_ptr),        value       :: file
         integer(c_size_t),  value       :: idx
         integer(c_int32_t), value       :: native_bo
         integer(c_int32_t), value       :: threads
         integer(c_int32_t), value       :: verify
         type(c_ptr),        intent(out) :: out
         integer(c_int)                  :: err
      end function

      function c_tgm_file_read_message(file, idx, out) &
            bind(C, name="tgm_file_read_message") result(err)
         import :: c_ptr, c_size_t, c_int, tgm_bytes_t
         type(c_ptr),       value       :: file
         integer(c_size_t), value       :: idx
         type(tgm_bytes_t), intent(out) :: out
         integer(c_int)                 :: err
      end function

      function c_tgm_file_append(file, meta, ptrs, lens, n, hash, threads) &
            bind(C, name="tgm_file_append") result(err)
         import :: c_ptr, c_size_t, c_int32_t, c_int
         type(c_ptr),        value :: file
         type(c_ptr),        value :: meta
         type(c_ptr),        value :: ptrs
         type(c_ptr),        value :: lens
         integer(c_size_t),  value :: n
         type(c_ptr),        value :: hash
         integer(c_int32_t), value :: threads
         integer(c_int)            :: err
      end function

      subroutine c_tgm_file_close(file) bind(C, name="tgm_file_close")
         import :: c_ptr
         type(c_ptr), value :: file
      end subroutine

      function c_tgm_message_metadata(msg, out) &
            bind(C, name="tgm_message_metadata") result(err)
         import :: c_ptr, c_int
         type(c_ptr), value       :: msg
         type(c_ptr), intent(out) :: out
         integer(c_int)           :: err
      end function

      function c_tgm_metadata_get_string(meta, key) &
            bind(C, name="tgm_metadata_get_string") result(p)
         import :: c_ptr
         type(c_ptr), value :: meta
         type(c_ptr), value :: key
         type(c_ptr)        :: p
      end function

      function c_tgm_metadata_get_int(meta, key, default_val) &
            bind(C, name="tgm_metadata_get_int") result(v)
         import :: c_ptr, c_int64_t
         type(c_ptr),        value :: meta
         type(c_ptr),        value :: key
         integer(c_int64_t), value :: default_val
         integer(c_int64_t)        :: v
      end function

      function c_tgm_metadata_get_float(meta, key, default_val) &
            bind(C, name="tgm_metadata_get_float") result(v)
         import :: c_ptr, c_double
         type(c_ptr),    value :: meta
         type(c_ptr),    value :: key
         real(c_double), value :: default_val
         real(c_double)        :: v
      end function

      subroutine c_tgm_metadata_free(meta) bind(C, name="tgm_metadata_free")
         import :: c_ptr
         type(c_ptr), value :: meta
      end subroutine

      function c_tgm_streaming_encoder_create(path, meta, hash, threads, out) &
            bind(C, name="tgm_streaming_encoder_create") result(err)
         import :: c_ptr, c_int32_t, c_int
         type(c_ptr),        value       :: path
         type(c_ptr),        value       :: meta
         type(c_ptr),        value       :: hash
         integer(c_int32_t), value       :: threads
         type(c_ptr),        intent(out) :: out
         integer(c_int)                  :: err
      end function

      function c_tgm_streaming_encoder_write(enc, descriptor_json, data, data_len) &
            bind(C, name="tgm_streaming_encoder_write") result(err)
         import :: c_ptr, c_size_t, c_int
         type(c_ptr),       value :: enc
         type(c_ptr),       value :: descriptor_json
         type(c_ptr),       value :: data
         integer(c_size_t), value :: data_len
         integer(c_int)           :: err
      end function

      function c_tgm_streaming_encoder_count(enc) &
            bind(C, name="tgm_streaming_encoder_count") result(n)
         import :: c_ptr, c_size_t
         type(c_ptr), value :: enc
         integer(c_size_t)  :: n
      end function

      function c_tgm_streaming_encoder_finish(enc) &
            bind(C, name="tgm_streaming_encoder_finish") result(err)
         import :: c_ptr, c_int
         type(c_ptr), value :: enc
         integer(c_int)     :: err
      end function

      subroutine c_tgm_streaming_encoder_free(enc) &
            bind(C, name="tgm_streaming_encoder_free")
         import :: c_ptr
         type(c_ptr), value :: enc
      end subroutine
   end interface

contains

   ! =========================================================================
   !  String / JSON helpers
   ! =========================================================================

   subroutine f_to_cstr(f, c)
      character(len=*),                    intent(in)  :: f
      character(kind=c_char), allocatable, intent(out) :: c(:)
      integer :: i, n
      n = len(f)
      allocate(c(n + 1))
      do i = 1, n
         c(i) = f(i:i)
      end do
      c(n + 1) = c_null_char
   end subroutine f_to_cstr

   function cptr_to_fstr(p) result(f)
      type(c_ptr), intent(in)         :: p
      character(len=:), allocatable   :: f
      character(kind=c_char), pointer :: buf(:)
      integer(c_size_t) :: n
      integer :: i
      if (.not. c_associated(p)) then
         f = ''
         return
      end if
      n = c_strlen(p)
      call c_f_pointer(p, buf, [n])
      allocate(character(len=int(n)) :: f)
      do i = 1, int(n)
         f(i:i) = buf(i)
      end do
   end function cptr_to_fstr

   pure function itoa(i) result(s)
      integer(c_int64_t), intent(in) :: i
      character(len=:), allocatable  :: s
      character(len=32)              :: tmp
      write (tmp, '(i0)') i
      s = trim(tmp)
   end function itoa

   !> Comma-separated list of int64 values ('' for an empty vector).
   pure function i64list(v) result(s)
      integer(c_int64_t), intent(in) :: v(:)
      character(len=:), allocatable  :: s
      integer :: k
      s = ''
      do k = 1, size(v)
         if (k > 1) s = s // ','
         s = s // itoa(v(k))
      end do
   end function i64list

   pure function host_byte_order() result(bo)
      character(len=:), allocatable :: bo
      integer(c_int8_t) :: bytes(4)
      bytes = transfer(1_c_int32_t, 0_c_int8_t, 4)
      if (bytes(1) == 1_c_int8_t) then
         bo = 'little'
      else
         bo = 'big'
      end if
   end function host_byte_order

   !> Resolve the optional hash-algorithm name to a C string pointer. Fills
   !> `hash_c` (a caller-owned target buffer that must outlive the C call) and
   !> `hash_ptr`: a present-but-empty name means "no hash" (NULL); absent means
   !> the default "xxh3".
   subroutine resolve_hash(hash, hash_c, hash_ptr)
      character(len=*), intent(in), optional :: hash
      character(kind=c_char), allocatable, target, intent(out) :: hash_c(:)
      type(c_ptr),                                 intent(out) :: hash_ptr
      character(len=:), allocatable :: hash_s
      if (present(hash)) then
         hash_s = hash
      else
         hash_s = 'xxh3'
      end if
      if (len(hash_s) == 0) then
         hash_ptr = c_null_ptr
      else
         call f_to_cstr(hash_s, hash_c)
         hash_ptr = c_loc(hash_c)
      end if
   end subroutine resolve_hash

   !> Total element count for a Fortran extent vector (1 for a scalar / empty).
   pure function num_elements(ext) result(n)
      integer(c_int64_t), intent(in) :: ext(:)
      integer(c_size_t) :: n
      n = product([1_c_size_t, int(ext, c_size_t)])
   end function num_elements

   !> Wire payload byte count = element count × element byte width (`bits/8`).
   !> Deriving the width from `storage_size(a)` keeps it consistent with the
   !> actual Fortran type rather than a hard-coded literal per dtype wrapper.
   pure function byte_count(num, bits) result(n)
      integer(c_size_t), intent(in) :: num, bits
      integer(c_size_t) :: n
      n = num * (bits / 8_c_size_t)
   end function byte_count

   !> Build the one-object descriptor JSON. `fshape` is the Fortran
   !> (column-major) extents; the on-wire shape and C-contiguous strides are
   !> the REVERSE (PLAN_FORTRAN.md §5.1). `extra`, if present, is raw JSON of
   !> additional top-level keys appended verbatim.
   !> Build a single DataObjectDescriptor object `{...}` with the on-wire
   !> shape and C-contiguous strides REVERSED from the Fortran extents
   !> (column-major contract). Used directly by the streaming encoder; wrapped
   !> in `{"descriptors":[...]}` by descriptor_json for the buffer/file path.
   function descriptor_object_json(fshape, dtype, encoding, filter, compression) result(js)
      integer(c_int64_t), intent(in)           :: fshape(:)
      character(len=*),   intent(in)           :: dtype
      character(len=*),   intent(in), optional :: encoding, filter, compression
      character(len=:), allocatable :: js, enc_s, flt_s, cmp_s
      integer(c_int64_t), allocatable :: wshape(:), wstrides(:)
      integer :: nd, k
      nd = size(fshape)
      allocate(wshape(nd), wstrides(nd))
      do k = 1, nd
         wshape(k) = fshape(nd - k + 1)          ! reverse to C order
      end do
      if (nd > 0) then
         wstrides(nd) = 1_c_int64_t
         do k = nd - 1, 1, -1
            wstrides(k) = wstrides(k + 1) * wshape(k + 1)
         end do
      end if
      enc_s = 'none'; if (present(encoding))    enc_s = trim(encoding)
      flt_s = 'none'; if (present(filter))      flt_s = trim(filter)
      cmp_s = 'none'; if (present(compression)) cmp_s = trim(compression)
      js = '{"type":"ntensor","ndim":' // itoa(int(nd, c_int64_t)) // &
           ',"shape":['   // i64list(wshape)   // ']' //                   &
           ',"strides":[' // i64list(wstrides) // ']' //                   &
           ',"dtype":"' // trim(dtype) // '","byte_order":"' // host_byte_order() // '"' // &
           ',"encoding":"' // enc_s // '","filter":"' // flt_s // &
           '","compression":"' // cmp_s // '"}'
   end function descriptor_object_json

   !> Wrap one descriptor in the `{"descriptors":[...], <extra>}` envelope the
   !> buffer/file encode entry points expect. `extra` is raw top-level JSON.
   function descriptor_json(fshape, dtype, extra, encoding, filter, compression) result(js)
      integer(c_int64_t), intent(in)           :: fshape(:)
      character(len=*),   intent(in)           :: dtype
      character(len=*),   intent(in), optional :: extra
      character(len=*),   intent(in), optional :: encoding, filter, compression
      character(len=:), allocatable :: js, e
      js = '{"descriptors":[' // &
           descriptor_object_json(fshape, dtype, encoding, filter, compression) // ']'
      if (present(extra)) then
         ! `extra` may be a complete JSON object (`{ ... }`, matching the
         ! streaming metadata argument) or a raw top-level key fragment
         ! (`"base":[...]`, e.g. the output of tensogram_meta%base_json()).
         ! Strip the outer braces of a complete object so the result is a
         ! single valid object either way.
         e = trim(adjustl(extra))
         if (len(e) >= 2) then
            if (e(1:1) == '{' .and. e(len(e):len(e)) == '}') e = e(2:len(e) - 1)
         end if
         if (len_trim(e) > 0) js = js // ',' // e
      end if
      js = js // '}'
   end function descriptor_json

   ! =========================================================================
   !  Error helpers
   ! =========================================================================

   !> Static, human-readable description of a `tgm_error` code.
   function tensogram_strerror(err) result(msg)
      integer(c_int), intent(in)    :: err
      character(len=:), allocatable :: msg
      msg = cptr_to_fstr(c_tgm_error_string(err))
   end function tensogram_strerror

   !> The thread-local detail message left by the most recent failing call
   !> (empty when there is none).
   function tensogram_last_error() result(msg)
      character(len=:), allocatable :: msg
      msg = cptr_to_fstr(c_tgm_last_error())
   end function tensogram_last_error

   !> Convenience guard: a no-op on `TGM_ERROR_OK`; otherwise print `context`
   !> plus the error detail to stderr and `error stop`. Prefer inspecting the
   !> error code directly in library code.
   subroutine tensogram_check(err, context)
      integer(c_int),   intent(in)           :: err
      character(len=*), intent(in), optional :: context
      character(len=:), allocatable :: detail
      if (err == TGM_ERROR_OK) return
      detail = tensogram_last_error()
      if (len(detail) == 0) detail = tensogram_strerror(err)
      if (present(context)) then
         write (error_unit, '(a)') 'tensogram: ' // trim(context) // ': ' // detail
      else
         write (error_unit, '(a)') 'tensogram: ' // detail
      end if
      error stop 1
   end subroutine tensogram_check

   ! =========================================================================
   !  tensogram_buffer methods
   ! =========================================================================

   function buffer_size(self) result(n)
      class(tensogram_buffer), intent(in) :: self
      integer(c_size_t) :: n
      n = self%raw%len
   end function buffer_size

   subroutine buffer_as_array(self, out)
      class(tensogram_buffer), intent(in)          :: self
      integer(c_int8_t), allocatable, intent(out)  :: out(:)
      integer(c_int8_t), pointer :: view(:)
      if (.not. c_associated(self%raw%data) .or. self%raw%len == 0_c_size_t) then
         allocate(out(0))
         return
      end if
      call c_f_pointer(self%raw%data, view, [self%raw%len])
      allocate(out(self%raw%len))
      out = view
   end subroutine buffer_as_array

   subroutine buffer_free(self)
      class(tensogram_buffer), intent(inout) :: self
      if (c_associated(self%raw%data)) then
         call c_tgm_bytes_free(self%raw)
         self%raw%data = c_null_ptr
         self%raw%len  = 0_c_size_t
      end if
   end subroutine buffer_free

   subroutine buffer_final(self)
      type(tensogram_buffer), intent(inout) :: self
      call self%free()
   end subroutine buffer_final

   subroutine buffer_assign(lhs, rhs)
      class(tensogram_buffer), intent(out) :: lhs
      type(tensogram_buffer),  intent(in)  :: rhs
      if (c_associated(rhs%raw%data)) then
         error stop "tensogram_buffer is non-copyable: assigning a live buffer would alias and double-free; pass by reference"
      else
         error stop "tensogram_buffer is non-copyable: do not assign buffers; pass by reference"
      end if
   end subroutine buffer_assign

   ! =========================================================================
   !  tensogram_message methods
   ! =========================================================================

   subroutine message_free(self)
      class(tensogram_message), intent(inout) :: self
      if (c_associated(self%ptr)) then
         call c_tgm_message_free(self%ptr)
         self%ptr = c_null_ptr
      end if
   end subroutine message_free

   subroutine message_final(self)
      type(tensogram_message), intent(inout) :: self
      call self%free()
   end subroutine message_final

   subroutine message_assign(lhs, rhs)
      class(tensogram_message), intent(out) :: lhs
      type(tensogram_message),  intent(in)  :: rhs
      if (c_associated(rhs%ptr)) then
         error stop "tensogram_message is non-copyable: assigning a live handle would alias and double-free; pass by reference"
      else
         error stop "tensogram_message is non-copyable: do not assign handles; pass by reference"
      end if
   end subroutine message_assign

   ! =========================================================================
   !  Encode core (shared by all dtype/rank overloads)
   ! =========================================================================

   !> Encode one tensor (described by `ptr` + `nbytes` + Fortran `fshape` +
   !> `dtype`) into `buf`. `ptr` must reference `nbytes` contiguous bytes that
   !> stay valid for the duration of this call.
   subroutine encode_core(ptr, nbytes, fshape, dtype, buf, err, &
                          metadata_json, hash, encoding, filter, compression)
      type(c_ptr),        intent(in)  :: ptr
      integer(c_size_t),  intent(in)  :: nbytes
      integer(c_int64_t), intent(in)  :: fshape(:)
      character(len=*),   intent(in)  :: dtype
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),     intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression

      character(kind=c_char), allocatable, target :: meta_c(:), hash_c(:)
      character(len=:),       allocatable         :: meta_s
      type(c_ptr),       target :: ptrs(1)
      integer(c_size_t), target :: lens(1)
      type(c_ptr)               :: hash_ptr

      meta_s = descriptor_json(fshape, dtype, metadata_json, encoding, filter, compression)
      call f_to_cstr(meta_s, meta_c)
      call resolve_hash(hash, hash_c, hash_ptr)
      ptrs(1) = ptr
      lens(1) = nbytes
      err = c_tgm_encode(c_loc(meta_c), c_loc(ptrs), c_loc(lens), &
                         1_c_size_t, hash_ptr, 0_c_int32_t, buf%raw)
   end subroutine encode_core

   ! =========================================================================
   !  Generic encode overloads (assumed-rank; c_loc/shape/size work directly)
   ! =========================================================================

   subroutine encode_f32(a, buf, err, metadata_json, hash, encoding, filter, compression)
      real(c_float), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call encode_core(c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'float32', buf, err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine encode_f32

   subroutine encode_f64(a, buf, err, metadata_json, hash, encoding, filter, compression)
      real(c_double), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call encode_core(c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'float64', buf, err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine encode_f64

   subroutine encode_i32(a, buf, err, metadata_json, hash, encoding, filter, compression)
      integer(c_int32_t), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call encode_core(c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'int32', buf, err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine encode_i32

   subroutine encode_i64(a, buf, err, metadata_json, hash, encoding, filter, compression)
      integer(c_int64_t), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call encode_core(c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'int64', buf, err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine encode_i64

   ! =========================================================================
   !  Decode wire bytes -> message handle
   ! =========================================================================

   !> Decode wire-format bytes into a message handle. `verify_hash` (default
   !> `.false.`, to match the core library) checks the per-object xxh3 digest;
   !> `native_byte_order` (default `.true.`) converts to the host byte order.
   subroutine tensogram_decode(wire, msg, err, verify_hash, native_byte_order)
      integer(c_int8_t), target, contiguous, intent(in)  :: wire(:)
      type(tensogram_message),               intent(out) :: msg
      integer(c_int),                        intent(out) :: err
      logical, intent(in), optional :: verify_hash        ! default .false.
      logical, intent(in), optional :: native_byte_order  ! default .true.
      integer(c_int32_t) :: vh, nbo
      type(c_ptr)        :: out
      vh = 0_c_int32_t
      if (present(verify_hash)) vh = merge(1_c_int32_t, 0_c_int32_t, verify_hash)
      nbo = 1_c_int32_t
      if (present(native_byte_order)) &
         nbo = merge(1_c_int32_t, 0_c_int32_t, native_byte_order)
      err = c_tgm_decode(c_loc(wire), size(wire, kind=c_size_t), &
                         nbo, 0_c_int32_t, vh, out)
      if (err == TGM_ERROR_OK) msg%ptr = out
   end subroutine tensogram_decode

   ! =========================================================================
   !  Object metadata accessors
   ! =========================================================================

   !> Number of decoded objects in the message.
   function tensogram_num_objects(msg) result(n)
      type(tensogram_message), intent(in) :: msg
      integer :: n
      n = int(c_tgm_message_num_objects(msg%ptr))
   end function tensogram_num_objects

   !> Rank of object `iobj` (1-based); `0` for an out-of-range index.
   function tensogram_object_ndim(msg, iobj) result(nd)
      type(tensogram_message), intent(in) :: msg
      integer,                 intent(in) :: iobj
      integer :: nd
      nd = int(c_tgm_object_ndim(msg%ptr, int(iobj - 1, c_size_t)))
   end function tensogram_object_ndim

   !> Object extents in FORTRAN (column-major) order — the on-wire shape
   !> REVERSED. ext(1) is the fastest-varying axis.
   function tensogram_object_shape(msg, iobj) result(ext)
      type(tensogram_message), intent(in) :: msg
      integer,                 intent(in) :: iobj
      integer(c_int64_t), allocatable :: ext(:)
      integer(c_int64_t), pointer     :: cshape(:)
      integer     :: nd, k
      type(c_ptr) :: p
      nd = tensogram_object_ndim(msg, iobj)
      allocate(ext(nd))
      if (nd == 0) return
      p = c_tgm_object_shape(msg%ptr, int(iobj - 1, c_size_t))
      call c_f_pointer(p, cshape, [nd])
      do k = 1, nd
         ext(k) = cshape(nd - k + 1)              ! reverse -> Fortran order
      end do
   end function tensogram_object_shape

   !> dtype string of object `iobj` (e.g. `"float32"`); empty for an
   !> out-of-range index.
   function tensogram_object_dtype(msg, iobj) result(dt)
      type(tensogram_message), intent(in) :: msg
      integer,                 intent(in) :: iobj
      character(len=:), allocatable :: dt
      dt = cptr_to_fstr(c_tgm_object_dtype(msg%ptr, int(iobj - 1, c_size_t)))
   end function tensogram_object_dtype

   ! =========================================================================
   !  Decode helper: validate object i against an expected dtype, return the
   !  Fortran extents + a c_ptr to the payload bytes (or err set).
   ! =========================================================================
   subroutine object_payload(msg, iobj, want_dtype, want_rank, ext, dptr, err)
      type(tensogram_message), intent(in)  :: msg
      integer,                 intent(in)  :: iobj
      character(len=*),        intent(in)  :: want_dtype
      integer,                 intent(in)  :: want_rank
      integer(c_int64_t), allocatable, intent(out) :: ext(:)
      type(c_ptr),             intent(out) :: dptr
      integer(c_int),          intent(out) :: err
      integer(c_size_t) :: nbytes
      err = TGM_ERROR_OK
      if (tensogram_object_dtype(msg, iobj) /= want_dtype) then
         err = TGM_ERROR_OBJECT
         return
      end if
      ext = tensogram_object_shape(msg, iobj)
      if (size(ext) /= want_rank) then
         err = TGM_ERROR_OBJECT
         return
      end if
      dptr = c_tgm_object_data(msg%ptr, int(iobj - 1, c_size_t), nbytes)
      if (.not. c_associated(dptr)) err = TGM_ERROR_OBJECT
   end subroutine object_payload

   ! =========================================================================
   !  Generic decode overloads (allocatable assumed-rank out; allocate via
   !  select rank, then a unified flat copy — the reversed-shape contract
   !  makes the C and Fortran flat layouts coincide).
   ! =========================================================================

   subroutine to_array_f32(msg, iobj, out, err)
      type(tensogram_message),            intent(in)  :: msg
      integer,                            intent(in)  :: iobj
      real(c_float), allocatable, target, intent(out) :: out(..)
      integer(c_int),                     intent(out) :: err
      integer(c_int64_t), allocatable :: ext(:)
      type(c_ptr) :: dptr
      real(c_float), pointer :: src(:), dst(:)
      integer(c_size_t) :: nelem
      call object_payload(msg, iobj, 'float32', rank(out), ext, dptr, err)
      if (err /= TGM_ERROR_OK) return
      select rank(out)
      rank(0); allocate(out)
      rank(1); allocate(out(ext(1)))
      rank(2); allocate(out(ext(1),ext(2)))
      rank(3); allocate(out(ext(1),ext(2),ext(3)))
      rank(4); allocate(out(ext(1),ext(2),ext(3),ext(4)))
      rank(5); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5)))
      rank(6); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6)))
      rank(7); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6),ext(7)))
      rank default; err = TGM_ERROR_OBJECT; return
      end select
      nelem = num_elements(ext)
      call c_f_pointer(dptr,       src, [nelem])
      call c_f_pointer(c_loc(out), dst, [nelem])
      dst = src
   end subroutine to_array_f32

   subroutine to_array_f64(msg, iobj, out, err)
      type(tensogram_message),             intent(in)  :: msg
      integer,                             intent(in)  :: iobj
      real(c_double), allocatable, target, intent(out) :: out(..)
      integer(c_int),                      intent(out) :: err
      integer(c_int64_t), allocatable :: ext(:)
      type(c_ptr) :: dptr
      real(c_double), pointer :: src(:), dst(:)
      integer(c_size_t) :: nelem
      call object_payload(msg, iobj, 'float64', rank(out), ext, dptr, err)
      if (err /= TGM_ERROR_OK) return
      select rank(out)
      rank(0); allocate(out)
      rank(1); allocate(out(ext(1)))
      rank(2); allocate(out(ext(1),ext(2)))
      rank(3); allocate(out(ext(1),ext(2),ext(3)))
      rank(4); allocate(out(ext(1),ext(2),ext(3),ext(4)))
      rank(5); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5)))
      rank(6); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6)))
      rank(7); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6),ext(7)))
      rank default; err = TGM_ERROR_OBJECT; return
      end select
      nelem = num_elements(ext)
      call c_f_pointer(dptr,       src, [nelem])
      call c_f_pointer(c_loc(out), dst, [nelem])
      dst = src
   end subroutine to_array_f64

   subroutine to_array_i32(msg, iobj, out, err)
      type(tensogram_message),                 intent(in)  :: msg
      integer,                                 intent(in)  :: iobj
      integer(c_int32_t), allocatable, target, intent(out) :: out(..)
      integer(c_int),                          intent(out) :: err
      integer(c_int64_t), allocatable :: ext(:)
      type(c_ptr) :: dptr
      integer(c_int32_t), pointer :: src(:), dst(:)
      integer(c_size_t) :: nelem
      call object_payload(msg, iobj, 'int32', rank(out), ext, dptr, err)
      if (err /= TGM_ERROR_OK) return
      select rank(out)
      rank(0); allocate(out)
      rank(1); allocate(out(ext(1)))
      rank(2); allocate(out(ext(1),ext(2)))
      rank(3); allocate(out(ext(1),ext(2),ext(3)))
      rank(4); allocate(out(ext(1),ext(2),ext(3),ext(4)))
      rank(5); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5)))
      rank(6); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6)))
      rank(7); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6),ext(7)))
      rank default; err = TGM_ERROR_OBJECT; return
      end select
      nelem = num_elements(ext)
      call c_f_pointer(dptr,       src, [nelem])
      call c_f_pointer(c_loc(out), dst, [nelem])
      dst = src
   end subroutine to_array_i32

   subroutine to_array_i64(msg, iobj, out, err)
      type(tensogram_message),                 intent(in)  :: msg
      integer,                                 intent(in)  :: iobj
      integer(c_int64_t), allocatable, target, intent(out) :: out(..)
      integer(c_int),                          intent(out) :: err
      integer(c_int64_t), allocatable :: ext(:)
      type(c_ptr) :: dptr
      integer(c_int64_t), pointer :: src(:), dst(:)
      integer(c_size_t) :: nelem
      call object_payload(msg, iobj, 'int64', rank(out), ext, dptr, err)
      if (err /= TGM_ERROR_OK) return
      select rank(out)
      rank(0); allocate(out)
      rank(1); allocate(out(ext(1)))
      rank(2); allocate(out(ext(1),ext(2)))
      rank(3); allocate(out(ext(1),ext(2),ext(3)))
      rank(4); allocate(out(ext(1),ext(2),ext(3),ext(4)))
      rank(5); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5)))
      rank(6); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6)))
      rank(7); allocate(out(ext(1),ext(2),ext(3),ext(4),ext(5),ext(6),ext(7)))
      rank default; err = TGM_ERROR_OBJECT; return
      end select
      nelem = num_elements(ext)
      call c_f_pointer(dptr,       src, [nelem])
      call c_f_pointer(c_loc(out), dst, [nelem])
      dst = src
   end subroutine to_array_i64

   ! =========================================================================
   !  tensogram_file methods
   ! =========================================================================

   subroutine file_close(self)
      class(tensogram_file), intent(inout) :: self
      if (c_associated(self%ptr)) then
         call c_tgm_file_close(self%ptr)
         self%ptr = c_null_ptr
      end if
   end subroutine file_close

   subroutine file_final(self)
      type(tensogram_file), intent(inout) :: self
      call self%close()
   end subroutine file_final

   subroutine file_assign(lhs, rhs)
      class(tensogram_file), intent(out) :: lhs
      type(tensogram_file),  intent(in)  :: rhs
      if (c_associated(rhs%ptr)) then
         error stop "tensogram_file is non-copyable: assigning a live handle would alias and double-close; pass by reference"
      else
         error stop "tensogram_file is non-copyable: do not assign handles; pass by reference"
      end if
   end subroutine file_assign

   !> Open an existing .tgm file for reading.
   subroutine tensogram_file_open(path, file, err)
      character(len=*),      intent(in)  :: path
      type(tensogram_file),  intent(out) :: file
      integer(c_int),        intent(out) :: err
      character(kind=c_char), allocatable, target :: path_c(:)
      type(c_ptr) :: out
      call f_to_cstr(path, path_c)
      err = c_tgm_file_open(c_loc(path_c), out)
      if (err == TGM_ERROR_OK) file%ptr = out
   end subroutine tensogram_file_open

   !> Create a new .tgm file for writing (append messages with file_append).
   subroutine tensogram_file_create(path, file, err)
      character(len=*),      intent(in)  :: path
      type(tensogram_file),  intent(out) :: file
      integer(c_int),        intent(out) :: err
      character(kind=c_char), allocatable, target :: path_c(:)
      type(c_ptr) :: out
      call f_to_cstr(path, path_c)
      err = c_tgm_file_create(c_loc(path_c), out)
      if (err == TGM_ERROR_OK) file%ptr = out
   end subroutine tensogram_file_create

   !> Number of messages in the file (may trigger a lazy scan).
   subroutine tensogram_file_message_count(file, count, err)
      type(tensogram_file), intent(in)  :: file
      integer,              intent(out) :: count
      integer(c_int),       intent(out) :: err
      integer(c_size_t) :: cnt
      cnt = 0_c_size_t
      err = c_tgm_file_message_count(file%ptr, cnt)
      count = int(cnt)
   end subroutine tensogram_file_message_count

   !> Decode the message at `index` (1-based) into a handle.
   subroutine tensogram_file_decode_message(file, index, msg, err, &
                                            verify_hash, native_byte_order)
      type(tensogram_file),    intent(in)  :: file
      integer,                 intent(in)  :: index
      type(tensogram_message), intent(out) :: msg
      integer(c_int),          intent(out) :: err
      logical, intent(in), optional :: verify_hash       ! default .false.
      logical, intent(in), optional :: native_byte_order ! default .true.
      integer(c_int32_t) :: vh, nbo
      type(c_ptr) :: out
      vh = 0_c_int32_t
      if (present(verify_hash)) vh = merge(1_c_int32_t, 0_c_int32_t, verify_hash)
      nbo = 1_c_int32_t
      if (present(native_byte_order)) &
         nbo = merge(1_c_int32_t, 0_c_int32_t, native_byte_order)
      err = c_tgm_file_decode_message(file%ptr, int(index - 1, c_size_t), &
                                      nbo, 0_c_int32_t, vh, out)
      if (err == TGM_ERROR_OK) msg%ptr = out
   end subroutine tensogram_file_decode_message

   !> Read the raw message bytes at `index` (1-based) into a buffer.
   subroutine tensogram_file_read_message(file, index, buf, err)
      type(tensogram_file),   intent(in)  :: file
      integer,                intent(in)  :: index
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      err = c_tgm_file_read_message(file%ptr, int(index - 1, c_size_t), buf%raw)
   end subroutine tensogram_file_read_message

   ! =========================================================================
   !  Append core + generic append overloads
   ! =========================================================================

   !> Encode one tensor and append it to `file_ptr` as a new message. Mirrors
   !> encode_core but targets the file-append entry point (no buffer returned).
   subroutine append_core(file_ptr, ptr, nbytes, fshape, dtype, err, &
                          metadata_json, hash, encoding, filter, compression)
      type(c_ptr),        intent(in)  :: file_ptr
      type(c_ptr),        intent(in)  :: ptr
      integer(c_size_t),  intent(in)  :: nbytes
      integer(c_int64_t), intent(in)  :: fshape(:)
      character(len=*),   intent(in)  :: dtype
      integer(c_int),     intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      character(kind=c_char), allocatable, target :: meta_c(:), hash_c(:)
      character(len=:),       allocatable         :: meta_s
      type(c_ptr),       target :: ptrs(1)
      integer(c_size_t), target :: lens(1)
      type(c_ptr)               :: hash_ptr
      meta_s = descriptor_json(fshape, dtype, metadata_json, encoding, filter, compression)
      call f_to_cstr(meta_s, meta_c)
      call resolve_hash(hash, hash_c, hash_ptr)
      ptrs(1) = ptr
      lens(1) = nbytes
      err = c_tgm_file_append(file_ptr, c_loc(meta_c), c_loc(ptrs), c_loc(lens), &
                              1_c_size_t, hash_ptr, 0_c_int32_t)
   end subroutine append_core

   subroutine append_f32(file, a, err, metadata_json, hash, encoding, filter, compression)
      type(tensogram_file),  intent(inout) :: file
      real(c_float), target, contiguous, intent(in) :: a(..)
      integer(c_int),        intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call append_core(file%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'float32', err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine append_f32

   subroutine append_f64(file, a, err, metadata_json, hash, encoding, filter, compression)
      type(tensogram_file),   intent(inout) :: file
      real(c_double), target, contiguous, intent(in) :: a(..)
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call append_core(file%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'float64', err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine append_f64

   subroutine append_i32(file, a, err, metadata_json, hash, encoding, filter, compression)
      type(tensogram_file),       intent(inout) :: file
      integer(c_int32_t), target, contiguous, intent(in) :: a(..)
      integer(c_int),             intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call append_core(file%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'int32', err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine append_i32

   subroutine append_i64(file, a, err, metadata_json, hash, encoding, filter, compression)
      type(tensogram_file),       intent(inout) :: file
      integer(c_int64_t), target, contiguous, intent(in) :: a(..)
      integer(c_int),             intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call append_core(file%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), 'int64', err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine append_i64

   ! =========================================================================
   !  Application-metadata builder (tensogram_meta) + JSON escaping
   ! =========================================================================

   pure function hex2(c) result(h)
      integer, intent(in) :: c
      character(len=2)               :: h
      character(len=16), parameter   :: dig = '0123456789abcdef'
      h(1:1) = dig(c / 16 + 1 : c / 16 + 1)
      h(2:2) = dig(mod(c, 16) + 1 : mod(c, 16) + 1)
   end function hex2

   !> Escape a string for inclusion as a JSON string body (no surrounding
   !> quotes): handles ", \, the short control escapes, and other control
   !> characters as \u00XX.
   pure function json_escape(s) result(o)
      character(len=*), intent(in)  :: s
      character(len=:), allocatable :: o
      integer   :: i, c
      character :: ch
      ! Fast path: most keys/values contain nothing that needs escaping, so
      ! return the input unchanged (a single allocation) rather than rebuilding
      ! it character by character. " (34), \ (92), and control chars (< 32) are
      ! the only characters this escaper rewrites.
      do i = 1, len(s)
         c = iachar(s(i:i))
         if (c == 34 .or. c == 92 .or. c < 32) exit
      end do
      if (i > len(s)) then
         o = s
         return
      end if
      ! Slow path: at least one character needs escaping.
      o = ''
      do i = 1, len(s)
         ch = s(i:i)
         c = iachar(ch)
         select case (c)
         case (34);  o = o // '\"'
         case (92);  o = o // '\\'
         case (8);   o = o // '\b'
         case (9);   o = o // '\t'
         case (10);  o = o // '\n'
         case (12);  o = o // '\f'
         case (13);  o = o // '\r'
         case (0:7, 11, 14:31); o = o // '\u00' // hex2(c)
         case default; o = o // ch
         end select
      end do
   end function json_escape

   subroutine meta_append_pair(self, key, valjson)
      class(tensogram_meta), intent(inout) :: self
      character(len=*),      intent(in)    :: key, valjson
      if (.not. allocated(self%body)) self%body = ''
      if (len(self%body) > 0) self%body = self%body // ','
      self%body = self%body // '"' // json_escape(key) // '":' // valjson
   end subroutine meta_append_pair

   subroutine meta_add_string(self, key, val)
      class(tensogram_meta), intent(inout) :: self
      character(len=*),      intent(in)    :: key, val
      call meta_append_pair(self, key, '"' // json_escape(val) // '"')
   end subroutine meta_add_string

   subroutine meta_add_int(self, key, val)
      class(tensogram_meta), intent(inout) :: self
      character(len=*),      intent(in)    :: key
      integer(c_int64_t),    intent(in)    :: val
      call meta_append_pair(self, key, itoa(val))
   end subroutine meta_add_int

   subroutine meta_add_real(self, key, val)
      class(tensogram_meta), intent(inout) :: self
      character(len=*),      intent(in)    :: key
      real(c_double),        intent(in)    :: val
      character(len=64) :: tmp
      write (tmp, '(g0)') val
      call meta_append_pair(self, key, trim(adjustl(tmp)))
   end subroutine meta_add_real

   !> Emit the accumulated keys as a `"base":[{...}]` top-level JSON fragment,
   !> suitable as the `metadata_json` argument of encode / append.
   function meta_base_json(self) result(js)
      class(tensogram_meta), intent(in) :: self
      character(len=:), allocatable :: js
      if (allocated(self%body)) then
         js = '"base":[{' // self%body // '}]'
      else
         js = '"base":[{}]'
      end if
   end function meta_base_json

   ! =========================================================================
   !  Metadata handle + getters
   ! =========================================================================

   subroutine metadata_free(self)
      class(tensogram_metadata), intent(inout) :: self
      if (c_associated(self%ptr)) then
         call c_tgm_metadata_free(self%ptr)
         self%ptr = c_null_ptr
      end if
   end subroutine metadata_free

   subroutine metadata_final(self)
      type(tensogram_metadata), intent(inout) :: self
      call self%free()
   end subroutine metadata_final

   subroutine metadata_assign(lhs, rhs)
      class(tensogram_metadata), intent(out) :: lhs
      type(tensogram_metadata),  intent(in)  :: rhs
      if (c_associated(rhs%ptr)) then
         error stop "tensogram_metadata is non-copyable: assigning a live handle would alias and double-free; pass by reference"
      else
         error stop "tensogram_metadata is non-copyable: do not assign handles; pass by reference"
      end if
   end subroutine metadata_assign

   !> Extract an independent metadata handle from a decoded message.
   subroutine tensogram_message_metadata(msg, meta, err)
      type(tensogram_message),  intent(in)  :: msg
      type(tensogram_metadata), intent(out) :: meta
      integer(c_int),           intent(out) :: err
      type(c_ptr) :: out
      err = c_tgm_message_metadata(msg%ptr, out)
      if (err == TGM_ERROR_OK) meta%ptr = out
   end subroutine tensogram_message_metadata

   !> Look up a string value by dot-notation key (searches base[i] then
   !> _extra_). Returns '' when the key is absent or not a string.
   function tensogram_metadata_get_string(meta, key) result(val)
      type(tensogram_metadata), intent(in) :: meta
      character(len=*),         intent(in) :: key
      character(len=:), allocatable :: val
      character(kind=c_char), allocatable, target :: key_c(:)
      call f_to_cstr(key, key_c)
      val = cptr_to_fstr(c_tgm_metadata_get_string(meta%ptr, c_loc(key_c)))
   end function tensogram_metadata_get_string

   !> Look up an integer value by dot-notation key; `default_val` when absent.
   function tensogram_metadata_get_int(meta, key, default_val) result(v)
      type(tensogram_metadata), intent(in) :: meta
      character(len=*),         intent(in) :: key
      integer(c_int64_t),       intent(in) :: default_val
      integer(c_int64_t) :: v
      character(kind=c_char), allocatable, target :: key_c(:)
      call f_to_cstr(key, key_c)
      v = c_tgm_metadata_get_int(meta%ptr, c_loc(key_c), default_val)
   end function tensogram_metadata_get_int

   !> Look up a float value by dot-notation key; `default_val` when absent.
   function tensogram_metadata_get_float(meta, key, default_val) result(v)
      type(tensogram_metadata), intent(in) :: meta
      character(len=*),         intent(in) :: key
      real(c_double),           intent(in) :: default_val
      real(c_double) :: v
      character(kind=c_char), allocatable, target :: key_c(:)
      call f_to_cstr(key, key_c)
      v = c_tgm_metadata_get_float(meta%ptr, c_loc(key_c), default_val)
   end function tensogram_metadata_get_float

   ! =========================================================================
   !  Streaming encoder
   ! =========================================================================

   subroutine stream_enc_free(self)
      class(tensogram_streaming_encoder), intent(inout) :: self
      if (c_associated(self%ptr)) then
         call c_tgm_streaming_encoder_free(self%ptr)
         self%ptr = c_null_ptr
      end if
   end subroutine stream_enc_free

   subroutine stream_enc_final(self)
      type(tensogram_streaming_encoder), intent(inout) :: self
      call self%free()
   end subroutine stream_enc_final

   subroutine stream_enc_assign(lhs, rhs)
      class(tensogram_streaming_encoder), intent(out) :: lhs
      type(tensogram_streaming_encoder),  intent(in)  :: rhs
      if (c_associated(rhs%ptr)) then
         error stop "tensogram_streaming_encoder is non-copyable (live handle); pass by reference"
      else
         error stop "tensogram_streaming_encoder is non-copyable; pass by reference"
      end if
   end subroutine stream_enc_assign

   !> Open a streaming encoder writing a single multi-object message to `path`.
   !> `metadata_json` is free-form message-level GlobalMetadata (NOT a
   !> descriptors map; objects are supplied via _write); defaults to `{}`.
   subroutine tensogram_streaming_encoder_create(path, enc, err, metadata_json, hash)
      character(len=*),                  intent(in)  :: path
      type(tensogram_streaming_encoder), intent(out) :: enc
      integer(c_int),                    intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(kind=c_char), allocatable, target :: path_c(:), meta_c(:), hash_c(:)
      character(len=:),       allocatable         :: meta_s
      type(c_ptr) :: hash_ptr, out
      call f_to_cstr(path, path_c)
      if (present(metadata_json)) then
         meta_s = metadata_json
      else
         meta_s = '{}'
      end if
      call f_to_cstr(meta_s, meta_c)
      call resolve_hash(hash, hash_c, hash_ptr)
      err = c_tgm_streaming_encoder_create(c_loc(path_c), c_loc(meta_c), hash_ptr, &
                                           0_c_int32_t, out)
      if (err == TGM_ERROR_OK) enc%ptr = out
   end subroutine tensogram_streaming_encoder_create

   !> Finalise the stream: write the footer index/hash + postamble and close
   !> the file. The handle remains valid (release it with `enc%free()`).
   subroutine tensogram_streaming_encoder_finish(enc, err)
      type(tensogram_streaming_encoder), intent(inout) :: enc
      integer(c_int),                    intent(out)   :: err
      err = c_tgm_streaming_encoder_finish(enc%ptr)
   end subroutine tensogram_streaming_encoder_finish

   !> Number of data objects written so far.
   function tensogram_streaming_encoder_count(enc) result(n)
      type(tensogram_streaming_encoder), intent(in) :: enc
      integer :: n
      n = int(c_tgm_streaming_encoder_count(enc%ptr))
   end function tensogram_streaming_encoder_count

   !> Encode one tensor (described by `ptr` + `nbytes` + Fortran `fshape` +
   !> `dtype`) and append it to the open stream as a data-object frame.
   subroutine stream_write_core(enc_ptr, ptr, nbytes, fshape, dtype, err, &
                                encoding, filter, compression)
      type(c_ptr),        intent(in)  :: enc_ptr
      type(c_ptr),        intent(in)  :: ptr
      integer(c_size_t),  intent(in)  :: nbytes
      integer(c_int64_t), intent(in)  :: fshape(:)
      character(len=*),   intent(in)  :: dtype
      integer(c_int),     intent(out) :: err
      character(len=*), intent(in), optional :: encoding, filter, compression
      character(kind=c_char), allocatable, target :: desc_c(:)
      character(len=:),       allocatable         :: desc_s
      desc_s = descriptor_object_json(fshape, dtype, encoding, filter, compression)
      call f_to_cstr(desc_s, desc_c)
      err = c_tgm_streaming_encoder_write(enc_ptr, c_loc(desc_c), ptr, nbytes)
   end subroutine stream_write_core

   subroutine stream_write_f32(enc, a, err, encoding, filter, compression)
      type(tensogram_streaming_encoder), intent(inout) :: enc
      real(c_float), target, contiguous, intent(in) :: a(..)
      integer(c_int),                    intent(out) :: err
      character(len=*), intent(in), optional :: encoding, filter, compression
      call stream_write_core(enc%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                             int(shape(a), c_int64_t), 'float32', err, encoding, filter, compression)
   end subroutine stream_write_f32

   subroutine stream_write_f64(enc, a, err, encoding, filter, compression)
      type(tensogram_streaming_encoder), intent(inout) :: enc
      real(c_double), target, contiguous, intent(in) :: a(..)
      integer(c_int),                     intent(out) :: err
      character(len=*), intent(in), optional :: encoding, filter, compression
      call stream_write_core(enc%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                             int(shape(a), c_int64_t), 'float64', err, encoding, filter, compression)
   end subroutine stream_write_f64

   subroutine stream_write_i32(enc, a, err, encoding, filter, compression)
      type(tensogram_streaming_encoder), intent(inout) :: enc
      integer(c_int32_t), target, contiguous, intent(in) :: a(..)
      integer(c_int),                         intent(out) :: err
      character(len=*), intent(in), optional :: encoding, filter, compression
      call stream_write_core(enc%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                             int(shape(a), c_int64_t), 'int32', err, encoding, filter, compression)
   end subroutine stream_write_i32

   subroutine stream_write_i64(enc, a, err, encoding, filter, compression)
      type(tensogram_streaming_encoder), intent(inout) :: enc
      integer(c_int64_t), target, contiguous, intent(in) :: a(..)
      integer(c_int),                         intent(out) :: err
      character(len=*), intent(in), optional :: encoding, filter, compression
      call stream_write_core(enc%ptr, c_loc(a), byte_count(size(a, kind=c_size_t), storage_size(a, kind=c_size_t)), &
                             int(shape(a), c_int64_t), 'int64', err, encoding, filter, compression)
   end subroutine stream_write_i64

end module tensogram

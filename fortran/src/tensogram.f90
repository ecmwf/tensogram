! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> tensogram — Fortran 2008/2018 interface to the Tensogram C ABI (libtensogram).
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

   !> Build the one-object descriptor JSON. `fshape` is the Fortran
   !> (column-major) extents; the on-wire shape and C-contiguous strides are
   !> the REVERSE (PLAN_FORTRAN.md §5.1). `extra`, if present, is raw JSON of
   !> additional top-level keys appended verbatim.
   function descriptor_json(fshape, dtype, extra) result(js)
      integer(c_int64_t), intent(in)         :: fshape(:)
      character(len=*),   intent(in)         :: dtype
      character(len=*),   intent(in), optional :: extra
      character(len=:), allocatable :: js
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
      js = '{"descriptors":[{"type":"ntensor","ndim":' // itoa(int(nd, c_int64_t)) // &
           ',"shape":['   // i64list(wshape)   // ']' //                              &
           ',"strides":[' // i64list(wstrides) // ']' //                              &
           ',"dtype":"' // trim(dtype) // '","byte_order":"' // host_byte_order() // '"' // &
           ',"encoding":"none","filter":"none","compression":"none"}]'
      if (present(extra)) then
         if (len_trim(extra) > 0) js = js // ',' // trim(extra)
      end if
      js = js // '}'
   end function descriptor_json

   ! =========================================================================
   !  Error helpers
   ! =========================================================================

   function tensogram_strerror(err) result(msg)
      integer(c_int), intent(in)    :: err
      character(len=:), allocatable :: msg
      msg = cptr_to_fstr(c_tgm_error_string(err))
   end function tensogram_strerror

   function tensogram_last_error() result(msg)
      character(len=:), allocatable :: msg
      msg = cptr_to_fstr(c_tgm_last_error())
   end function tensogram_last_error

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
   subroutine encode_core(ptr, nbytes, fshape, dtype, buf, err, metadata_json, hash)
      type(c_ptr),        intent(in)  :: ptr
      integer(c_size_t),  intent(in)  :: nbytes
      integer(c_int64_t), intent(in)  :: fshape(:)
      character(len=*),   intent(in)  :: dtype
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),     intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash

      character(kind=c_char), allocatable, target :: meta_c(:), hash_c(:)
      character(len=:),       allocatable         :: meta_s, hash_s
      type(c_ptr),       target :: ptrs(1)
      integer(c_size_t), target :: lens(1)
      type(c_ptr)               :: hash_ptr

      meta_s = descriptor_json(fshape, dtype, metadata_json)
      call f_to_cstr(meta_s, meta_c)

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

      ptrs(1) = ptr
      lens(1) = nbytes
      err = c_tgm_encode(c_loc(meta_c), c_loc(ptrs), c_loc(lens), &
                         1_c_size_t, hash_ptr, 0_c_int32_t, buf%raw)
   end subroutine encode_core

   ! =========================================================================
   !  Generic encode overloads (assumed-rank; c_loc/shape/size work directly)
   ! =========================================================================

   subroutine encode_f32(a, buf, err, metadata_json, hash)
      real(c_float), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call encode_core(c_loc(a), size(a, kind=c_size_t) * 4_c_size_t, &
                       int(shape(a), c_int64_t), 'float32', buf, err, metadata_json, hash)
   end subroutine encode_f32

   subroutine encode_f64(a, buf, err, metadata_json, hash)
      real(c_double), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call encode_core(c_loc(a), size(a, kind=c_size_t) * 8_c_size_t, &
                       int(shape(a), c_int64_t), 'float64', buf, err, metadata_json, hash)
   end subroutine encode_f64

   subroutine encode_i32(a, buf, err, metadata_json, hash)
      integer(c_int32_t), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call encode_core(c_loc(a), size(a, kind=c_size_t) * 4_c_size_t, &
                       int(shape(a), c_int64_t), 'int32', buf, err, metadata_json, hash)
   end subroutine encode_i32

   subroutine encode_i64(a, buf, err, metadata_json, hash)
      integer(c_int64_t), target, contiguous, intent(in) :: a(..)
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call encode_core(c_loc(a), size(a, kind=c_size_t) * 8_c_size_t, &
                       int(shape(a), c_int64_t), 'int64', buf, err, metadata_json, hash)
   end subroutine encode_i64

   ! =========================================================================
   !  Decode wire bytes -> message handle
   ! =========================================================================

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

   function tensogram_num_objects(msg) result(n)
      type(tensogram_message), intent(in) :: msg
      integer :: n
      n = int(c_tgm_message_num_objects(msg%ptr))
   end function tensogram_num_objects

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
      nelem = product([1_c_size_t, int(ext, c_size_t)])
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
      nelem = product([1_c_size_t, int(ext, c_size_t)])
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
      nelem = product([1_c_size_t, int(ext, c_size_t)])
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
      nelem = product([1_c_size_t, int(ext, c_size_t)])
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
   subroutine append_core(file_ptr, ptr, nbytes, fshape, dtype, err, metadata_json, hash)
      type(c_ptr),        intent(in)  :: file_ptr
      type(c_ptr),        intent(in)  :: ptr
      integer(c_size_t),  intent(in)  :: nbytes
      integer(c_int64_t), intent(in)  :: fshape(:)
      character(len=*),   intent(in)  :: dtype
      integer(c_int),     intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(kind=c_char), allocatable, target :: meta_c(:), hash_c(:)
      character(len=:),       allocatable         :: meta_s, hash_s
      type(c_ptr),       target :: ptrs(1)
      integer(c_size_t), target :: lens(1)
      type(c_ptr)               :: hash_ptr
      meta_s = descriptor_json(fshape, dtype, metadata_json)
      call f_to_cstr(meta_s, meta_c)
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
      ptrs(1) = ptr
      lens(1) = nbytes
      err = c_tgm_file_append(file_ptr, c_loc(meta_c), c_loc(ptrs), c_loc(lens), &
                              1_c_size_t, hash_ptr, 0_c_int32_t)
   end subroutine append_core

   subroutine append_f32(file, a, err, metadata_json, hash)
      type(tensogram_file),  intent(inout) :: file
      real(c_float), target, contiguous, intent(in) :: a(..)
      integer(c_int),        intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call append_core(file%ptr, c_loc(a), size(a, kind=c_size_t) * 4_c_size_t, &
                       int(shape(a), c_int64_t), 'float32', err, metadata_json, hash)
   end subroutine append_f32

   subroutine append_f64(file, a, err, metadata_json, hash)
      type(tensogram_file),   intent(inout) :: file
      real(c_double), target, contiguous, intent(in) :: a(..)
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call append_core(file%ptr, c_loc(a), size(a, kind=c_size_t) * 8_c_size_t, &
                       int(shape(a), c_int64_t), 'float64', err, metadata_json, hash)
   end subroutine append_f64

   subroutine append_i32(file, a, err, metadata_json, hash)
      type(tensogram_file),       intent(inout) :: file
      integer(c_int32_t), target, contiguous, intent(in) :: a(..)
      integer(c_int),             intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call append_core(file%ptr, c_loc(a), size(a, kind=c_size_t) * 4_c_size_t, &
                       int(shape(a), c_int64_t), 'int32', err, metadata_json, hash)
   end subroutine append_i32

   subroutine append_i64(file, a, err, metadata_json, hash)
      type(tensogram_file),       intent(inout) :: file
      integer(c_int64_t), target, contiguous, intent(in) :: a(..)
      integer(c_int),             intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      call append_core(file%ptr, c_loc(a), size(a, kind=c_size_t) * 8_c_size_t, &
                       int(shape(a), c_int64_t), 'int64', err, metadata_json, hash)
   end subroutine append_i64

end module tensogram

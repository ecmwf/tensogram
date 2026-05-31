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
!> MEMORY-ORDER CONTRACT (PLAN_FORTRAN.md §5.1)
!>   A Fortran array `a(ni, nj)` is written with the on-wire descriptor shape
!>   and strides REVERSED to C/row-major order ([nj, ni], strides [ni, 1]).
!>   Consequences:
!>     * Round-trips Fortran <-> Fortran are bit-identical.
!>     * A NumPy / C reader sees an array of shape (nj, ni) — the transpose.
!>
!> OWNERSHIP / LIFETIMES (PLAN_FORTRAN.md §5.4 — approach "A+")
!>   `tensogram_buffer`  owns Rust-allocated encoded bytes (tgm_bytes_free).
!>   `tensogram_message` owns a decoded message handle  (tgm_message_free).
!>   Both free in a `final` procedure and via an idempotent `free` binding.
!>   The types are NON-COPYABLE: a defined `assignment(=)` calls `error stop`,
!>   so an accidental `b = a` aborts loudly at the copy site instead of
!>   aliasing the handle and double-freeing later. Pass handles by reference;
!>   factory procedures return them via `intent(out)` arguments.
module tensogram
   use, intrinsic :: iso_c_binding
   use, intrinsic :: iso_fortran_env, only : error_unit
   implicit none
   private

   ! ---- Public API ---------------------------------------------------------
   public :: tensogram_buffer, tensogram_message
   public :: tensogram_encode_r2_f32
   public :: tensogram_decode
   public :: tensogram_num_objects, tensogram_object_ndim
   public :: tensogram_object_shape, tensogram_object_dtype
   public :: tensogram_object_to_r2_f32
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
   !  typedef struct { uint8_t *data; size_t len; } tgm_bytes_t;
   type, bind(C) :: tgm_bytes_t
      type(c_ptr)       :: data = c_null_ptr
      integer(c_size_t) :: len  = 0_c_size_t
   end type tgm_bytes_t

   ! ---- Owned encoded buffer (RAII over tgm_bytes_t) -----------------------
   type :: tensogram_buffer
      type(tgm_bytes_t), private :: raw
   contains
      procedure :: as_array => buffer_as_array   !> copy bytes into int8(:)
      procedure :: size     => buffer_size       !> length in bytes
      procedure :: free     => buffer_free       !> idempotent release
      procedure, private :: buffer_assign
      generic, public :: assignment(=) => buffer_assign
      final     :: buffer_final
   end type tensogram_buffer

   ! ---- Owned decoded message (RAII over tgm_message_t*) -------------------
   type :: tensogram_message
      type(c_ptr), private :: ptr = c_null_ptr
   contains
      procedure :: free => message_free          !> idempotent release
      procedure, private :: message_assign
      generic, public :: assignment(=) => message_assign
      final     :: message_final
   end type tensogram_message

   ! =========================================================================
   !  Raw C ABI — synchronous subset of tensogram.h.
   !  Convention: every input pointer (const T*) is `type(c_ptr), value`;
   !  out-handles (T**) are `type(c_ptr), intent(out)`; returned const char* /
   !  const T* are `type(c_ptr)`. cbindgen emits `tgm_error` as a C `int`.
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
   end interface

contains

   ! =========================================================================
   !  String helpers
   ! =========================================================================

   !> Fortran string -> NUL-terminated c_char array (allocatable + target so
   !> the caller may take c_loc(c) for a `const char*` parameter).
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

   !> C `const char*` -> Fortran allocatable string. Returns '' for NULL.
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

   !> Integer -> compact decimal string (no leading blanks).
   pure function itoa(i) result(s)
      integer(c_int64_t), intent(in) :: i
      character(len=:), allocatable  :: s
      character(len=32)              :: tmp
      write (tmp, '(i0)') i
      s = trim(tmp)
   end function itoa

   !> Host byte order as the wire string ("little" or "big").
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

   !> Build the metadata JSON for one rank-2 float32 object. The shape and
   !> strides are REVERSED (column-major contract): on-wire shape [nj, ni],
   !> C-contiguous strides [ni, 1]. `extra`, if given, is a raw JSON fragment
   !> of additional TOP-LEVEL keys appended verbatim
   !> (e.g. '"base":[{"product":{"name":"t"}}]').
   function descriptor_json_r2_f32(ni, nj, extra) result(js)
      integer,          intent(in)           :: ni, nj
      character(len=*), intent(in), optional :: extra
      character(len=:), allocatable          :: js
      integer(c_int64_t) :: ni64, nj64
      ni64 = int(ni, c_int64_t)
      nj64 = int(nj, c_int64_t)
      js = '{"descriptors":[{"type":"ntensor","ndim":2'                    // &
           ',"shape":['   // itoa(nj64) // ',' // itoa(ni64) // ']'        // &
           ',"strides":[' // itoa(ni64) // ',1]'                          // &
           ',"dtype":"float32","byte_order":"' // host_byte_order() // '"'// &
           ',"encoding":"none","filter":"none","compression":"none"}]'
      if (present(extra)) then
         if (len_trim(extra) > 0) js = js // ',' // trim(extra)
      end if
      js = js // '}'
   end function descriptor_json_r2_f32

   ! =========================================================================
   !  Error helpers
   ! =========================================================================

   !> Static description of an error code (from tgm_error_string).
   function tensogram_strerror(err) result(msg)
      integer(c_int), intent(in)    :: err
      character(len=:), allocatable :: msg
      msg = cptr_to_fstr(c_tgm_error_string(err))
   end function tensogram_strerror

   !> The thread-local message left by the most recent failing FFI call.
   function tensogram_last_error() result(msg)
      character(len=:), allocatable :: msg
      msg = cptr_to_fstr(c_tgm_last_error())
   end function tensogram_last_error

   !> Idiomatic guard: no-op on TGM_ERROR_OK, otherwise print details and stop.
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
   !  tensogram_buffer methods (encoded bytes owned by Rust)
   ! =========================================================================

   function buffer_size(self) result(n)
      class(tensogram_buffer), intent(in) :: self
      integer(c_size_t) :: n
      n = self%raw%len
   end function buffer_size

   !> Copy the owned bytes into a fresh Fortran int8 array, decoupling the data
   !> from the buffer's lifetime (safe to free() afterwards).
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
         call c_tgm_bytes_free(self%raw)       ! struct passed by value
         self%raw%data = c_null_ptr
         self%raw%len  = 0_c_size_t
      end if
   end subroutine buffer_free

   subroutine buffer_final(self)
      type(tensogram_buffer), intent(inout) :: self
      call self%free()
   end subroutine buffer_final

   !> Non-copyable guard (PLAN_FORTRAN.md §5.4): a whole-type assignment would
   !> alias the owned bytes and double-free. Abort loudly at the copy site.
   !> Branching on the source state gives a precise diagnostic (and uses `rhs`).
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
   !  tensogram_message methods (decoded handle owned by Rust)
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

   !> Non-copyable guard (PLAN_FORTRAN.md §5.4). Branching on the source state
   !> gives a precise diagnostic (and uses `rhs`).
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
   !  Public encode / decode / inspect
   ! =========================================================================

   !> Encode a single rank-2 float32 tensor as a one-object Tensogram message.
   !> `a` may be a non-contiguous slice: the `contiguous` attribute makes the
   !> compiler gather it into a temporary, so c_loc(a) is always valid.
   subroutine tensogram_encode_r2_f32(a, buf, err, metadata_json, hash)
      real(c_float), target, contiguous, intent(in)  :: a(:,:)
      type(tensogram_buffer),            intent(out)  :: buf
      integer(c_int),                    intent(out)  :: err
      character(len=*), intent(in), optional :: metadata_json  ! extra top-level JSON keys
      character(len=*), intent(in), optional :: hash           ! "xxh3" (default); "" => none

      character(kind=c_char), allocatable, target :: meta_c(:), hash_c(:)
      character(len=:),       allocatable         :: meta_s, hash_s
      type(c_ptr),       target :: ptrs(1)
      integer(c_size_t), target :: lens(1)
      type(c_ptr)               :: hash_ptr

      meta_s = descriptor_json_r2_f32(size(a, 1), size(a, 2), metadata_json)
      call f_to_cstr(meta_s, meta_c)

      if (present(hash)) then
         hash_s = hash
      else
         hash_s = 'xxh3'
      end if
      if (len(hash_s) == 0) then
         hash_ptr = c_null_ptr                     ! NULL hash_algo => no hash
      else
         call f_to_cstr(hash_s, hash_c)
         hash_ptr = c_loc(hash_c)
      end if

      ptrs(1) = c_loc(a)                            ! contiguous + target => valid
      lens(1) = size(a, kind=c_size_t) * (storage_size(a, kind=c_size_t) / 8_c_size_t)

      err = c_tgm_encode(c_loc(meta_c), c_loc(ptrs), c_loc(lens), &
                         1_c_size_t, hash_ptr, 0_c_int32_t, buf%raw)
   end subroutine tensogram_encode_r2_f32

   !> Decode a wire-format message into a handle. `verify_hash` defaults to
   !> .false. to match the library default (PLAN_FORTRAN.md §10); set it true
   !> for end-to-end integrity. `native_byte_order` defaults true (host order).
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

   !> Number of objects in the decoded message.
   function tensogram_num_objects(msg) result(n)
      type(tensogram_message), intent(in) :: msg
      integer :: n
      n = int(c_tgm_message_num_objects(msg%ptr))
   end function tensogram_num_objects

   !> Rank of object `iobj` (1-based).
   function tensogram_object_ndim(msg, iobj) result(nd)
      type(tensogram_message), intent(in) :: msg
      integer,                 intent(in) :: iobj
      integer :: nd
      nd = int(c_tgm_object_ndim(msg%ptr, int(iobj - 1, c_size_t)))
   end function tensogram_object_ndim

   !> Object extents in FORTRAN (column-major) order — the on-wire descriptor
   !> shape REVERSED. ext(1) is the fastest-varying axis.
   function tensogram_object_shape(msg, iobj) result(ext)
      type(tensogram_message), intent(in) :: msg
      integer,                 intent(in) :: iobj
      integer(c_int64_t), allocatable :: ext(:)
      integer(c_int64_t), pointer     :: cshape(:)
      integer     :: nd, k
      type(c_ptr) :: p
      nd = tensogram_object_ndim(msg, iobj)
      p  = c_tgm_object_shape(msg%ptr, int(iobj - 1, c_size_t))
      call c_f_pointer(p, cshape, [nd])
      allocate(ext(nd))
      do k = 1, nd
         ext(k) = cshape(nd - k + 1)              ! reverse -> Fortran order
      end do
   end function tensogram_object_shape

   !> dtype string of object `iobj` (e.g. "float32").
   function tensogram_object_dtype(msg, iobj) result(dt)
      type(tensogram_message), intent(in) :: msg
      integer,                 intent(in) :: iobj
      character(len=:), allocatable :: dt
      dt = cptr_to_fstr(c_tgm_object_dtype(msg%ptr, int(iobj - 1, c_size_t)))
   end function tensogram_object_dtype

   !> Copy decoded object `iobj` (must be rank-2 float32) into a Fortran array
   !> shaped (ni, nj), matching what tensogram_encode_r2_f32 wrote. The copy is
   !> deliberate: the raw pointer aliases message-owned memory that dangles once
   !> `msg` is freed.
   subroutine tensogram_object_to_r2_f32(msg, iobj, out, err)
      type(tensogram_message),    intent(in)  :: msg
      integer,                    intent(in)  :: iobj
      real(c_float), allocatable, intent(out) :: out(:,:)
      integer(c_int),             intent(out) :: err
      integer(c_int64_t), allocatable :: ext(:)
      character(len=:),   allocatable :: dt
      real(c_float),      pointer     :: view(:,:)
      integer(c_size_t) :: nbytes
      type(c_ptr)       :: dptr

      err = TGM_ERROR_OK
      dt = tensogram_object_dtype(msg, iobj)
      if (dt /= 'float32') then
         err = TGM_ERROR_OBJECT
         return
      end if
      ext = tensogram_object_shape(msg, iobj)      ! Fortran (reversed) order
      if (size(ext) /= 2) then
         err = TGM_ERROR_OBJECT
         return
      end if

      dptr = c_tgm_object_data(msg%ptr, int(iobj - 1, c_size_t), nbytes)
      if (.not. c_associated(dptr)) then
         err = TGM_ERROR_OBJECT
         return
      end if

      ! C-contiguous bytes of shape [nj,ni] == Fortran-contiguous [ni,nj].
      call c_f_pointer(dptr, view, [int(ext(1)), int(ext(2))])
      allocate(out(int(ext(1)), int(ext(2))))
      out = view                                   ! decouple from msg lifetime
   end subroutine tensogram_object_to_r2_f32

end module tensogram

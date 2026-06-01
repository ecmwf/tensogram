# Fortran API

Tensogram ships a shallow **Fortran 2018** binding (`fortran/`) over
the [C ABI](c-api.md). It is a thin `iso_c_binding` layer that links the
same `libtensogram` the C and C++ wrappers use — no new C or Rust code —
following the `eccodes_f90` model ECMWF Fortran codes already expect.

> **Status.** Emerging. The synchronous surface is complete: generic
> encode/decode, the multi-message file API, application metadata, the
> encoding pipeline, and the streaming encoder. Only the async surface is
> deferred; see
> [`PLAN_FORTRAN.md`](https://github.com/ecmwf/tensogram/blob/main/PLAN_FORTRAN.md)
> for the staged roadmap.

## The memory-order contract (read this first)

A Fortran array is **column-major**; Tensogram (like NumPy / C) is
**row-major**. The binding bridges this by writing a Fortran `a(ni, nj)`
with the on-wire descriptor **shape and strides reversed** to
`[nj, ni]` / `[ni, 1]`. The consequences are deliberate and important:

- **Fortran → Tensogram → Fortran round-trips are bit-identical.** What
  you encode as `a(ni, nj)` decodes back as `(ni, nj)`.
- **A NumPy / C / xarray / Zarr reader sees the transpose** — shape
  `(nj, ni)`, with `arr[j-1, i-1] == a(i, j)`.

If you exchange `.tgm` data between Fortran and other languages, account
for that transpose (or transpose on one side). This is the single most
common source of cross-language confusion.

## Install and link

The binding links the system `libtensogram` discovered through
**pkg-config** (`tensogram.pc`), shipped by `cargo cinstall -p
tensogram-ffi` and by the [release tarballs](c-api.md). CMake is the
build system of record; an `fpm` manifest is also provided.

### CMake (recommended)

```bash
cmake -S fortran -B build/fortran
cmake --build build/fortran
ctest --test-dir build/fortran --output-on-failure
```

`fortran/CMakeLists.txt` discovers the FFI with
`pkg_check_modules(... IMPORTED_TARGET tensogram)` and exposes a
`tensogram_f` static library plus the `tensogram` Fortran module. For a
non-default FFI prefix, point pkg-config at it first:

```bash
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
```

In your own project:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(TENSOGRAM REQUIRED IMPORTED_TARGET tensogram)
add_executable(my_app my_app.f90)
target_link_libraries(my_app PRIVATE tensogram_f)   # + PkgConfig::TENSOGRAM
```

### fpm

fpm has no native pkg-config integration, so inject the flags:

```bash
cd fortran
fpm build \
    --flag      "$(pkg-config --cflags tensogram)" \
    --link-flag "$(pkg-config --libs   tensogram)"
```

### Plain gfortran

```bash
gfortran -std=f2018 fortran/src/tensogram.f90 my_app.f90 \
    $(pkg-config --cflags --libs tensogram) -o my_app
```

## Quick start

```fortran
program demo
   use, intrinsic :: iso_c_binding, only : c_float, c_int8_t, c_int
   use tensogram
   implicit none

   real(c_float)                  :: field(100, 200)
   real(c_float),    allocatable  :: out(:,:)
   integer(c_int8_t), allocatable :: wire(:)
   type(tensogram_buffer)  :: buf
   type(tensogram_message) :: msg
   integer(c_int) :: err

   call random_number(field)

   ! Encode -> Rust-owned bytes (lossless: encoding/compression "none").
   ! `tensogram_encode` is generic over dtype and rank.
   call tensogram_encode(field, buf, err)
   call tensogram_check(err, 'encode')

   ! Copy the wire bytes out, then decode them back.
   call buf%as_array(wire)
   call tensogram_decode(wire, msg, err)
   call tensogram_check(err, 'decode')

   ! Extract object 1 -> a Fortran array shaped like the input.
   call tensogram_to_array(msg, 1, out, err)
   call tensogram_check(err, 'decode object')

   print *, 'bit-identical: ', .not. any(transfer(out, [0]) /= transfer(field, [0]))
end program demo
```

See [`examples/fortran/encode_decode.f90`](https://github.com/ecmwf/tensogram/blob/main/examples/fortran/encode_decode.f90)
for the fully annotated version.

## Ownership and non-copyable handles

`tensogram_buffer` (encoded bytes) and `tensogram_message` (a decoded
handle) own resources allocated by Rust. They are released automatically
by a `final` procedure at scope exit, or eagerly via `call buf%free()` /
`call msg%free()` (idempotent).

These handle types are **non-copyable**. Fortran cannot delete the
intrinsic assignment, so a `b = a` would alias the underlying handle and
free it twice. To make that mistake impossible to do silently, the types
define an `assignment(=)` that **aborts with `error stop`**:

```fortran
type(tensogram_message) :: a, b
b = a        ! ERROR STOP: tensogram_message is non-copyable ...
```

Pass handles **by reference** (the default), and let factory procedures
return them through `intent(out)` arguments (as `tensogram_decode` does).

## Error handling

Every fallible call returns a `tgm_error` code via an `intent(out)`
argument; `TGM_ERROR_OK` (= 0) is success. The named constants
(`TGM_ERROR_FRAMING`, `TGM_ERROR_OBJECT`, …) mirror the C `tgm_error`
enum.

```fortran
integer(c_int) :: err
call tensogram_decode(wire, msg, err)
if (err /= TGM_ERROR_OK) then
   write(*,*) 'decode failed: ', tensogram_last_error()
   ! ... handle / return ...
end if
```

- `tensogram_last_error()` returns the thread-local message from the most
  recent failing call.
- `tensogram_strerror(err)` returns a static description of a code.
- `tensogram_check(err, context)` is a convenience guard: it is a no-op
  on success and prints `context` + the detail then `error stop`s on
  failure. Use it in examples and scripts; in library code prefer
  inspecting `err` yourself.

The codes the binding surfaces (see [Error Handling](error-handling.md)
for the full taxonomy):

| Code | Raised by, for example |
|------|------------------------|
| `TGM_ERROR_FRAMING` | decoding a buffer that is not a valid message |
| `TGM_ERROR_ENCODING` | encoding **non-finite** values (NaN / ±Inf are rejected by default) |
| `TGM_ERROR_OBJECT` | `tensogram_to_array` with a wrong dtype, wrong rank, or out-of-range object index |
| `TGM_ERROR_IO` | opening / reading a file that does not exist or cannot be read |
| `TGM_ERROR_HASH_MISMATCH` / `TGM_ERROR_MISSING_HASH` | `verify_hash = .true.` and the recorded digest disagrees / is absent |

The binding never aborts on these (only the deliberate non-copyable
handle guard and `tensogram_check` use `error stop`). The metadata
getters do not raise: a missing key returns the supplied default (or an
empty string).

## Decoding options

`tensogram_decode` accepts two optional logicals:

- `verify_hash` (default `.false.`) — when `.true.`, verifies the
  per-object xxh3 digest. Off by default to match the core library
  (most transports already provide integrity); enable it for end-to-end
  checks.
- `native_byte_order` (default `.true.`) — convert decoded bytes to the
  host byte order. Leave it true unless you want raw wire-order bytes.

## What is available now

| Procedure | Purpose |
|---|---|
| `tensogram_encode(a, buf, err [, metadata_json, hash, encoding, filter, compression])` | Encode an array into a one-object message — **generic** over dtype and rank |
| `tensogram_decode(wire, msg, err [, verify_hash, native_byte_order])` | Decode wire bytes into a message handle |
| `tensogram_num_objects(msg)` | Number of decoded objects |
| `tensogram_object_ndim(msg, iobj)` | Rank of object `iobj` (1-based) |
| `tensogram_object_shape(msg, iobj)` | Extents in Fortran (column-major) order |
| `tensogram_object_dtype(msg, iobj)` | dtype string (e.g. `"float32"`) |
| `tensogram_to_array(msg, iobj, out, err)` | Copy an object into a Fortran array — **generic** over dtype and rank of `out` |
| `tensogram_file_open/create(path, file, err)` | Open / create a `.tgm` file |
| `tensogram_file_message_count(file, n, err)` | Number of messages in the file |
| `tensogram_file_append(file, a, err [, metadata_json, hash, encoding, filter, compression])` | Encode `a` and append it as a message — **generic** over dtype and rank |
| `tensogram_file_decode_message(file, index, msg, err [, verify_hash, native_byte_order])` | Decode message `index` (1-based) |
| `tensogram_file_read_message(file, index, buf, err)` | Read raw message bytes at `index` |
| `tensogram_meta` + `%add_string` / `%add_int` / `%add_real` / `%base_json` | Build per-object application metadata (JSON-escaped) |
| `tensogram_message_metadata(msg, meta, err)` | Extract a metadata handle from a decoded message |
| `tensogram_metadata_get_string/int/float(meta, key [, default])` | Look up metadata by dot-notation key |
| `tensogram_streaming_encoder_create(path, enc, err [, metadata_json, hash])` | Open a streaming encoder (writes a message progressively to a file) |
| `tensogram_streaming_encoder_write(enc, a, err [, encoding, filter, compression])` | Append one object to the stream — **generic** over dtype and rank |
| `tensogram_streaming_encoder_count(enc)` / `_finish(enc, err)` / `enc%free()` | Objects written so far / finalise (footer + close) / release |
| `buf%as_array(out)` / `buf%size()` / `buf%free()` | Buffer access and release |
| `file%close()` | Close a file handle (also automatic at scope exit) |
| `tensogram_check` / `tensogram_last_error` / `tensogram_strerror` | Error helpers |

`tensogram_encode`, `tensogram_to_array`, `tensogram_file_append`, and
`tensogram_streaming_encoder_write` are generic interfaces over **dtype**
(`real32`, `real64`, `int32`, `int64`) and **rank** (`0`–`7`). `a` / `out` are assumed-rank; the dtype is
resolved from the array's type/kind, the rank from the array itself. A
dtype mismatch on decode returns `TGM_ERROR_OBJECT`.
(`int8`/`int16`/`complex`/`float16` are follow-ups — Fortran has no
native unsigned or half/complex-as-pair mapping.)

## Working with files

`tensogram_file` is an RAII handle for a multi-message `.tgm` file (the
common "append forecast steps" pattern). Like the buffer/message handles
it is non-copyable and closes automatically at scope exit (or eagerly via
`call f%close()`). Message indices are **1-based**.

```fortran
type(tensogram_file)    :: f
type(tensogram_message) :: msg
real(c_float)              :: field(ni, nj)
real(c_float), allocatable :: out(:,:)
integer(c_int) :: err, n, step

call tensogram_file_create('forecast.tgm', f, err)
do step = 1, nsteps
   ! ... fill field ...
   call tensogram_file_append(f, field, err)      ! one message per step
end do
call f%close()

call tensogram_file_open('forecast.tgm', f, err)
call tensogram_file_message_count(f, n, err)       ! n == nsteps
do step = 1, n
   call tensogram_file_decode_message(f, step, msg, err)   ! random access
   call tensogram_to_array(msg, 1, out, err)
end do
call f%close()
```

See [`examples/fortran/file_api.f90`](https://github.com/ecmwf/tensogram/blob/main/examples/fortran/file_api.f90).

## Application metadata and compression

The encoding pipeline is configurable per call: pass `encoding`, `filter`,
and/or `compression` (default `"none"`). Use a **lossless** codec
(`"zstd"`, `"lz4"`, `"szip"`, `"blosc2"`) to keep round-trips bit-exact.

Per-object application metadata (names, units, namespaced vocabularies)
is built with the zero-dependency `tensogram_meta` builder, which
JSON-escapes keys and values for you, and read back with the dot-notation
getters (which search the `base[i]` entries, then `_extra_`).

```fortran
type(tensogram_meta)     :: m
type(tensogram_buffer)   :: buf
type(tensogram_message)  :: msg
type(tensogram_metadata) :: meta
integer(c_int) :: err

call m%add_string('name', 'temperature')
call m%add_string('units', 'K')
call m%add_int('level', 850_c_int64_t)

call tensogram_encode(field, buf, err, &
                      metadata_json = m%base_json(), compression = 'zstd')

! ... decode into msg ...
call tensogram_message_metadata(msg, meta, err)
print *, tensogram_metadata_get_string(meta, 'name')          ! temperature
print *, tensogram_metadata_get_int(meta, 'level', -1_c_int64_t)  ! 850
```

The library remains vocabulary-agnostic: `tensogram_meta` just builds the
`base` JSON; the meaning of the keys is the application's concern.

## Streaming encoder

For producers that emit objects incrementally, `tensogram_streaming_encoder`
writes a single multi-object message to a file **progressively** — the
preamble and header metadata on create, one data-object frame per
`_write`, and the footer + postamble on `_finish` — without buffering the
whole message. The handle is non-copyable; `enc%free()` (and the
finalizer) only release the handle, so **call `_finish` for a valid
file** before freeing.

```fortran
type(tensogram_streaming_encoder) :: enc
integer(c_int) :: err, obj

call tensogram_streaming_encoder_create('out.tgm', enc, err)
do obj = 1, nobj
   ! ... fill field ...
   call tensogram_streaming_encoder_write(enc, field, err, compression='zstd')
end do
call tensogram_streaming_encoder_finish(enc, err)   ! footer + close
call enc%free()
```

The result is one message with `nobj` objects; reopen it with
`tensogram_file_open` + `tensogram_file_decode_message` and pull each
object out with `tensogram_to_array`. See
[`examples/fortran/streaming.f90`](https://github.com/ecmwf/tensogram/blob/main/examples/fortran/streaming.f90).

Planned next (see `PLAN_FORTRAN.md`): the async surface (deferred).

## Edge cases

The binding fails gracefully rather than crashing:

- **Zero-size tensors** (e.g. `real :: a(0)`) encode and round-trip to a
  zero-size array.
- **Out-of-range object indices** (`< 1` or `> num_objects`) return
  `TGM_ERROR_OBJECT` from `tensogram_to_array`; the raw accessors
  (`tensogram_object_dtype` / `_ndim` / `_shape`) return an empty / zero
  result for an out-of-range index rather than aborting.
- **Empty or truncated wire buffers** decode to a non-`OK` error with
  `tensogram_last_error()` populated, never a crash.
- **Zero-object messages** (e.g. a streamed message finished with no
  writes) are valid and decode to `num_objects == 0`.
- **Unicode (raw UTF-8) and long strings** in metadata survive the
  builder / JSON escaper / decode / getter round-trip byte-for-byte.

## Build and test

```bash
make fortran-build      # CMake configure + build
make fortran-test       # ctest
```

The test suite covers bit-identical round-trips across dtypes and ranks,
the file API, application metadata, lossless-compression round-trips, the
streaming encoder, the error path, and the non-copyable guard (a negative
test). Cross-language
parity tests assert the column-major contract in **both directions**
against a C/C++ reader/writer, and **both directions** against Python /
NumPy (when a Python with the `tensogram` package is present).

## See also

- [C API](c-api.md) — the ABI the binding sits on.
- [Objects and Dtypes](../concepts/objects.md) — strides and layout.
- [`PLAN_FORTRAN.md`](https://github.com/ecmwf/tensogram/blob/main/PLAN_FORTRAN.md)
  — design rationale and the full roadmap.

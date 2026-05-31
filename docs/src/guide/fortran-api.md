# Fortran API

Tensogram ships a shallow **Fortran 2008/2018** binding (`fortran/`) over
the [C ABI](c-api.md). It is a thin `iso_c_binding` layer that links the
same `libtensogram` the C and C++ wrappers use — no new C or Rust code —
following the `eccodes_f90` model ECMWF Fortran codes already expect.

> **Status.** Emerging. The first synchronous slice — encode a native
> `real(:,:)` field, decode it back, and inspect object metadata — is in
> place. The file API, more dtypes/ranks, streaming, and async are
> planned; see [`PLAN_FORTRAN.md`](https://github.com/ecmwf/tensogram/blob/main/PLAN_FORTRAN.md)
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
| `tensogram_encode(a, buf, err [, metadata_json, hash])` | Encode an array into a one-object message — **generic** over dtype and rank |
| `tensogram_decode(wire, msg, err [, verify_hash, native_byte_order])` | Decode wire bytes into a message handle |
| `tensogram_num_objects(msg)` | Number of decoded objects |
| `tensogram_object_ndim(msg, iobj)` | Rank of object `iobj` (1-based) |
| `tensogram_object_shape(msg, iobj)` | Extents in Fortran (column-major) order |
| `tensogram_object_dtype(msg, iobj)` | dtype string (e.g. `"float32"`) |
| `tensogram_to_array(msg, iobj, out, err)` | Copy an object into a Fortran array — **generic** over dtype and rank of `out` |
| `buf%as_array(out)` / `buf%size()` / `buf%free()` | Buffer access and release |
| `tensogram_check` / `tensogram_last_error` / `tensogram_strerror` | Error helpers |

`tensogram_encode` and `tensogram_to_array` are generic interfaces over
**dtype** (`real32`, `real64`, `int32`, `int64`) and **rank** (`0`–`7`).
`a` / `out` are assumed-rank; the dtype is resolved from the array's
type/kind, the rank from the array itself. A dtype mismatch on decode
returns `TGM_ERROR_OBJECT`. (`int8`/`int16`/`complex`/`float16` are
follow-ups — Fortran has no native unsigned or half/complex-as-pair
mapping.)

Planned next (see `PLAN_FORTRAN.md`): the file API (`.tgm` multi-message
append / random-access decode), structured metadata builders, then the
streaming encoder.

## Build and test

```bash
make fortran-build      # CMake configure + build
make fortran-test       # ctest (incl. the Fortran<->Python parity test
                        #         when a Python with `tensogram` is present)
```

The test suite covers a bit-identical round-trip, the error path, the
non-copyable guard (a negative test), and cross-language parity that
asserts the column-major contract against the Python reader.

## See also

- [C API](c-api.md) — the ABI the binding sits on.
- [Objects and Dtypes](../concepts/objects.md) — strides and layout.
- [`PLAN_FORTRAN.md`](https://github.com/ecmwf/tensogram/blob/main/PLAN_FORTRAN.md)
  — design rationale and the full roadmap.

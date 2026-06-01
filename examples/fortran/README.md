# Fortran examples

Runnable examples for the Tensogram Fortran binding (`fortran/`). They link
the system `libtensogram` discovered via pkg-config (`tensogram.pc`, shipped
by `cargo cinstall -p tensogram-ffi` or the release tarballs).

| Example | Shows |
|---------|-------|
| `encode_decode.f90` | Encode a `real(:,:)` field, decode it back, and assert the round-trip is bit-identical |
| `file_api.f90` | Append several fields to a multi-message `.tgm`, then reopen, count, and decode each by index |
| `streaming.f90` | Stream a multi-object message progressively (one object at a time), then reopen and decode every object |

## Build & run

The simplest path is through the binding's CMake project, which builds the
examples alongside the library and tests:

```bash
cmake -S fortran -B build/fortran
cmake --build build/fortran
./build/fortran/fortran_encode_decode
```

Or compile a single example directly against an installed `tensogram.pc`:

```bash
gfortran -std=f2018 \
    fortran/src/tensogram.F90 examples/fortran/encode_decode.f90 \
    $(pkg-config --cflags --libs tensogram) -o encode_decode
./encode_decode
```

The binding module is `tensogram.F90` (capital F — preprocessed
automatically); keep the `fortran/src/tgm_*.inc` templates beside it.
`-std=f2018` is used here because example/application code that declares
handle locals trips a gfortran `f08/0011` advisory under `-std=f2008`
(valid F2008; see the Fortran API guide). The binding *library* itself is
F2008-clean.

## Note on array order

A Fortran `a(ni, nj)` round-trips Fortran↔Fortran bit-identically, but a
NumPy / C reader sees the **transpose** `(nj, ni)`. See the
[Fortran API guide](../../docs/src/guide/fortran-api.md) for the full
column-major contract.

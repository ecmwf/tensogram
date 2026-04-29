# tensogram-ffi

C FFI bindings for Tensogram. Produces a C-callable library
(`libtensogram.{so,dylib,a}`) and a `tensogram.h` header generated
by [`cbindgen`](https://github.com/mozilla/cbindgen).

All public functions are prefixed `tgm_`. Opaque handles are returned
by `tgm_*` constructors and released by the matching `tgm_*_free`
function.

## Quickstart (C)

```c
#include <tensogram/tensogram.h>

const char* meta_json =
    "{\"descriptors\": [{"
    "\"type\": \"ntensor\", \"ndim\": 2, \"shape\": [100, 200], "
    "\"strides\": [200, 1], \"dtype\": \"float32\", "
    "\"byte_order\": \"little\", \"encoding\": \"none\", "
    "\"filter\": \"none\", \"compression\": \"none\"}]}";

const uint8_t* ptrs[1] = { payload_bytes };
const size_t lens[1] = { payload_len };

tgm_bytes_t out;
tgm_error err = tgm_encode(meta_json, ptrs, lens, 1, "xxh3", 0, &out);
if (err != TGM_ERROR_OK) {
    fprintf(stderr, "encode failed: %s\n", tgm_last_error());
    return 1;
}

// Use out.data / out.len ...
tgm_bytes_free(out);
```

A thread-local error string is available via `tgm_last_error()` after
any `tgm_*` call returns non-`TGM_ERROR_OK`.

## Install

Three install paths. Pick whichever matches your context.

### Pre-built binary (no Rust toolchain)

Download the tarball for your platform from the
[GitHub Releases page](https://github.com/ecmwf/tensogram/releases),
then extract under `/usr/local`:

```bash
VERSION=<release-version>      # e.g. 0.20.0
PLATFORM=linux-x86_64          # or linux-aarch64 / macos-x86_64 / macos-aarch64
ASSET="tensogram-ffi-${VERSION}-${PLATFORM}.tar.gz"

curl -LO "https://github.com/ecmwf/tensogram/releases/download/${VERSION}/${ASSET}"
sudo tar --no-same-owner -C /usr/local -xzf "${ASSET}"
sudo ldconfig                  # Linux only
pkg-config --modversion tensogram
```

Available platforms: `linux-x86_64`, `linux-aarch64`, `macos-x86_64`,
`macos-aarch64`. The bundled `tensogram.pc` hard-codes
`prefix=/usr/local`, so the default extract path matters; see the
[C API guide](https://sites.ecmwf.int/docs/tensogram/main/guide/c-api.html)
for non-default prefixes.

### `cargo cinstall` (custom prefix)

[`cargo-c`](https://github.com/lu-zero/cargo-c) builds and installs
the versioned shared library, static library, pkg-config descriptor,
and header in one step:

```bash
cargo install cargo-c
# --libdir=lib pins the layout; on Debian-style multiarch systems
# cargo-c would otherwise pick lib/<triplet>, hiding the .pc file from
# the PKG_CONFIG_PATH below.
cargo cinstall --release -p tensogram-ffi \
    --prefix="$HOME/.local" --libdir=lib
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
pkg-config --modversion tensogram
```

### Build from source (contributor flow)

```bash
cargo build --release -p tensogram-ffi
```

Outputs:

- `target/release/libtensogram_ffi.{a,so,dylib}` — the library (the
  `_ffi` suffix is the cargo-build artefact name; the in-tree
  `cpp/CMakeLists.txt` consumes this path)
- `rust/tensogram-ffi/tensogram.h` — the cbindgen-generated header,
  regenerated on every build by `build.rs`

Plain `cargo build` does **not** produce a SONAME, a pkg-config file,
or an installed header. Use `cargo cinstall` for those.

## Linking

After installing via either pre-built tarball or `cargo cinstall`:

```bash
cc $(pkg-config --cflags tensogram) my_program.c \
   $(pkg-config --libs tensogram) \
   -o my_program
```

`pkg-config --cflags` returns `-I<includedir>` so `#include
<tensogram/tensogram.h>` resolves; `pkg-config --libs` returns
`-L<libdir> -ltensogram`.

## crates.io

```bash
cargo add tensogram-ffi
```

Most C / C++ consumers do not need this — depend on the pre-built
binary or `cargo cinstall` instead. The crate is published primarily
so other Rust crates can re-export the C ABI.

## Versioning

Pre-1.0 SONAME policy: the library SONAME is `MAJOR.MINOR`
(`libtensogram.so.0.20` for 0.20.x). Every minor release bumps the
SONAME, so consumers must rebuild on each minor. Patch releases keep
the SONAME stable.

The wire-format version (independent of SONAME) is exposed via the
`TGM_WIRE_VERSION` constant in `tensogram.h`.

See the [C API guide](https://sites.ecmwf.int/docs/tensogram/main/guide/c-api.html)
for the full policy.

## Documentation

- [C API guide](https://sites.ecmwf.int/docs/tensogram/main/guide/c-api.html)
- [C++ API guide](https://sites.ecmwf.int/docs/tensogram/main/guide/cpp-api.html)
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.

# tensogram-ffi

C FFI bindings for Tensogram.

Building this crate produces `libtensogram_ffi.{so,dylib,a}` and a
generated `tensogram.h` header (via `cbindgen`) for use from C, C++,
and any language with a C foreign-function interface.

All public functions are prefixed `tgm_`. Opaque handles are
returned by `tgm_*` constructors and released by the matching
`tgm_*_free` function.

## Usage (C)

```c
#include "tensogram.h"

const char* meta_json =
    "{\"version\": 3, \"descriptors\": [{"
    "\"type\": \"ntensor\", \"ndim\": 2, \"shape\": [100, 200], "
    "\"strides\": [200, 1], \"dtype\": \"float32\", "
    "\"byte_order\": \"little\", \"encoding\": \"none\", "
    "\"filter\": \"none\", \"compression\": \"none\"}]}";

const uint8_t* ptrs[1] = { payload_bytes };
const size_t lens[1] = { payload_len };

TgmBytes out;
TgmError err = tgm_encode(meta_json, ptrs, lens, 1, "xxh3", 0, &out);
if (err != TGM_OK) {
    fprintf(stderr, "encode failed: %s\n", tgm_last_error());
    return 1;
}

// Use out.data / out.len ...
tgm_bytes_free(out);
```

A thread-local error string is available via `tgm_last_error()`
after any `tgm_*` call returns non-`TGM_OK`.

## Building

```bash
cargo build --release -p tensogram-ffi
```

Output:

- `target/release/libtensogram_ffi.{a,so,dylib}` — the library
- `cpp/include/tensogram.h` — the generated C header

## Installation

```bash
cargo add tensogram-ffi
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.

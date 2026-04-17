# tensogram-ffi

C FFI bindings for Tensogram.

Produces `libtensogram.so/.a` and `tensogram.h` header for embedding in C/C++ applications. Provides encode, decode, file I/O, and partial decode.

## Usage

```c
#include "tensogram.h"

TGMessage* msg = tg_encode(&meta, objects, count, &options);
TGDecodeResult* result = tg_decode(msg, &options);
```

## Building

```bash
cargo build --release -p tensogram-ffi
```

Generates `target/release/libtensogram.so`, `libtensogram.a`, and `tensogram.h`.

## Installation

```toml
[dependencies]
tensogram-ffi = "0.1"
```

## Documentation

- Full docs: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.

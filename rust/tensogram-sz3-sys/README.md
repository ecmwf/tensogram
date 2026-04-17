# tensogram-sz3-sys

Clean-room FFI bindings for the SZ3 lossy compression library.

This crate provides low-level unsafe Rust bindings to the SZ3 C++ library. It vendors the SZ3 source code and builds it automatically. This is a sys-level crate intended for use by the higher-level `tensogram-sz3` crate.

## Usage

```rust
use tensogram_sz3_sys::SZ3_Init;

let compressor = unsafe { SZ3_Init(config_ptr) };
```

## Installation

```toml
[dependencies]
tensogram-sz3-sys = "0.1"
```

## Documentation

- Full documentation: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0 for the wrapper code. The vendored SZ3 source is licensed under Argonne BSD and Boost Software License 1.0. See LICENSES.md for details.

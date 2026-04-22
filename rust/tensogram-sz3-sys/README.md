# tensogram-sz3-sys

Clean-room FFI bindings to the [SZ3](https://github.com/szcompressor/SZ3)
lossy-compression C++ library, via a thin C shim
(`cpp/sz3_ffi.cpp`).

This is a `-sys` crate — it exposes raw `extern "C"` bindings and
vendors the SZ3 header-only source. Downstream code should normally
depend on the safe higher-level
[`tensogram-sz3`](https://crates.io/crates/tensogram-sz3) crate
instead.

## Installation

```bash
cargo add tensogram-sz3-sys
```

A C++ compiler that supports C++17 is required at build time. The
crate's `build.rs` compiles the vendored SZ3 source alongside the
shim.

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

This crate contains code under multiple licences:

- Our wrapper code (Rust + `cpp/sz3_ffi.cpp`) — Apache-2.0.
- Vendored SZ3 C++ library — UChicago Argonne BSD-style licence.
- Vendored ska flat hash map (inside the SZ3 tree) — Boost
  Software License 1.0.

See [LICENSES.md](LICENSES.md) for the full text of each licence.

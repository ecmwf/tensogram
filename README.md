<p align="center">
  <img src="logo.png" alt="Tensogram" width="200">
</p>

<p align="center">
  <em>A fast, efficient 'telegram' for multidimensional tensors</em>
</p>

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg" alt="Emerging">
  </a>
</p>

> [!IMPORTANT]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

A library to encode and decode binary N-Tensor scientific data with semantic metadata close to the data, in a serialisable format that can be sent over the network, encoded into in-memory buffers and decoded with zero-copy.

Tensogram defines a network-transmissible binary message format, not a file format. Multiple messages can be appended to a file, each carrying its own begin/terminator codes.

## Features

- **Self-describing messages** — CBOR-encoded metadata vocabulary agnostic
- **N-Tensor support** — multiple tensors of different dtypes per message (float16 through float64, int8 through int64, complex, bfloat16)
- **No panics** — robust library where all fallible operations return `Result<T, TensogramError>`
- **File API** — `TensogramFile` for multi-message `.tgm` files: append, random-access read, iterate, and decode individual messages or objects
- **Partial decode** — `decode_range` extracts sub-tensor slices without decoding the full object, with random-access support for szip, blosc2, and zfp
- **Remote access** — read `.tgm` files directly from S3, GCS, Azure Blob, or HTTP via `object_store` integration (`open_remote`, `open_source`)
- **Async API** — full async counterparts for file open, message read, decode, and iteration via tokio (`open_async`, `decode_message_async`, etc.)
- **Streaming encoder** — progressive encode/transmit without buffering the full message; preceder metadata frames enable consumer-side streaming decode
- **Compression** — szip, zstd, lz4, blosc2, zfp, sz3 per data object; pure-Rust backends available (`szip-pure`, `zstd-pure`) for environments without C libraries
- **Hash verification** — xxHash xxh3-64 integrity check per object
- **Validation** — 4-level structural and data integrity validation with optional JSON output (`tensogram validate --quick|--checksum|--full`)
- **Multiple languages** — Rust, Python (NumPy), C/C++, WebAssembly
- **Free-threaded Python** — GIL-free operation on Python 3.13t with full parallel encode/decode
- **xarray backend** — `xr.open_dataset("file.tgm", engine="tensogram")` with lazy loading, coordinate auto-detection, and hypercube stacking via `open_datasets()`
- **Dask integration** — parallel chunked computation via `xr.open_dataset(..., chunks={})` with per-chunk `decode_range` for efficient out-of-core processing
- **Zarr v3 store** — `zarr.open_group(store=TensogramStore.open_tgm("file.tgm"), mode="r")` for standard Zarr API access with 14 bidirectionally-mapped dtypes
- **GRIB conversion** — import GRIB data with MARS metadata preservation and configurable namespace extraction
- **NetCDF conversion** — import NetCDF-3 and NetCDF-4 files with CF metadata lifting (`--cf`), packed data unpacking, and configurable encoding/compression pipeline shared with `convert-grib`
- **CLI** — `tensogram info/ls/dump/get/set/copy/merge/split/reshuffle/convert-grib/convert-netcdf` with `--strategy first|last|error` merge conflict resolution
- **Optional features** — `mmap` (zero-copy file reads), `async` (tokio I/O), `remote` (S3/GCS/Azure/HTTP)

## Quick Start

### Rust
```rust
let desc = DataObjectDescriptor {
    obj_type: "ntensor".into(), ndim: 2,
    shape: vec![100, 200], strides: vec![200, 1],
    dtype: Dtype::Float32, byte_order: ByteOrder::Big,
    encoding: "simple_packing".into(), filter: "none".into(),
    compression: "szip".into(), params, hash: None,
};

let message = encode(&meta, &[(&desc, &raw)], &EncodeOptions::default())?;
let (_, objects) = decode(&message, &DecodeOptions::default())?;
```

See `examples/rust/` for MARS metadata, streaming, compression, file API, and more.

### Python

```python
data = np.random.randn(100, 200).astype(np.float32)
msg = tensogram.encode(
    {"version": 2, "base": [{"mars": {"param": "2t"}}]},
    [({"type": "ntensor", "shape": [100, 200], "dtype": "float32",
       "encoding": "simple_packing", "compression": "szip"}, data)],
)
result = tensogram.decode(msg)
arr = result.objects[0][1]  # numpy array
```

### xarray

```python
ds = xr.open_dataset("forecast.tgm", engine="tensogram")  # lazy-loaded
```

### Zarr v3

```python
# open all .tgm files in a directory
group = zarr.open_group(store=TensogramStore.open_dir("somedir/"), mode="r")  # loads somedir/*.tgm
```

See `examples/python/` for encode/decode, metadata, packing, file API, iterators, xarray, zarr, and streaming consumer patterns.

## Build & Test

```bash
cargo build --workspace                                          # build
cargo test --workspace                                           # test
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lint
```

**Optional features:**
```bash
cargo build -p tensogram-core --features mmap,async,remote
```

**C++ wrapper** (`cpp/include/tensogram.hpp`):
```bash
cargo build --release                  # build Rust static library first
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure  # run C++ tests
```
See `examples/cpp/` for encode/decode, metadata, file API, and iterator examples.

**Python bindings** (PyO3 + maturin, requires [uv](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv venv .venv && source .venv/bin/activate
uv pip install maturin numpy
cd python/bindings && maturin develop
python -m pytest python/tests/ -v              # ~400 tests
```

**xarray + Zarr backends:**
```bash
source .venv/bin/activate                      # activate venv from above step
uv pip install -e "python/tensogram-xarray/[dask]"    # ~190 tests
uv pip install -e python/tensogram-zarr/              # ~220 tests
```

**GRIB conversion** (requires [ecCodes](https://confluence.ecmwf.int/display/ECC)):
```bash
cargo build -p tensogram-cli --features grib
tensogram convert-grib forecast.grib -o forecast.tgm
```

**NetCDF conversion** (requires [libnetcdf](https://www.unidata.ucar.edu/software/netcdf/)):
```bash
cargo build -p tensogram-cli --features netcdf
tensogram convert-netcdf --cf --compression zstd forecast.nc -o forecast.tgm
```

## Support

This software is developed by ECMWF and provided on a **best-effort** basis.
No operational support is provided. For questions, bug reports, or feature
requests please [open a GitHub issue](https://github.com/ecmwf/tensogram/issues).
For general enquiries about ECMWF software, visit the
[ECMWF Support Portal](https://support.ecmwf.int).

## Documentation

- [mdbook docs](docs/) — full developer guide (`cd docs && mdbook build`)
- [Architecture](ARCHITECTURE.md) — crate structure and design decisions
- [Contributing](CONTRIBUTING.md) — setup and workflow
- [Code of Conduct](CODE_OF_CONDUCT.md) — community guidelines
- [Changelog](CHANGELOG.md) — release history
- [Python API](docs/src/guide/python-api.md) — encoding, decoding, file API, validation

## Repository Layout

```
rust/
├── tensogram-core/       Core encode/decode library
├── tensogram-encodings/  Encoding pipeline + compression codecs
├── tensogram-cli/        CLI binary (tensogram command)
├── tensogram-ffi/        C FFI layer
├── tensogram-szip/       Pure-Rust szip implementation
├── tensogram-sz3/        SZ3 lossy compressor bindings
├── tensogram-wasm/       WebAssembly bindings
├── tensogram-grib/       GRIB converter (ecCodes, excluded from default build)
├── tensogram-netcdf/     NetCDF converter (libnetcdf, excluded from default build)
└── benchmarks/           Benchmark suite
python/
├── bindings/             Python bindings (PyO3, excluded from default build)
├── tensogram-xarray/     xarray backend engine (Python package)
├── tensogram-zarr/       Zarr v3 store backend (Python package)
└── tests/                Python test suite
cpp/
├── include/              C++ wrapper header + C header
├── tests/                C++ GoogleTest suite
└── CMakeLists.txt        CMake build system
examples/{rust,cpp,python}/
docs/                     mdBook documentation
.github/workflows/ci.yml  CI matrix (Rust, Python, C++, GRIB, xarray, zarr, docs)
```

## Copyright and License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

This software is licensed under the terms of the [Apache License, Version 2.0](LICENSE) which can also be obtained at http://www.apache.org/licenses/LICENSE-2.0.

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

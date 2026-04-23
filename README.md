<p align="center">
  <img src="logo.png" alt="Tensogram" width="200">
</p>

<p align="center">
  <em>A fast, efficient 'telegram' for multidimensional tensors</em>
</p>

<p align="center">
  <a href="https://crates.io/crates/tensogram">
    <img src="https://img.shields.io/crates/v/tensogram.svg" alt="crates.io">
  </a>
  <a href="https://pypi.org/project/tensogram/">
    <img src="https://img.shields.io/pypi/v/tensogram.svg" alt="PyPI">
  </a>
  <a href="https://www.npmjs.com/package/@ecmwf.int/tensogram">
    <img src="https://img.shields.io/npm/v/@ecmwf.int/tensogram.svg" alt="npm">
  </a>
  <a href="https://sites.ecmwf.int/docs/tensogram/main">
    <img src="https://img.shields.io/badge/docs-online-blue" alt="Documentation">
  </a>
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg" alt="Emerging">
  </a>
</p>

> [!IMPORTANT]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

A library to encode and decode binary N-tensor scientific data with semantic metadata attached to the data.

Tensogram is designed for scientific computing at scale — Machine Learning, Weather and Climate, Earth observation, Fluid mechanics,  Materials simulation, etc. It is vocabulary-agnostic: the library never interprets metadata keys, so application layers (conventions, in-house namespaces, etc) own meaning.

Tensogram defines a binary message format, not strictly a file format. Multiple messages can be appended to a file, each carrying its own begin/terminator codes. This means it is a serialisable format that can be sent over the network, encoded into in-memory buffers, saved to file-systems, stored in archival systems and decoded with zero-copy.

**NOTE** Tensogram is develop through Agentic Engineering by Humans working in tandem with AI Code Agents.

## Features

- **Self-describing messages** — CBOR-encoded metadata vocabulary agnostic
- **N-Tensor support** — multiple tensors of different dtypes per message (float16 through float64, int8 through int64, complex, bfloat16)
- **Robustness** — robust Rust library with 'no panic' policy where all fallible operations return `Result<T, TensogramError>`
- **Partial decode** — `decode_range` minimises IO by extracting sub-tensor slices without decoding the full data objects, with random-access support for szip, blosc2, and zfp.
- **File API** — `TensogramFile` for multi-message `.tgm` files: append, random-access read, iterate, and decode individual messages or objects
- **Remote access** — read `.tgm` files directly from S3, GCS, Azure Blob, or HTTP via `object_store` integration (`open_remote`, `open_source`)
- **Async API** — full async counterparts for file open, message read, decode, and iteration via tokio (`open_async`, `decode_message_async`, etc.)
- **Streaming encoder** — progressive encode/transmit without buffering the full message; preceder metadata frames enable consumer-side streaming decode
- **Compression** — szip, zstd, lz4, blosc2, zfp, sz3 per data object; pure-Rust backends available (`szip-pure`, `zstd-pure`) for environments without C libraries
- **Hash verified integrity** — xxHash xxh3-64 integrity check per frame
- **Validation** — 4-level structural and data integrity validation with optional JSON output (`tensogram validate --quick|--checksum|--full`)
- **Threaded** - multi-threaded encoding/decoding
- **CLI** — `tensogram info/ls/dump/get/set/copy/merge/split/reshuffle/convert-grib/convert-netcdf` with `--strategy first|last|error` merge conflict resolution
- **Optional features** — `mmap` (zero-copy file reads), `async` (tokio I/O), `remote` (S3/GCS/Azure/HTTP)

## Language Support

- **Multiple languages** — Rust, Python (NumPy), C/C++, WebAssembly
- **Free-threaded Python** — GIL-free operation on Python 3.13t with full parallel encode/decode

## Extensions

- **xarray backend** — `xr.open_dataset("file.tgm", engine="tensogram")` with lazy loading, coordinate auto-detection, and hypercube stacking via `open_datasets()`
- **Dask integration** — parallel chunked computation via `xr.open_dataset(..., chunks={})` with per-chunk `decode_range` for efficient out-of-core processing
- **Zarr v3 store** — `zarr.open_group(store=TensogramStore.open_tgm("file.tgm"), mode="r")` for standard Zarr API access with 14 bidirectionally-mapped dtypes
- **Anemoi-inference output** — store AI weather forecast steps directly to `.tgm` via an auto-discovered plugin; each step is encoded and appended immediately, with optional pressure-level stacking, lossy simple packing, variable filtering, and remote (S3/GCS/Azure) output
- **GRIB import** — bring GRIB data into Tensogram with ecCodes-driven metadata lifting and configurable namespace extraction
- **NetCDF import** — bring NetCDF-3 and NetCDF-4 files in with CF metadata lifting (`--cf`), packed-data unpacking, and a configurable encoding/compression pipeline shared with `convert-grib`


## Installation

### Rust

```bash
cargo add tensogram
```

With optional features:
```bash
cargo add tensogram --features mmap,async,remote
```

### Python

```bash
pip install tensogram
```

With xarray and Zarr backends:
```bash
pip install tensogram[all]
```

### anemoi-inference plugin

```bash
pip install tensogram-anemoi
```

Once installed, anemoi-inference auto-discovers the output plugin. Configure it in your run YAML:

```yaml
output:
  tensogram:
    path: forecast.tgm       # local path or s3://, gs://, az://, ...
    compression: zstd        # none | zstd | lz4 | szip | blosc2
    dtype: float32
    stack_pressure_levels: true
```

Each forecast step is encoded and appended to the `.tgm` file as it is produced. See `examples/python/` and the [plugin docs](docs/src/guide/anemoi-integration.md) for the full option reference and reading the output back.

### CLI

```bash
cargo install tensogram-cli
```

With GRIB/NetCDF importers (requires system libraries):
```bash
cargo install tensogram-cli --features grib,netcdf
```

---

## Quick Start

### Rust
```rust
let desc = DataObjectDescriptor {
    obj_type: "ntensor".into(), ndim: 2,
    shape: vec![100, 200], strides: vec![200, 1],
    dtype: Dtype::Float32, byte_order: ByteOrder::Big,
    encoding: "simple_packing".into(), filter: "none".into(),
    compression: "szip".into(), masks: None, params,
};

let message = encode(&meta, &[(&desc, &raw)], &EncodeOptions::default())?;
let (_, objects) = decode(&message, &DecodeOptions::default())?;
```

See `examples/rust/` for metadata, streaming, compression, the file API, and more.

### Python

```python
data = np.random.randn(100, 200).astype(np.float32)
# Application metadata (names, units, vocabularies) lives in meta["base"][i].
# The "product" key below is an example namespace — the library is
# vocabulary-agnostic, so any application-defined keys work the same way.
# The descriptor dict is strictly for tensor shape/dtype and encoding parameters.
msg = tensogram.encode(
    {"version": 3, "base": [{"product": {"name": "temperature", "units": "K"}}]},
    [({"type": "ntensor", "shape": [100, 200], "dtype": "float32",
       "encoding": "simple_packing", "compression": "szip"}, data)],
)
result = tensogram.decode(msg)
arr = result.objects[0][1]  # numpy array
```

For default zarr/xarray variable discovery, `meta["base"][i]["name"]` is
the simplest form; use `variable_key="product.name"` (or any dotted path)
to pick a name out of a namespaced vocabulary.

### xarray

```python
ds = xr.open_dataset("data.tgm", engine="tensogram")  # lazy-loaded
```

### Zarr v3

```python
# open all .tgm files in a directory
group = zarr.open_group(store=TensogramStore.open_dir("somedir/"), mode="r")  # loads somedir/*.tgm
```

See `examples/python/` for encode/decode, metadata, packing, file API, iterators, xarray, zarr, and streaming consumer patterns. For **narrative walk-throughs** with live visualisations, see `examples/jupyter/` — five Jupyter notebooks covering quickstart/MARS, encoding pipeline fidelity, GRIB conversion, NetCDF → xarray, and validation + multi-threaded encoding.

## Build & Test

```bash
cargo build --workspace                                          # build
cargo test --workspace                                           # test
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lint
```

**Optional features:**
```bash
cargo build -p tensogram --features mmap,async,remote
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

**GRIB import** (requires [ecCodes](https://confluence.ecmwf.int/display/ECC)):
```bash
cargo build -p tensogram-cli --features grib
tensogram convert-grib input.grib -o output.tgm
```

**NetCDF import** (requires [libnetcdf](https://www.unidata.ucar.edu/software/netcdf/)):
```bash
cargo build -p tensogram-cli --features netcdf
tensogram convert-netcdf --cf --compression zstd input.nc -o output.tgm
```

## Support and community

Tensogram is developed and maintained by ECMWF with community contributions
welcome. The software is provided on a **best-effort** basis; no operational
support is provided. For questions, bug reports, or feature requests please
[open a GitHub issue](https://github.com/ecmwf/tensogram/issues) or a pull
request. Contributions — bug fixes, documentation, new language bindings,
integrations with domain tooling — are welcome.

## Documentation

Full documentation is available online at **[sites.ecmwf.int/docs/tensogram/main](https://sites.ecmwf.int/docs/tensogram/main)**.

The source markdown files live under [`docs/src/`](docs/src/) and are easy to browse directly on GitHub.

To build and serve locally:

```bash
cargo install mdbook mdbook-mermaid mdbook-tabs
mdbook serve docs/
```

Then open `http://localhost:3000`.

### Other resources

- [Architecture](plans/ARCHITECTURE.md) — crate structure and design decisions
- [Contributing](CONTRIBUTING.md) — setup and workflow
- [Code of Conduct](CODE_OF_CONDUCT.md) — community guidelines
- [Changelog](CHANGELOG.md) — release history

## Repository Layout

High-level shape of the repository.

- **`rust/`** — Core Rust workspace: encode/decode library, encoding pipeline,
  compression codec crates, CLI binary, C FFI layer, WASM bindings, GRIB and
  NetCDF importers, benchmarks.
- **`python/`** — PyO3 bindings (published as `tensogram` on PyPI) plus
  pure-Python extras for xarray, Zarr v3, and anemoi-inference integration.
- **`cpp/`** — C++ wrapper over the C FFI: headers, CMake build, GoogleTest suite.
- **`typescript/`** — TypeScript bindings over the WASM crate, published as
  `@ecmwf.int/tensogram` on npm.
- **`tensoscope/`** — Browser-based interactive `.tgm` viewer, built on the
  TypeScript + WASM package.
- **`examples/`** — Runnable examples per language (Rust, C++, Python,
  TypeScript, Jupyter).
- **`docs/`** — mdBook user guide published at
  [sites.ecmwf.int/docs/tensogram/main](https://sites.ecmwf.int/docs/tensogram/main).
- **`plans/`** — Design documents: architecture, wire-format spec, style guide,
  roadmap.
- **`.github/workflows/`** — CI matrix, publish workflows (crates.io, PyPI,
  npm), docs deployment, and CI image build.

## Copyright and License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

This software is licensed under the terms of the [Apache License, Version 2.0](LICENSE) which can also be obtained at http://www.apache.org/licenses/LICENSE-2.0.

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

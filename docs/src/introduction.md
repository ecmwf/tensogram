<div class="landing-logo">
  <img src="images/logo.png" alt="Tensogram logo" />
</div>

# Introduction

Tensogram is a binary message format for **N-dimensional scientific tensors** —
the kind of data that appears in weather and climate forecasting, Earth
observation, medical and microscopy imaging, genomics, particle physics,
materials simulation, and machine-learning pipelines. It carries its own
metadata, supports arbitrary tensor dimensions, and is fast to encode and
decode.

## What Tensogram gives you

- **Self-describing messages.** Every message carries the metadata needed to
  decode it — shape, dtype, encoding pipeline, application annotations — using
  CBOR. No external schema required.
- **Any number of dimensions.** A single message can carry multiple tensors,
  each with its own shape, dtype, and encoding. A 3-D spectrum, a 2-D field,
  and a 4-D ensemble tensor can coexist in one message.
- **Vocabulary-agnostic.** The library never interprets metadata keys.
  Application layers (MARS at ECMWF, CF in climate, BIDS in neuroimaging, your
  in-house taxonomy) own key names.
- **Transport and file in one format.** The same bytes that traverse a socket
  can be appended to a `.tgm` file; both support O(1) random access to any
  object.
- **Interop with existing formats.** Importers for GRIB and NetCDF let you
  bring existing data into Tensogram pipelines without a lossy re-modelling
  step.
- **Partial range decode.** Extract sub-tensor slices without decoding the
  whole object — useful for remote data at scale.

Tensogram is developed and maintained by ECMWF and is used in operational
weather-forecasting workloads, but nothing in the format is weather-specific.
The design targets the N-tensor-at-scale problem common to many scientific
domains.

## Crate Layout

The primary four Rust crates make up the default workspace build:

```
tensogram/
├── rust/
│   ├── tensogram        ← encode, decode, framing, file API,
│   │                            validation, remote object store
│   ├── tensogram-encodings   ← simple_packing, shuffle, compression
│   ├── tensogram-cli         ← `tensogram` command-line tool
│   └── tensogram-ffi         ← C FFI layer for C/C++ callers
├── python/
│   └── bindings/             ← Python bindings (PyO3 / maturin)
├── cpp/
│   └── include/              ← C++ wrapper header + C header
```

On top of those, the repository ships several opt-in crates — the
`tensogram-grib` / `tensogram-netcdf` importers (exposed as the
`convert-grib` / `convert-netcdf` CLI subcommands), the `tensogram-wasm`
WebAssembly bindings, and the pure-Rust `tensogram-szip` /
`tensogram-sz3` / `tensogram-sz3-sys` compression crates — together
with the separate Python packages `tensogram-xarray` (xarray backend)
and `tensogram-zarr` (Zarr v3 store backend), and a `tensogram-benchmarks`
crate. See [`plans/ARCHITECTURE.md`](https://github.com/ecmwf/tensogram/blob/main/plans/ARCHITECTURE.md)
for the full crate list and build recipes.

Most users interact with `tensogram` and the CLI. The encodings
crate is used internally by the core but is also importable directly
if you need to call the encoding functions outside of a full message.

## Installation

**Rust:**
```bash
cargo add tensogram
```

**Python:**
```bash
pip install tensogram          # core
pip install tensogram[all]     # with xarray + zarr backends
```

**CLI:**
```bash
cargo install tensogram-cli
```

See the [Quick Start](guide/quickstart.md) for feature flags, optional dependencies, and detailed setup.

## Quick Example

```rust
use std::collections::BTreeMap;
use tensogram::{
    encode, decode, GlobalMetadata, DataObjectDescriptor,
    ByteOrder, Dtype, EncodeOptions, DecodeOptions,
};

// Describe what you're storing: a 100×200 grid of f32 values
let desc = DataObjectDescriptor {
    obj_type: "ntensor".to_string(),
    ndim: 2,
    shape: vec![100, 200],
    strides: vec![200, 1],
    dtype: Dtype::Float32,
    byte_order: ByteOrder::Big,
    encoding: "none".to_string(),
    filter: "none".to_string(),
    compression: "none".to_string(),
    masks: None,
    params: BTreeMap::new(),
};

// `GlobalMetadata::default()` stamps the current wire version (3).
let global_meta = GlobalMetadata::default();

// Your raw bytes (100 × 200 × 4 bytes = 80,000 bytes)
let data = vec![0u8; 100 * 200 * 4];

// Encode into a self-contained message
let message = encode(&global_meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

// Decode it back
let (meta, objects) = decode(&message, &DecodeOptions::default()).unwrap();
assert_eq!(objects[0].0.shape, vec![100, 200]);
assert_eq!(objects[0].1, data);
```

The `message` bytes can be written to a file, sent over a socket, or stored in a database. The receiver does not need any external schema — everything is self-describing.

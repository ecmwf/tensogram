# tensogram

The primary Tensogram library: encode and decode binary N-tensor
scientific messages with self-describing CBOR metadata.

## Usage

```rust
use std::collections::BTreeMap;
use tensogram::{
    encode, decode, ByteOrder, DataObjectDescriptor, DecodeOptions,
    Dtype, EncodeOptions, GlobalMetadata,
};

// Describe the tensor.
let desc = DataObjectDescriptor {
    obj_type: "ntensor".into(),
    ndim: 2,
    shape: vec![100, 200],
    strides: vec![200, 1],
    dtype: Dtype::Float32,
    byte_order: ByteOrder::native(),
    encoding: "none".into(),
    filter: "none".into(),
    compression: "none".into(),
    masks: None,
    params: BTreeMap::new(),
};

// `GlobalMetadata::default()` stamps the current wire version (3).
let meta = GlobalMetadata::default();
let data = vec![0u8; 100 * 200 * 4];

let message = encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
let (_, objects) = decode(&message, &DecodeOptions::default())?;
assert_eq!(objects[0].1, data);
# Ok::<(), tensogram::TensogramError>(())
```

## Features

- Self-describing messages — CBOR metadata travels with the data
- Multiple tensors per message, each with its own shape, dtype, and
  encoding pipeline
- `TensogramFile` for multi-message `.tgm` files with O(1) random
  access
- Partial range decode via `decode_range`
- Optional Cargo features: `mmap` (zero-copy reads), `async` (tokio),
  `remote` (S3/GCS/Azure/HTTP via `object_store`)

## Installation

```bash
cargo add tensogram
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.

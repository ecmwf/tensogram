# Quick Start

This page walks you through encoding and decoding a real tensor — a 2D temperature field — in about 20 lines of Rust.

## Add the Dependency

```toml
[dependencies]
tensogram-core = { path = "../crates/tensogram-core" }
```

## Encode a Temperature Field

```rust
use std::collections::BTreeMap;
use tensogram_core::{
    encode, decode, GlobalMetadata, DataObjectDescriptor,
    ByteOrder, Dtype, EncodeOptions, DecodeOptions,
};

fn main() {
    // 1. Make some fake temperature data: 100×200 float32 grid
    //    In production, this would come from your model output.
    let shape = vec![100u64, 200];
    let strides = vec![200u64, 1]; // C-contiguous (row-major)
    let num_elements = 100 * 200;
    let data: Vec<u8> = (0..num_elements)
        .flat_map(|i| (273.15f32 + (i as f32 / 100.0)).to_be_bytes())
        .collect();

    // 2. Describe the tensor
    let global = GlobalMetadata {
        version: 2,
        extra: BTreeMap::new(),
    ..Default::default()
    };

    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape,
        strides,
        dtype: Dtype::Float32,
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None, // hash is added automatically by EncodeOptions::default()
    };

    // 3. Encode — produces a self-contained message
    let message = encode(global, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

    println!("Encoded {} bytes", message.len());

    // 4. Decode it back
    let (meta, objects) = decode(&message, &DecodeOptions::default()).unwrap();

    println!(
        "Decoded: {} objects, shape {:?}, dtype {}",
        objects.len(),
        objects[0].0.shape,
        objects[0].0.dtype,
    );
    assert_eq!(objects[0].1, data);
}
```

## Add MARS Metadata

Real messages need application-layer metadata so downstream tools know what the data represents:

```rust
use ciborium::Value;

// Build mars namespace: mars.class = "od", mars.param = "2t"
let mars_map = vec![
    (Value::Text("class".into()), Value::Text("od".into())),
    (Value::Text("date".into()),  Value::Text("20260401".into())),
    (Value::Text("step".into()),  Value::Integer(6.into())),
    (Value::Text("type".into()),  Value::Text("fc".into())),
];

let mut extra = BTreeMap::new();
extra.insert("mars".to_string(), Value::Map(mars_map));

let global = GlobalMetadata {
    version: 2,
    extra, // message-level MARS keys here
};

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
    params: BTreeMap::new(),
    hash: None,
};
```

## What's Next?

- Use [simple_packing](../encodings/simple-packing.md) to reduce payload size by 4-8x
- Use the [File API](file-api.md) to append many messages to a `.tgm` file
- Use the [CLI](../cli/ls.md) to inspect files without writing any code

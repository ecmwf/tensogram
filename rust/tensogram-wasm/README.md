# tensogram-wasm

WebAssembly bindings for Tensogram via `wasm-bindgen`.

This crate compiles Tensogram's encode / decode / scan / streaming
API to WebAssembly. It is the underlying blob used by the TypeScript
package [`@ecmwf.int/tensogram`](../../typescript/README.md); most
JavaScript callers should use that package rather than the raw
bindings here.

## Building

```bash
cargo install wasm-pack
wasm-pack build rust/tensogram-wasm --target web
```

This produces `rust/tensogram-wasm/pkg/` containing the `.wasm`,
`.js` glue, and `.d.ts` TypeScript types.

## Usage (from JavaScript, after `wasm-pack build`)

```javascript
import init, { encode, decode } from './pkg/tensogram_wasm.js';

await init();

const payload = new Float32Array(100 * 200);
const message = encode(
    { version: 3 },
    [{
        descriptor: {
            type: 'ntensor', ndim: 2,
            shape: [100, 200], strides: [200, 1],
            dtype: 'float32', byte_order: 'little',
            encoding: 'none', filter: 'none', compression: 'none',
        },
        data: payload,
    }],
);

const decoded = decode(message);
console.log(decoded.objectCount());
```

## Supported codecs

`lz4`, `szip` (via pure-Rust `tensogram-szip`), and `zstd` (via
pure-Rust `ruzstd`). `blosc2`, `zfp`, and `sz3` are not available
in WebAssembly — the underlying libraries have C dependencies that
do not compile to `wasm32-unknown-unknown`.

## Installation

```bash
cargo add tensogram-wasm
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- TypeScript user guide: <https://sites.ecmwf.int/docs/tensogram/main/guide/typescript-api.html>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.

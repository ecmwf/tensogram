# tensogram-wasm

WebAssembly bindings for Tensogram via wasm-bindgen.

Enables Tensogram in web browsers and WASM runtimes. Foundation for TypeScript wrapper. Targets `wasm32-unknown-unknown`.

## Usage

```javascript
import { encode, decode } from 'tensogram-wasm';

const message = encode(metadata, objects);
const result = decode(message);
```

## Building

```bash
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg
```

Generates `tensogram_wasm.js`, `tensogram_wasm_bg.wasm`, and `.d.ts`.

## Installation

```toml
[dependencies]
tensogram-wasm = "0.1"
```

## Documentation

- Full docs: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.

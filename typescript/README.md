# @ecmwf.int/tensogram

TypeScript bindings for [Tensogram](https://github.com/ecmwf/tensogram),
a binary message format for N-dimensional scientific tensors. Tensogram is
developed and maintained by ECMWF.

Wraps the Rust core (via WebAssembly) with a typed, idiomatic TS API:

- Strong types for metadata, descriptors, dtypes, and errors
- Dtype-aware payload dispatch (`object.data()` returns the correct TypedArray)
- Web Streams API for progressive decode (`decodeStream`)
- Node + browser file helpers with HTTP Range support (`TensogramFile`)
- Full API parity with Rust / Python / C++ (encode/decode/validate,
  streaming encoder, pre-encoded bytes, first-class half-precision +
  complex view classes)

## Install

Not yet published to npm. Build from source:

```bash
# From the repo root
cd typescript
npm install
npm run build     # wasm-pack + tsc
```

Requirements:
- Rust + [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- Node ≥ 20

## Quick start

```ts
import { init, encode, decode } from '@ecmwf.int/tensogram';

await init();

const temps = new Float32Array(100 * 200);
for (let i = 0; i < temps.length; i++) temps[i] = 273.15 + i / 100;

const msg = encode(
  { version: 3 },
  [{
    descriptor: {
      type: 'ntensor', ndim: 2,
      shape: [100, 200], strides: [200, 1],
      dtype: 'float32', byte_order: 'little',
      encoding: 'none', filter: 'none', compression: 'none',
    },
    data: temps,
  }],
);

const { metadata, objects } = decode(msg);
console.log(objects[0].data());  // Float32Array(20000)
```

See the [user guide](../docs/src/guide/typescript-api.md) for the full API
and [examples/typescript/](../examples/typescript/) for runnable scripts.

## Licence

Apache 2.0 — see [LICENSE](../LICENSE).

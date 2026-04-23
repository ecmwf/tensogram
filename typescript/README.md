# @ecmwf.int/tensogram

TypeScript bindings for [Tensogram](https://github.com/ecmwf/tensogram),
a binary message format for N-dimensional scientific tensors. Tensogram is
developed and maintained by ECMWF.

Wraps the Rust core (via WebAssembly) with a typed, idiomatic TS API:

- Strong types for metadata, descriptors, dtypes, and errors
- Dtype-aware payload dispatch (`object.data()` returns the correct TypedArray)
- Web Streams API for progressive decode (`decodeStream`)
- Node + browser file helpers with HTTP Range support (`TensogramFile`)
- **Layout-aware remote reads** — `messageMetadata`, `messageDescriptors`,
  `messageObject`, `messageObjectRange`, `messageObjectBatch`,
  `messageObjectRangeBatch`, `prefetchLayouts`; on the lazy HTTP backend
  each fetches only the bytes it needs (header chunk, footer suffix,
  per-object frame), with bounded-concurrency fan-out
- **AWS SigV4 helper** — `createAwsSigV4Fetch` plugs into
  `TensogramFile.fromUrl({ fetch })` for read-only S3-compatible
  endpoints; pure signer at `signAwsV4Request` for byte-for-byte
  AWS sig-v4-test-suite parity
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

## Remote access

`TensogramFile.fromUrl` opens a `.tgm` over HTTP(S) with Range support
detection.  When the server advertises `Accept-Ranges: bytes` it falls
into the lazy backend, which fetches only the bytes you ask for:

```ts
const file = await TensogramFile.fromUrl(
  'https://example.org/forecast.tgm',
  { concurrency: 6 },               // per-host fan-out cap (default 6)
);
try {
  // Header chunk only — no payload bytes downloaded.
  const meta = await file.messageMetadata(0);

  // Index frame + per-frame CBOR descriptors.
  const { descriptors } = await file.messageDescriptors(0);

  // Exactly one Range GET for object 0's frame.
  const obj = await file.messageObject(0, 0);
  try {
    const arr = obj.objects[0].data() as Float32Array;
    // ... use arr ...
  } finally {
    obj.close();
  }

  // Pre-warm layout cache for all messages, bounded to 6 concurrent fetches.
  await file.prefetchLayouts(
    Array.from({ length: file.messageCount }, (_, i) => i),
  );

  // Parallel decode the same object across many messages.
  const xs = await file.messageObjectBatch([0, 1, 2, 3], 0);
  xs.forEach((x) => x.close());
} finally {
  file.close();
}
```

For S3-compatible endpoints behind AWS Signature V4 authentication,
build a signed `fetch` and pass it through `FromUrlOptions.fetch`:

```ts
import { createAwsSigV4Fetch, TensogramFile } from '@ecmwf.int/tensogram';

const signedFetch = createAwsSigV4Fetch({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  region: 'eu-west-1',
});

const file = await TensogramFile.fromUrl(
  'https://my-bucket.s3.eu-west-1.amazonaws.com/data.tgm',
  { fetch: signedFetch },
);
```

The pure signer (`signAwsV4Request`) is also exported for callers
that want to handle the request lifecycle themselves.  Both are
covered by byte-for-byte AWS sig-v4-test-suite parity tests.

For Azure or Google Cloud Storage, generate a presigned (signed) URL
in your control plane and pass it as a plain HTTPS URL — Azure Shared
Key signing and GCS HMAC are out of scope for this package.

For runnable end-to-end demos, see
[`examples/typescript/15_remote_access.ts`](../examples/typescript/15_remote_access.ts),
[`16_remote_batch.ts`](../examples/typescript/16_remote_batch.ts), and
[`17_remote_s3_signed_fetch.ts`](../examples/typescript/17_remote_s3_signed_fetch.ts).

## Licence

Apache 2.0 — see [LICENSE](../LICENSE).

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 11 — Pre-encoded payloads (TypeScript)
 *
 * Some callers already have encoded bytes (another Tensogram
 * implementation, a GPU pipeline, a cached tile store).
 * `encodePreEncoded` stamps those bytes directly into a wire-format
 * message — no second encoding pipeline pass — while still recomputing
 * the integrity hash over the supplied bytes.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 11_encode_pre_encoded.ts
 */

import {
  computeHash,
  decode,
  encodePreEncoded,
  init,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

function describe(shape: number[], dtype: DataObjectDescriptor['dtype']): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype,
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

async function main(): Promise<void> {
  await init();

  // Raw bytes that represent a 4×4 float32 image.  For this demo we
  // use `encoding: 'none'`, so "pre-encoded" is just the native bytes.
  // In a real pipeline the bytes might be szip-compressed with
  // `szip_block_offsets` already computed.
  const img = new Float32Array(16);
  for (let i = 0; i < 16; i++) img[i] = i * 1.5;
  const bytes = new Uint8Array(img.buffer, img.byteOffset, img.byteLength);

  // Stamp into a message.  encodePreEncoded does not re-encode the
  // bytes; it only validates the descriptor and wraps the bytes in a
  // data-object frame.
  const msg = encodePreEncoded(
    { version: 3, base: [{ mars: { param: 'tile' } }] },
    [{ descriptor: describe([4, 4], 'float32'), data: bytes }],
  );
  console.log(`pre-encoded message: ${msg.byteLength} bytes`);

  // Verify the hash stamped on the descriptor equals computeHash(bytes).
  const decoded = decode(msg);
  try {
    const hash = decoded.objects[0].descriptor.hash;
    console.log(`descriptor hash: ${hash?.type} ${hash?.value}`);
    const expected = computeHash(bytes);
    console.log(`computeHash(bytes) matches: ${hash?.value === expected}`);

    // Decoded payload equals the original bytes (encoding=none).
    const out = decoded.objects[0].data() as Float32Array;
    const matches = out.length === img.length && out.every((v, i) => v === img[i]);
    console.log(`decode round-trip: ${matches ? 'OK' : 'MISMATCH'}`);
  } finally {
    decoded.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

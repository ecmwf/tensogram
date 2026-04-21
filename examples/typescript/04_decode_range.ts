// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 04 — Partial range decode (TypeScript)
 *
 * For large tensors where you only need a slab, `decodeRange` avoids
 * paying the cost of a full decompress + dequantise.  The example
 * builds an uncompressed 100 000-element float32 tensor, then pulls
 * three disjoint slabs by element offset/count — and also shows the
 * `join: true` mode that returns a single concatenated buffer.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 04_decode_range.ts
 */

import {
  decodeRange,
  encode,
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

  // 1. Build a 100 000-element float32 tensor: values = index * 0.5.
  const N = 100_000;
  const values = new Float32Array(N);
  for (let i = 0; i < N; i++) values[i] = i * 0.5;

  const msg = encode({ version: 2 }, [{ descriptor: describe([N], 'float32'), data: values }]);
  console.log(`message size: ${msg.byteLength.toLocaleString()} bytes`);

  // 2. Split mode: one TypedArray per requested range.
  const split = decodeRange(msg, 0, [
    [0, 5],
    [1000, 3],
    [99_990, 5],
  ]);
  console.log('split mode:');
  for (let i = 0; i < split.parts.length; i++) {
    const arr = split.parts[i] as Float32Array;
    console.log(`  part ${i}: first=${arr[0]}  last=${arr[arr.length - 1]}  length=${arr.length}`);
  }

  // 3. Join mode: a single concatenated buffer — handy when the
  //    consumer just wants "all the bytes I asked for".
  const joined = decodeRange(
    msg,
    0,
    [
      [0, 5],
      [1000, 3],
      [99_990, 5],
    ],
    { join: true },
  );
  const merged = joined.parts[0] as Float32Array;
  console.log(`\njoin mode: ${merged.length} elements total`);
  console.log(`  contents: [${Array.from(merged).join(', ')}]`);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

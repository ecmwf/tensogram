// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 01 — Basic encode / decode round-trip (TypeScript)
 *
 * Encodes a 100×200 float32 temperature grid into a Tensogram message,
 * then decodes it back. Mirrors `examples/python/01_encode_decode.py`.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 01_encode_decode.ts
 */

import {
  decode,
  encode,
  init,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '@ecmwf.int/tensogram';

async function main(): Promise<void> {
  // ── 1. Initialise WASM (one-time, idempotent) ──────────────────────────
  await init();

  // ── 2. Source data ─────────────────────────────────────────────────────
  const temps = new Float32Array(100 * 200);
  for (let i = 0; i < temps.length; i++) {
    temps[i] = 273.15 + i / 100;
  }
  console.log(
    `Input: shape=[100, 200]  dtype=float32  size=${temps.byteLength} bytes`,
  );

  // ── 3. Describe the message ────────────────────────────────────────────
  const metadata: GlobalMetadata = { version: 3 };
  const descriptor: DataObjectDescriptor = {
    type: 'ntensor',
    ndim: 2,
    shape: [100, 200],
    strides: [200, 1],
    dtype: 'float32',
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };

  // ── 4. Encode ──────────────────────────────────────────────────────────
  const message = encode(metadata, [{ descriptor, data: temps }]);
  console.log(`Message: ${message.byteLength} bytes`);
  const magic = new TextDecoder().decode(message.subarray(0, 8));
  const tail = new TextDecoder().decode(message.subarray(message.length - 8));
  console.log(`  magic:      ${magic}`);
  console.log(`  terminator: ${tail}`);

  // ── 5. Decode ──────────────────────────────────────────────────────────
  const result = decode(message);
  try {
    console.log(`\nDecoded: ${result.objects.length} object(s)`);
    const obj = result.objects[0];
    console.log(`  shape:    ${JSON.stringify(obj.descriptor.shape)}`);
    console.log(`  dtype:    ${obj.descriptor.dtype}`);
    console.log(`  hash:     ${obj.descriptor.hash?.type ?? 'none'} (${obj.descriptor.hash?.value ?? ''})`);

    const arr = obj.data() as Float32Array;
    console.log(`  elements: ${arr.length}`);
    console.log(`  first 5:  [${Array.from(arr.slice(0, 5)).join(', ')}]`);

    // Verify bit-exact round-trip
    let mismatches = 0;
    for (let i = 0; i < temps.length; i++) {
      if (arr[i] !== temps[i]) mismatches++;
    }
    console.log(`\nRound-trip mismatches: ${mismatches} / ${temps.length}`);
    if (mismatches !== 0) {
      process.exit(1);
    }
  } finally {
    result.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

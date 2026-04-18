// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 12 — Frame-at-a-time StreamingEncoder (TypeScript)
 *
 * `StreamingEncoder` builds a message frame by frame so callers can
 * emit data as it becomes available.  The output is identical in
 * *semantic* content to `encode(meta, objects)`, but wire-byte layout
 * differs — streaming puts the index and hash frames in the footer,
 * buffered encode puts them in the header.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 12_streaming_encoder.ts
 */

import {
  decode,
  init,
  StreamingEncoder,
  type DataObjectDescriptor,
} from '@ecmwf/tensogram';

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

  const enc = new StreamingEncoder({ version: 2 });
  try {
    // Attach per-object MARS metadata via write_preceder ahead of each
    // data object — handy when application metadata is only known at
    // emission time.
    enc.writePreceder({ mars: { param: '2t', step: 0 }, units: 'K' });
    enc.writeObject(describe([3], 'float32'), new Float32Array([273.15, 274.0, 275.0]));

    enc.writePreceder({ mars: { param: '10u', step: 0 }, units: 'm/s' });
    enc.writeObject(describe([3], 'float32'), new Float32Array([-1.5, 0.0, 2.5]));

    enc.writePreceder({ mars: { param: 'msl', step: 0 }, units: 'Pa' });
    enc.writeObject(describe([3], 'float64'), new Float64Array([101325, 101200, 100950]));

    console.log(
      `streamed: ${enc.objectCount} objects, ${enc.bytesWritten} bytes before finish()`,
    );
    const bytes = enc.finish();
    console.log(`finished: ${bytes.byteLength} bytes total`);

    const msg = decode(bytes);
    try {
      console.log(`decoded: ${msg.objects.length} objects`);
      msg.objects.forEach((obj, i) => {
        const base = msg.metadata.base?.[i];
        const mars = base?.['mars'] as Record<string, unknown> | undefined;
        console.log(`  [${i}] shape=${JSON.stringify(obj.descriptor.shape)}  param=${mars?.['param']}`);
      });
    } finally {
      msg.close();
    }
  } finally {
    enc.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

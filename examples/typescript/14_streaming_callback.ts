// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 14 — StreamingEncoder with callback-per-frame sink
 *
 * `StreamingEncoder` accepts an optional `onBytes` callback.  When
 * present, each chunk of wire-format bytes the encoder produces is
 * forwarded to the callback as it is produced — no internal buffering.
 * Useful for browser uploads to `fetch(..., { body: stream })`,
 * WebSocket pushes, or any streaming sink that needs bytes
 * incrementally.
 *
 * The `finish()` call in streaming mode returns an empty `Uint8Array`
 * (zero-length marker) — every byte has already been delivered to the
 * callback.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 14_streaming_callback.ts
 */

import {
  decode,
  init,
  StreamingEncoder,
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

  // ── Collect every chunk the encoder emits ────────────────────────────
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  const enc = new StreamingEncoder(
    { version: 2 },
    {
      onBytes: (chunk) => {
        // IMPORTANT: copy the chunk if you need it to survive past the
        // next writeObject.  The underlying ArrayBuffer is invalidated
        // when WASM memory grows.
        chunks.push(new Uint8Array(chunk));
        totalBytes += chunk.byteLength;
        console.log(`  chunk #${chunks.length}: ${chunk.byteLength} bytes`);
      },
    },
  );

  console.log('── Streaming mode ──');
  console.log(`after construction: ${chunks.length} chunk(s), ${totalBytes} bytes`);
  console.log(`(preamble + header metadata frame have already been delivered)`);
  console.log(`streaming flag: ${enc.streaming}`);

  // Each writeObject flushes a data-object frame through the callback.
  enc.writeObject(describe([3], 'float32'), new Float32Array([1, 2, 3]));
  enc.writeObject(describe([3], 'float64'), new Float64Array([10, 20, 30]));

  // finish() flushes footer frames + postamble through the callback.
  const returned = enc.finish();
  enc.close();

  console.log(`\nfinish() returned: ${returned.byteLength} bytes (empty in streaming mode)`);
  console.log(`total chunks collected: ${chunks.length}`);
  console.log(`total bytes delivered: ${totalBytes}`);

  // Reassemble the stream and decode to prove semantic equivalence.
  const bytes = new Uint8Array(totalBytes);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.byteLength;
  }
  const msg = decode(bytes);
  try {
    console.log(`\nreassembled message → ${msg.objects.length} object(s)`);
    msg.objects.forEach((obj, i) => {
      console.log(
        `  object[${i}]: dtype=${obj.descriptor.dtype}, shape=${JSON.stringify(obj.descriptor.shape)}`,
      );
    });
  } finally {
    msg.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 05 — Progressive decode over a ReadableStream (TypeScript)
 *
 * Demonstrates {@link decodeStream}: feed a stream of bytes (e.g. from
 * `fetch().body`) and receive decoded data-object frames as complete
 * messages arrive. Corrupt messages are skipped and observed through the
 * `onError` callback without interrupting the stream.
 *
 * This example synthesises a stream in-process, so it runs anywhere;
 * replace the `streamOf(...)` call with `(await fetch(url)).body!` to
 * drive it off a real HTTP request in a browser.
 */

import {
  decodeStream,
  encode,
  init,
  type DataObjectDescriptor,
  type StreamDecodeError,
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

/** Feed bytes into a ReadableStream split into small chunks. */
function streamOf(bytes: Uint8Array, chunkSize = 1024): ReadableStream<Uint8Array> {
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (offset >= bytes.byteLength) {
        controller.close();
        return;
      }
      const end = Math.min(offset + chunkSize, bytes.byteLength);
      controller.enqueue(bytes.subarray(offset, end));
      offset = end;
    },
  });
}

async function main(): Promise<void> {
  await init();

  // Build a multi-message "feed": three temperature fields back-to-back.
  const encoder = (values: number[], param: string): Uint8Array =>
    encode(
      { version: 2, base: [{ mars: { param } }] },
      [{ descriptor: describe([values.length], 'float32'), data: new Float32Array(values) }],
    );

  const m1 = encoder([273.15, 274.0, 275.0], '2t');
  const m2 = encoder([0.0, 1.0, 2.0, 3.0, 4.0], '10u');
  const m3 = encoder([101_325, 100_000], 'msl');

  const concatenated = new Uint8Array(m1.byteLength + m2.byteLength + m3.byteLength);
  concatenated.set(m1, 0);
  concatenated.set(m2, m1.byteLength);
  concatenated.set(m3, m1.byteLength + m2.byteLength);
  console.log(`feed size: ${concatenated.byteLength} bytes across 3 messages\n`);

  const stream = streamOf(concatenated, /* chunkSize */ 256);

  const errors: StreamDecodeError[] = [];
  let frameIndex = 0;
  for await (const frame of decodeStream(stream, {
    maxBufferBytes: 16 * 1024 * 1024,
    onError: (err) => errors.push(err),
  })) {
    const data = frame.data() as Float32Array;
    const param = (frame.baseEntry as Record<string, { param: string }> | null)?.mars?.param ?? '?';
    console.log(
      `frame ${frameIndex++}  dtype=${frame.descriptor.dtype}  ` +
        `shape=${JSON.stringify(frame.descriptor.shape)}  param=${param}  ` +
        `first=${data[0]}`,
    );
    frame.close();
  }

  console.log(`\ntotal frames: ${frameIndex}  errors reported: ${errors.length}`);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

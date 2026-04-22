// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 09 — Streaming consumer (TypeScript)
 *
 * Decode tensogram messages as bytes arrive on a `ReadableStream`
 * without ever buffering the full file.  On every chunk the consumer
 * calls `scan(buf)` on its rolling buffer, decodes each completed
 * message, and drops the consumed prefix — peak memory is one complete
 * message, not the whole stream.
 *
 * This is the explicit-loop counterpart to {@link decodeStream} (see
 * `05_streaming_fetch.ts`).  Use `decodeStream` when you want the
 * library to manage the buffer internally; use this pattern when the
 * caller needs direct visibility into message boundaries (progress
 * bars, per-message cancellation, mixing with other protocols on the
 * same byte stream, ...).
 *
 * The example synthesises a stream in-process so it runs anywhere;
 * point `streamOf(...)` at `(await fetch(url)).body!` to drive the
 * same loop off a real HTTP response.
 */

import {
  decode,
  encode,
  init,
  scan,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '@ecmwf.int/tensogram';

function describe(shape: number[]): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype: 'float32',
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

function buildFourMessages(): Uint8Array {
  const params: Array<[string, number]> = [
    ['2t', 0],
    ['10u', 0],
    ['10v', 0],
    ['msl', 0],
  ];
  const parts: Uint8Array[] = [];
  for (const [param, step] of params) {
    const metadata: GlobalMetadata = {
      version: 3,
      base: [{ mars: { param, step } }],
    };
    // 181×360 float32 — a realistic gridded field.
    const data = new Float32Array(181 * 360).fill(Math.random());
    parts.push(encode(metadata, [{ descriptor: describe([181, 360]), data }]));
  }
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.byteLength;
  }
  return out;
}

/** Feed bytes into a ReadableStream split into small chunks. */
function streamOf(bytes: Uint8Array, chunkSize = 4096): ReadableStream<Uint8Array> {
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

  const fileBytes = buildFourMessages();
  console.log(`Built ${fileBytes.byteLength} bytes over 4 messages\n`);

  const stream = streamOf(fileBytes, 4096);
  const reader = stream.getReader();

  // Rolling buffer: grows with each chunk, shrinks as complete
  // messages are consumed.  Its peak length is at most one message.
  let buffer = new Uint8Array(0);
  let messagesDecoded = 0;

  console.log('Streaming (chunk size = 4096):');
  console.log('─'.repeat(60));

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Extend the rolling buffer with the new chunk.
    const merged = new Uint8Array(buffer.byteLength + value.byteLength);
    merged.set(buffer);
    merged.set(value, buffer.byteLength);
    buffer = merged;

    // scan tolerates a partial trailing message — it only returns
    // positions it could verify start-to-end.
    const positions = scan(buffer);
    let consumed = 0;
    for (const { offset, length } of positions) {
      const slice = buffer.subarray(offset, offset + length);
      const msg = decode(slice);
      try {
        const param = (msg.metadata.base?.[0] as { mars?: { param: string } } | undefined)
          ?.mars?.param ?? '?';
        messagesDecoded += 1;
        console.log(
          `  message ${messagesDecoded}: param=${param.padEnd(3)}  ` +
          `shape=[${msg.objects[0].descriptor.shape.join(', ')}]  ` +
          `${slice.byteLength} bytes`,
        );
      } finally {
        msg.close();
      }
      consumed = offset + length;
    }

    // Drop the consumed prefix; keep any partial-message tail.
    if (consumed > 0) {
      buffer = buffer.subarray(consumed);
    }
  }

  console.log('─'.repeat(60));
  console.log(`Total: ${messagesDecoded} messages decoded from stream`);
  console.log('Peak buffer: single message (not whole file)');

  if (messagesDecoded !== 4) {
    throw new Error(`expected 4 messages, decoded ${messagesDecoded}`);
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

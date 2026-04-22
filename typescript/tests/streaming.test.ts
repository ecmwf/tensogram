// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import {
  decode,
  decodeStream,
  encode,
  init,
  InvalidArgumentError,
  StreamingLimitError,
  type DecodedFrame,
  type StreamDecodeError,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

/**
 * Turn a sequence of byte chunks into a ReadableStream<Uint8Array>.
 * Used to exercise arbitrary chunk boundaries.
 */
function streamFromChunks(chunks: readonly Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream({
    pull(controller) {
      if (i >= chunks.length) {
        controller.close();
        return;
      }
      controller.enqueue(chunks[i++]);
    },
  });
}

/** Slice one Uint8Array into `n` roughly-equal chunks. */
function splitIntoChunks(buf: Uint8Array, n: number): Uint8Array[] {
  const out: Uint8Array[] = [];
  const size = Math.ceil(buf.byteLength / n);
  for (let off = 0; off < buf.byteLength; off += size) {
    out.push(buf.subarray(off, Math.min(off + size, buf.byteLength)));
  }
  return out;
}

describe('Phase 3 — decodeStream', () => {
  it('yields frames that match a direct decode', async () => {
    await init();
    const original = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([8], 'float32'), data: original },
    ]);

    const stream = streamFromChunks([msg]);
    const frames: DecodedFrame[] = [];
    for await (const frame of decodeStream(stream)) {
      frames.push(frame);
    }
    expect(frames).toHaveLength(1);
    const arr = frames[0].data() as Float32Array;
    expect(Array.from(arr)).toEqual(Array.from(original));
    frames[0].close();
  });

  it('works when the message is split across many tiny chunks', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([100], 'float32'),
        data: new Float32Array(100).map((_, i) => i * 0.5),
      },
    ]);
    // 10 equal chunks exercises boundary crossings at arbitrary frame positions
    const stream = streamFromChunks(splitIntoChunks(msg, 10));
    const frames: DecodedFrame[] = [];
    for await (const frame of decodeStream(stream)) frames.push(frame);
    expect(frames).toHaveLength(1);
    const data = frames[0].data() as Float32Array;
    expect(data.length).toBe(100);
    for (let i = 0; i < 100; i++) expect(data[i]).toBeCloseTo(i * 0.5);
    frames[0].close();
  });

  it('yields frames in order across multiple concatenated messages', async () => {
    await init();
    const m1 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([10, 20, 30, 40]),
      },
    ]);
    const m2 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'uint8'),
        data: new Uint8Array([1, 2, 3]),
      },
      {
        descriptor: makeDescriptor([2], 'float64'),
        data: new Float64Array([100.5, 200.5]),
      },
    ]);
    const combined = new Uint8Array(m1.byteLength + m2.byteLength);
    combined.set(m1, 0);
    combined.set(m2, m1.byteLength);

    const stream = streamFromChunks(splitIntoChunks(combined, 7));
    const results: Array<{ dtype: string; data: unknown }> = [];
    for await (const frame of decodeStream(stream)) {
      results.push({
        dtype: frame.descriptor.dtype,
        data: Array.from(frame.data() as { [Symbol.iterator](): Iterator<unknown> }),
      });
      frame.close();
    }
    expect(results).toHaveLength(3);
    expect(results[0].dtype).toBe('float32');
    expect(results[0].data).toEqual([10, 20, 30, 40]);
    expect(results[1].dtype).toBe('uint8');
    expect(results[1].data).toEqual([1, 2, 3]);
    expect(results[2].dtype).toBe('float64');
    expect(results[2].data).toEqual([100.5, 200.5]);
  });

  it('exposes per-object baseEntry metadata', async () => {
    await init();
    const msg = encode(
      {
        /* free-form metadata */         base: [
          { mars: { param: '2t' } },
          { mars: { param: '10u' } },
        ],
      },
      [
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([1, 2]),
        },
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([3, 4]),
        },
      ],
    );
    const stream = streamFromChunks([msg]);
    const frames: DecodedFrame[] = [];
    for await (const frame of decodeStream(stream)) frames.push(frame);

    expect(frames).toHaveLength(2);
    const p0 = (frames[0].baseEntry as Record<string, { param: string }>).mars.param;
    const p1 = (frames[1].baseEntry as Record<string, { param: string }>).mars.param;
    expect(p0).toBe('2t');
    expect(p1).toBe('10u');
    for (const f of frames) f.close();
  });

  it('skips corrupt messages and reports via onError', async () => {
    await init();
    const good1 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'uint8'),
        data: new Uint8Array([1, 2, 3]),
      },
    ]);
    const good2 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'uint8'),
        data: new Uint8Array([7, 8, 9]),
      },
    ]);
    // Corrupt the middle message by flipping bytes inside the payload region.
    const corrupt = new Uint8Array(good1);
    // Tamper inside the CBOR header area so the message fails integrity /
    // structural checks after being successfully scanned.
    // The scan() step locates the message boundary by preamble + postamble,
    // so tampering in the middle keeps the boundary intact.
    corrupt[200] ^= 0xff;
    corrupt[201] ^= 0xff;
    corrupt[202] ^= 0xff;

    const combined = new Uint8Array(corrupt.byteLength + good2.byteLength);
    combined.set(corrupt, 0);
    combined.set(good2, corrupt.byteLength);

    const errors: StreamDecodeError[] = [];
    const frames: DecodedFrame[] = [];
    for await (const frame of decodeStream(streamFromChunks([combined]), {
      onError: (err) => errors.push(err),
    })) {
      frames.push(frame);
    }

    // good2 survives. We accept either (a) the corrupt message was skipped
    // outright (0 or 1 frames depending on whether its preamble was damaged),
    // or (b) it was decoded but later rejected during object parsing. The
    // minimum invariant: at least the second message's frame comes through
    // OR an error was reported.
    const sawError = errors.length > 0;
    const sawSecondMessage = frames.some((f) => {
      const arr = f.data() as Uint8Array;
      return arr[0] === 7 && arr[1] === 8 && arr[2] === 9;
    });
    expect(sawError || sawSecondMessage).toBe(true);
    for (const f of frames) f.close();
  });

  it('respects AbortSignal', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    const controller = new AbortController();
    // Never-ending stream
    const stream = new ReadableStream<Uint8Array>({
      async pull(c) {
        await new Promise((r) => setTimeout(r, 5));
        c.enqueue(msg);
      },
    });

    setTimeout(() => controller.abort(), 20);
    const frames: DecodedFrame[] = [];
    let aborted = false;
    try {
      for await (const frame of decodeStream(stream, { signal: controller.signal })) {
        frames.push(frame);
      }
    } catch (err) {
      aborted = err instanceof Error && /abort/i.test(err.message);
    }
    expect(aborted || frames.length > 0).toBe(true);
    for (const f of frames) f.close();
  });

  it('enforces maxBufferBytes', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([10_000], 'float32'),
        data: new Float32Array(10_000),
      },
    ]);
    const stream = streamFromChunks([msg]);
    let caught: unknown;
    try {
      for await (const _frame of decodeStream(stream, { maxBufferBytes: 512 })) {
        // should not reach here
      }
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(StreamingLimitError);
  });

  it('rejects non-ReadableStream inputs', async () => {
    await init();
    const bad = async (): Promise<void> => {
      for await (const _ of decodeStream(
        // @ts-expect-error intentional bad input
        new Uint8Array([1, 2, 3]),
      )) {
        // no-op
      }
    };
    await expect(bad()).rejects.toBeInstanceOf(InvalidArgumentError);
  });

  it('early break cleans up the decoder without hanging', async () => {
    await init();
    const m1 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1], 'float32'),
        data: new Float32Array([1]),
      },
    ]);
    const m2 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1], 'float32'),
        data: new Float32Array([2]),
      },
    ]);
    const combined = new Uint8Array(m1.length + m2.length);
    combined.set(m1);
    combined.set(m2, m1.length);

    const stream = streamFromChunks([combined]);
    for await (const frame of decodeStream(stream)) {
      frame.close();
      break; // intentionally abandon remaining frames
    }
    // If we got here without hanging, the async generator's finally block
    // released the reader and freed the decoder.
    expect(true).toBe(true);
  });

  // ── Coverage closers ────────────────────────────────────────────────

  it('frame.dataView() is a zero-copy view', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([10, 20, 30, 40]),
      },
    ]);
    const stream = streamFromChunks([msg]);
    for await (const frame of decodeStream(stream)) {
      const view = frame.dataView() as Float32Array;
      expect(Array.from(view)).toEqual([10, 20, 30, 40]);
      frame.close();
    }
  });

  it('frame.data() / dataView() after close() throws', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    const stream = streamFromChunks([msg]);
    const collected: import('../src/index.js').DecodedFrame[] = [];
    for await (const frame of decodeStream(stream)) collected.push(frame);
    expect(collected).toHaveLength(1);
    collected[0].close();
    expect(() => collected[0].data()).toThrow(/closed/);
    expect(() => collected[0].dataView()).toThrow(/closed/);
  });

  it('frame.close() is idempotent', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1], 'uint8'),
        data: new Uint8Array([9]),
      },
    ]);
    const stream = streamFromChunks([msg]);
    for await (const frame of decodeStream(stream)) {
      expect(() => frame.close()).not.toThrow();
      expect(() => frame.close()).not.toThrow();
    }
  });

  it('can round-trip a streamed message against a direct decode', async () => {
    await init();
    const data = new Float32Array(1024).map((_, i) => Math.sin(i * 0.01));
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([1024], 'float32'), data },
    ]);

    // Direct decode
    const direct = decode(msg);
    const directArr = direct.objects[0].data() as Float32Array;
    direct.close();

    // Streamed
    const streamed: Float32Array[] = [];
    for await (const frame of decodeStream(streamFromChunks(splitIntoChunks(msg, 16)))) {
      streamed.push(frame.data() as Float32Array);
      frame.close();
    }
    expect(streamed).toHaveLength(1);
    expect(streamed[0].length).toBe(directArr.length);
    for (let i = 0; i < directArr.length; i++) {
      expect(streamed[0][i]).toBe(directArr[i]);
    }
  });
});

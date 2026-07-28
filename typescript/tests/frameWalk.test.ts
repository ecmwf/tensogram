// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for the frame walker (`frames`) and the typed message envelope
// (`messageHeader`).  Mirror the Rust core `tensogram::frames` /
// `tensogram::message_header` (`rust/tensogram/src/frame_walk.rs`).

import { describe, expect, it } from 'vitest';
import {
  FramingError,
  InvalidArgumentError,
  StreamingEncoder,
  WIRE_VERSION,
  encode,
  frames,
  init,
  messageHeader,
  scan,
} from '../src/index.js';
import type { Frame, FrameTypeName } from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

/** Bytes in the fixed message preamble — never a frame. */
const PREAMBLE_BYTES = 24;
/** Bytes in the fixed message postamble — never a frame. */
const POSTAMBLE_BYTES = 24;
/** Bytes in every frame header. */
const FRAME_HEADER_BYTES = 16;
/** Frame-type-specific footer sizes: 20 B for type 9, 12 B otherwise. */
function footerBytes(frameTypeCode: number): number {
  return frameTypeCode === 9 ? 20 : 12;
}

/** A buffered (random-access) message with `n` float32 objects. */
function bufferedMessage(n: number): Uint8Array {
  const objects = Array.from({ length: n }, (_, i) => ({
    descriptor: makeDescriptor([4], 'float32'),
    data: new Float32Array([i, i + 1, i + 2, i + 3]),
  }));
  return encode(defaultMeta(), objects);
}

/** A streaming-mode message (footer-side index / hashes). */
function streamedMessage(): Uint8Array {
  const enc = new StreamingEncoder(defaultMeta());
  try {
    enc.writeObject(makeDescriptor([4], 'float32'), new Float32Array([1, 2, 3, 4]));
    return enc.finish();
  } finally {
    enc.close();
  }
}

function typesOf(fs: readonly Frame[]): FrameTypeName[] {
  return fs.map((f) => f.frameType);
}

describe('frames', () => {
  it('walks a buffered message in wire order', async () => {
    await init();
    const msg = bufferedMessage(2);
    // Buffered (random-access) layout, default options: metadata, index
    // and the aggregate hash all live in the header; the two data
    // objects follow.  The preamble / postamble are NOT frames.
    expect(typesOf(frames(msg))).toEqual([
      'HeaderMetadata',
      'HeaderIndex',
      'HeaderHash',
      'NTensorFrame',
      'NTensorFrame',
    ]);
  });

  it('yields one data-object frame per encoded object', async () => {
    await init();
    for (const n of [1, 3, 5]) {
      const fs = frames(bufferedMessage(n));
      expect(fs.filter((f) => f.frameType === 'NTensorFrame')).toHaveLength(n);
    }
  });

  it('pairs every frameType with its wire frameTypeCode', async () => {
    await init();
    const codes: Record<FrameTypeName, number> = {
      HeaderMetadata: 1,
      HeaderIndex: 2,
      HeaderHash: 3,
      FooterHash: 5,
      FooterIndex: 6,
      FooterMetadata: 7,
      PrecederMetadata: 8,
      NTensorFrame: 9,
    };
    for (const f of frames(bufferedMessage(2))) {
      expect(f.frameTypeCode).toBe(codes[f.frameType]);
    }
    for (const f of frames(streamedMessage())) {
      expect(f.frameTypeCode).toBe(codes[f.frameType]);
    }
  });

  it('reports offsets and lengths that are in bounds and never overlap', async () => {
    await init();
    const msg = bufferedMessage(3);
    let prevEnd = PREAMBLE_BYTES;
    for (const f of frames(msg)) {
      // Offsets are relative to the message start, after the preamble.
      expect(f.offset).toBeGreaterThanOrEqual(prevEnd);
      expect(f.length).toBeGreaterThanOrEqual(FRAME_HEADER_BYTES);
      expect(f.offset + f.length).toBeLessThanOrEqual(msg.byteLength - POSTAMBLE_BYTES);
      prevEnd = f.offset + f.length;
    }
    // At least one frame was walked, so prevEnd advanced.
    expect(prevEnd).toBeGreaterThan(PREAMBLE_BYTES);
  });

  it('returns payload with the frame header and type footer stripped', async () => {
    await init();
    const msg = bufferedMessage(2);
    for (const f of frames(msg)) {
      const footer = footerBytes(f.frameTypeCode);
      expect(f.payload.byteLength).toBe(f.length - FRAME_HEADER_BYTES - footer);
      // The payload is exactly the frame's content slice.
      const start = f.offset + FRAME_HEADER_BYTES;
      expect(Array.from(f.payload)).toEqual(
        Array.from(msg.subarray(start, start + f.payload.byteLength)),
      );
      // The stripped tail always ends with the ENDF magic.
      const end = msg.subarray(f.offset + f.length - 4, f.offset + f.length);
      expect(new TextDecoder().decode(end)).toBe('ENDF');
      // ... and the frame header starts with the FR magic.
      const head = msg.subarray(f.offset, f.offset + 2);
      expect(new TextDecoder().decode(head)).toBe('FR');
    }
  });

  it('copies payload onto the JS heap (independent of the source buffer)', async () => {
    await init();
    const msg = bufferedMessage(1);
    const fs = frames(msg);
    const first = fs[0];
    const before = first.payload[0];
    // Mutating the copy must not touch the caller's buffer ...
    first.payload[0] = (before ^ 0xff) & 0xff;
    expect(msg[first.offset + FRAME_HEADER_BYTES]).toBe(before);
    // ... and the copy survives further WASM calls (memory growth safe).
    frames(bufferedMessage(64));
    expect(first.payload[0]).toBe((before ^ 0xff) & 0xff);
    expect(first.payload.buffer).not.toBe(msg.buffer);
  });

  it('exposes hasHash mirroring the per-frame HASH_PRESENT flag', async () => {
    await init();
    const hashed = frames(bufferedMessage(1));
    expect(hashed.some((f) => f.hasHash)).toBe(true);
    for (const f of hashed) {
      expect(f.hasHash).toBe((f.flags & 0b10) !== 0);
    }
    const unhashed = frames(
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) }],
        { hash: false },
      ),
    );
    expect(unhashed.every((f) => !f.hasHash)).toBe(true);
  });

  it('reports the wire version on every frame', async () => {
    await init();
    for (const f of frames(bufferedMessage(1))) {
      expect(f.version).toBe(1);
    }
  });

  it('walks a streaming message including its footer frames', async () => {
    await init();
    const types = typesOf(frames(streamedMessage()));
    expect(types[0]).toBe('HeaderMetadata');
    expect(types).toContain('NTensorFrame');
    // Streaming defers index / metadata to the footer region; the walk
    // must reach them even though `total_length` may never have been
    // back-filled.
    expect(types).toContain('FooterIndex');
  });

  it('throws on a buffer that is not a message', async () => {
    await init();
    // Too short for a preamble, empty, and long-enough-but-wrong-magic.
    expect(() => frames(new Uint8Array([1, 2, 3, 4]))).toThrow(FramingError);
    expect(() => frames(new Uint8Array(0))).toThrow(FramingError);
    expect(() => frames(new Uint8Array(64))).toThrow(/magic/);
  });

  it('throws on a message with a truncated frame chain', async () => {
    await init();
    const msg = bufferedMessage(2);
    const fs = frames(msg);
    const last = fs[fs.length - 1];
    // Cut inside the last frame: the preamble still parses, the chain
    // does not.  An eagerly-materialised array is all-or-nothing — a
    // corrupt chain must never surface as a silently shorter array.
    expect(() => frames(msg.subarray(0, last.offset + 8))).toThrow(FramingError);
  });

  it('throws on non-Uint8Array input', async () => {
    await init();
    // @ts-expect-error deliberate wrong type
    expect(() => frames('nope')).toThrow(InvalidArgumentError);
    // @ts-expect-error deliberate wrong type
    expect(() => frames('nope')).toThrow(/Uint8Array/);
  });

  it('walks each message of a multi-message buffer after scan() + slice', async () => {
    await init();
    const a = bufferedMessage(1);
    const b = bufferedMessage(3);
    const buf = new Uint8Array(a.byteLength + b.byteLength);
    buf.set(a, 0);
    buf.set(b, a.byteLength);

    const positions = scan(buf);
    expect(positions).toHaveLength(2);
    const counts = positions.map((p) => {
      const slice = buf.subarray(p.offset, p.offset + p.length);
      return frames(slice).filter((f) => f.frameType === 'NTensorFrame').length;
    });
    expect(counts).toEqual([1, 3]);
  });
});

describe('messageHeader', () => {
  it('reports the wire version and total length of a buffered message', async () => {
    await init();
    const msg = bufferedMessage(2);
    const h = messageHeader(msg);
    expect(h.version).toBe(WIRE_VERSION);
    expect(h.totalLength).toBe(msg.byteLength);
  });

  it('flags match the frames present exactly in buffered mode', async () => {
    await init();
    const msg = bufferedMessage(2);
    const h = messageHeader(msg);
    const types = new Set(typesOf(frames(msg)));
    expect(h.hasHeaderMetadata).toBe(types.has('HeaderMetadata'));
    expect(h.hasFooterMetadata).toBe(types.has('FooterMetadata'));
    expect(h.hasHeaderIndex).toBe(types.has('HeaderIndex'));
    expect(h.hasFooterIndex).toBe(types.has('FooterIndex'));
    expect(h.hasHeaderHashes).toBe(types.has('HeaderHash'));
    expect(h.hasFooterHashes).toBe(types.has('FooterHash'));
    expect(h.hasPrecederMetadata).toBe(types.has('PrecederMetadata'));
  });

  it('reports hasHashesPresent following the hash option', async () => {
    await init();
    expect(messageHeader(bufferedMessage(1)).hasHashesPresent).toBe(true);
    const unhashed = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) }],
      { hash: false },
    );
    expect(messageHeader(unhashed).hasHashesPresent).toBe(false);
  });

  it('never understates the frames present in streaming mode', async () => {
    await init();
    const msg = streamedMessage();
    const h = messageHeader(msg);
    expect(h.version).toBe(WIRE_VERSION);
    const types = new Set(typesOf(frames(msg)));
    // The streaming encoder writes the preamble before any object, so
    // its flags are advisory: only `frame present => flag set` holds.
    const implies = (present: boolean, flag: boolean): boolean => !present || flag;
    expect(implies(types.has('HeaderMetadata'), h.hasHeaderMetadata)).toBe(true);
    expect(implies(types.has('FooterMetadata'), h.hasFooterMetadata)).toBe(true);
    expect(implies(types.has('HeaderIndex'), h.hasHeaderIndex)).toBe(true);
    expect(implies(types.has('FooterIndex'), h.hasFooterIndex)).toBe(true);
    expect(implies(types.has('HeaderHash'), h.hasHeaderHashes)).toBe(true);
    expect(implies(types.has('FooterHash'), h.hasFooterHashes)).toBe(true);
    expect(implies(types.has('PrecederMetadata'), h.hasPrecederMetadata)).toBe(true);
  });

  it('throws on a buffer that is not a message', async () => {
    await init();
    expect(() => messageHeader(new Uint8Array([1, 2, 3, 4]))).toThrow(FramingError);
    expect(() => messageHeader(new Uint8Array(0))).toThrow(FramingError);
    expect(() => messageHeader(new Uint8Array(64))).toThrow(/magic/);
  });

  it('throws on non-Uint8Array input', async () => {
    await init();
    // @ts-expect-error deliberate wrong type
    expect(() => messageHeader(123)).toThrow(InvalidArgumentError);
    // @ts-expect-error deliberate wrong type
    expect(() => messageHeader(123)).toThrow(/Uint8Array/);
  });
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import {
  decode,
  decodeMetadata,
  decodeObject,
  encode,
  FramingError,
  HashMismatchError,
  init,
  InvalidArgumentError,
  ObjectError,
  scan,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('Phase 1 — decode wrapper', () => {
  it('decodeMetadata returns the correct version', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'float32'),
        data: new Float32Array([1, 2, 3]),
      },
    ]);
    const meta = decodeMetadata(msg);
    expect(meta.version).toBe(2);
  });

  it('decodeObject retrieves a single object by index', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
      {
        descriptor: makeDescriptor([3], 'uint8'),
        data: new Uint8Array([10, 20, 30]),
      },
    ]);
    const one = decodeObject(msg, 1);
    expect(one.objects).toHaveLength(1);
    expect(Array.from(one.objects[0].data() as Uint8Array)).toEqual([10, 20, 30]);
    one.close();
  });

  it('throws ObjectError for an out-of-range index', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    expect(() => decodeObject(msg, 99)).toThrow(ObjectError);
  });

  it('throws InvalidArgumentError for a non-Uint8Array buf', () => {
    // @ts-expect-error intentional
    expect(() => decode(new Float32Array([1, 2, 3]))).toThrow(InvalidArgumentError);
  });

  it('throws FramingError for garbage input', async () => {
    await init();
    const garbage = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(() => decode(garbage)).toThrow(FramingError);
  });

  it('verifyHash: true is a no-op in v3 (integrity moved to validate)', async () => {
    // v3: `verifyHash: true` on decode is retained for source
    // compatibility but is a no-op.  Frame-level integrity
    // verification moved to the validate API
    // (plans/WIRE_FORMAT.md §11).  Corruption detection lives in
    // `validate --checksum`; decode is a pure deserialisation.
    await init();
    const data = new Float32Array(1000);
    for (let i = 0; i < data.length; i++) data[i] = i;
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([1000], 'float32'), data },
    ]);
    const tampered = new Uint8Array(msg);
    tampered[500] ^= 0xff;
    // decode must NOT throw HashMismatchError — that surface moved
    // to validate.  It may still throw FramingError if the tamper
    // lands on a structural byte; otherwise it succeeds silently.
    try {
      const decoded = decode(tampered, { verifyHash: true });
      decoded.close();
    } catch (e) {
      // Structural tamper → FramingError is acceptable; hash-only
      // tamper → decode succeeds.  Either way, HashMismatchError
      // must not escape the decode path.
      expect(e).not.toBeInstanceOf(HashMismatchError);
    }
  });

  it('verifyHash: true on untampered data succeeds', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    const decoded = decode(msg, { verifyHash: true });
    expect(decoded.objects).toHaveLength(1);
    decoded.close();
  });

  it('scan handles single-message and multi-message buffers', async () => {
    await init();
    const m1 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    const m2 = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'uint8'),
        data: new Uint8Array([1, 2, 3]),
      },
    ]);
    const combined = new Uint8Array(m1.length + m2.length);
    combined.set(m1, 0);
    combined.set(m2, m1.length);

    const positions = scan(combined);
    expect(positions).toHaveLength(2);
    expect(positions[0].offset).toBe(0);
    expect(positions[0].length).toBe(m1.length);
    expect(positions[1].offset).toBe(m1.length);
    expect(positions[1].length).toBe(m2.length);
  });

  it('scan tolerates garbage between messages', async () => {
    await init();
    const m = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    const garbage = new Uint8Array([9, 9, 9, 9, 9, 9, 9, 9]);
    const combined = new Uint8Array(garbage.length + m.length + garbage.length);
    combined.set(garbage, 0);
    combined.set(m, garbage.length);
    combined.set(garbage, garbage.length + m.length);

    const positions = scan(combined);
    expect(positions).toHaveLength(1);
    expect(positions[0].offset).toBe(garbage.length);
    expect(positions[0].length).toBe(m.length);
  });

  it('objects.data() is safe after close() on another message', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'float32'),
        data: new Float32Array([10, 20, 30]),
      },
    ]);
    const decoded = decode(msg);
    const copy = decoded.objects[0].data() as Float32Array;
    decoded.close();
    // The copy survives the close because it is JS-heap owned.
    expect(Array.from(copy)).toEqual([10, 20, 30]);
  });

  it('accessing data() after close() throws', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'float32'),
        data: new Float32Array([10, 20, 30]),
      },
    ]);
    const decoded = decode(msg);
    decoded.close();
    expect(() => decoded.objects[0].data()).toThrow(/closed/);
  });

  // ── Coverage closers ────────────────────────────────────────────────

  it('decodeObject rejects fractional indices', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    expect(() => decodeObject(msg, 1.5)).toThrow(InvalidArgumentError);
  });

  it('decodeObject rejects negative indices', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    expect(() => decodeObject(msg, -1)).toThrow(InvalidArgumentError);
  });

  it('decodeObject rejects non-Uint8Array buf', () => {
    // @ts-expect-error intentional
    expect(() => decodeObject(new Float32Array([1, 2, 3]), 0)).toThrow(
      InvalidArgumentError,
    );
  });

  it('dataView() after close() throws', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'float32'),
        data: new Float32Array([1, 2, 3]),
      },
    ]);
    const decoded = decode(msg);
    decoded.close();
    expect(() => decoded.objects[0].dataView()).toThrow(/closed/);
  });

  it('decodeMetadata rejects non-Uint8Array buf', () => {
    // @ts-expect-error intentional
    expect(() => decodeMetadata([1, 2, 3])).toThrow(InvalidArgumentError);
  });

  it('scan rejects non-Uint8Array buf', () => {
    // @ts-expect-error intentional
    expect(() => scan([1, 2, 3])).toThrow(InvalidArgumentError);
  });

  it('close() is idempotent', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1], 'uint8'),
        data: new Uint8Array([42]),
      },
    ]);
    const decoded = decode(msg);
    expect(() => decoded.close()).not.toThrow();
    expect(() => decoded.close()).not.toThrow();
  });
});

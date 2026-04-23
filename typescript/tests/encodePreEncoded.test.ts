// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `encodePreEncoded` tests.  Core contract: caller-supplied bytes are
 * written verbatim into the wire message, and a round-trip through
 * `decode` recovers both the original bytes (for no-pipeline
 * descriptors) and the metadata structure.
 */

import { describe, expect, it } from 'vitest';
import {
  computeHash,
  decode,
  encodePreEncoded,
  InvalidArgumentError,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.1 — encodePreEncoded', () => {
  initOnce();

  it('round-trips uncompressed float32 bytes', () => {
    const values = new Float32Array([10, 20, 30, 40]);
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const msg = encodePreEncoded(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data: bytes },
    ]);

    const decoded = decode(msg);
    try {
      expect(decoded.objects).toHaveLength(1);
      expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([10, 20, 30, 40]);
    } finally {
      decoded.close();
    }
  });

  it('encodePreEncoded round-trips cleanly (v3: descriptor.hash is undefined)', () => {
    // v3: the per-object hash lives in the frame footer's inline
    // slot, not on the CBOR descriptor (plans/WIRE_FORMAT.md §2.4).
    // `descriptor.hash` on the decoded output is always undefined.
    // `computeHash` still returns a stable 16-char hex digest that
    // callers can compare against the preamble-level validation.
    const values = new Float64Array([1.25, 2.5, 3.75, 4.0]);
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const msg = encodePreEncoded(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float64'), data: bytes },
    ]);
    const decoded = decode(msg);
    try {
      expect(decoded.objects[0].descriptor.hash).toBeUndefined();
      // computeHash remains a stable independent helper:
      expect(computeHash(bytes)).toHaveLength(16);
    } finally {
      decoded.close();
    }
  });

  it('disables hashing when hash: false', () => {
    const values = new Uint8Array([1, 2, 3, 4]);
    const msg = encodePreEncoded(
      defaultMeta(),
      [{ descriptor: makeDescriptor([4], 'uint8'), data: values }],
      { hash: false },
    );
    const decoded = decode(msg);
    try {
      expect(decoded.objects[0].descriptor.hash).toBeUndefined();
    } finally {
      decoded.close();
    }
  });

  it('handles multiple pre-encoded objects in one message', () => {
    const a = new Float32Array([1, 2]);
    const b = new BigInt64Array([42n, 43n, 44n]);
    const msg = encodePreEncoded(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Uint8Array(a.buffer, a.byteOffset, a.byteLength),
      },
      {
        descriptor: makeDescriptor([3], 'int64'),
        data: new Uint8Array(b.buffer, b.byteOffset, b.byteLength),
      },
    ]);
    const decoded = decode(msg);
    try {
      expect(decoded.objects).toHaveLength(2);
      expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1, 2]);
      expect(Array.from(decoded.objects[1].data() as BigInt64Array)).toEqual([42n, 43n, 44n]);
    } finally {
      decoded.close();
    }
  });

  it('rejects non-Uint8Array data', () => {
    expect(() =>
      encodePreEncoded(defaultMeta(), [
        // @ts-expect-error intentional: data must be Uint8Array
        { descriptor: makeDescriptor([1], 'float32'), data: new Float32Array([1]) },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects client-written _reserved_', () => {
    const bad = {
_reserved_: { foo: 'bar' },
    } as Parameters<typeof encodePreEncoded>[0];
    expect(() =>
      encodePreEncoded(bad, [
        {
          descriptor: makeDescriptor([1], 'uint8'),
          data: new Uint8Array([1]),
        },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects bad descriptor structure', () => {
    expect(() =>
      encodePreEncoded(defaultMeta(), [
        // @ts-expect-error intentional: dtype is required
        { descriptor: { type: 'ntensor', shape: [1] }, data: new Uint8Array([1]) },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects non-array objects parameter', () => {
    expect(() =>
      // @ts-expect-error intentional: objects must be an array
      encodePreEncoded(defaultMeta(), 'not an array'),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects null entry in objects', () => {
    expect(() =>
      encodePreEncoded(defaultMeta(), [
        // @ts-expect-error intentional
        null,
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects metadata.base when not an array', () => {
    const bad = {
base: 'nope',
    } as unknown as Parameters<typeof encodePreEncoded>[0];
    expect(() =>
      encodePreEncoded(bad, [
        {
          descriptor: makeDescriptor([1], 'uint8'),
          data: new Uint8Array([1]),
        },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects metadata.base[i] when not a plain object', () => {
    const bad = {
base: [null],
    } as unknown as Parameters<typeof encodePreEncoded>[0];
    expect(() =>
      encodePreEncoded(bad, [
        {
          descriptor: makeDescriptor([1], 'uint8'),
          data: new Uint8Array([1]),
        },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects client-written _reserved_ in metadata.base[i]', () => {
    const bad = {
base: [{ _reserved_: { foo: 'bar' } }],
    } as unknown as Parameters<typeof encodePreEncoded>[0];
    expect(() =>
      encodePreEncoded(bad, [
        {
          descriptor: makeDescriptor([1], 'uint8'),
          data: new Uint8Array([1]),
        },
      ]),
    ).toThrow(InvalidArgumentError);
  });
});

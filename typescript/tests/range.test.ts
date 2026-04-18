// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `decodeRange` — partial sub-tensor decode tests.  Covers the happy
 * path (single range, multiple ranges, join mode), dtype fidelity, and
 * the error contract around shape/dtype/filter limitations.
 */

import { describe, expect, it } from 'vitest';
import {
  decode,
  decodeRange,
  encode,
  EncodingError,
  InvalidArgumentError,
  ObjectError,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';
import { initOnce } from './helpers.js';

describe('Scope C.1 — decodeRange', () => {
  initOnce();

  it('decodes a single range in element units', async () => {
    const values = new Float32Array(16);
    for (let i = 0; i < values.length; i++) values[i] = i;
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([16], 'float32'), data: values },
    ]);

    const { descriptor, parts } = decodeRange(msg, 0, [[4, 4]]);
    expect(descriptor.dtype).toBe('float32');
    expect(parts).toHaveLength(1);
    const f32 = parts[0] as Float32Array;
    expect(f32).toBeInstanceOf(Float32Array);
    expect(Array.from(f32)).toEqual([4, 5, 6, 7]);
  });

  it('returns one part per requested range in split mode', async () => {
    const values = new Float64Array(32).map((_, i) => i * 0.5);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([32], 'float64'), data: values },
    ]);

    const { parts } = decodeRange(msg, 0, [
      [0, 4],
      [10, 2],
      [28, 4],
    ]);
    expect(parts).toHaveLength(3);
    expect(Array.from(parts[0] as Float64Array)).toEqual([0, 0.5, 1, 1.5]);
    expect(Array.from(parts[1] as Float64Array)).toEqual([5, 5.5]);
    expect(Array.from(parts[2] as Float64Array)).toEqual([14, 14.5, 15, 15.5]);
  });

  it('concatenates ranges with join: true', async () => {
    const values = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([10], 'int32'), data: values },
    ]);

    const { parts } = decodeRange(
      msg,
      0,
      [
        [0, 2],
        [5, 2],
        [9, 1],
      ],
      { join: true },
    );
    expect(parts).toHaveLength(1);
    expect(Array.from(parts[0] as Int32Array)).toEqual([0, 1, 5, 6, 9]);
  });

  it('preserves dtype typing (int64 → BigInt64Array)', async () => {
    const values = new BigInt64Array([-5n, -1n, 0n, 1n, 10n, 42n]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([6], 'int64'), data: values },
    ]);
    const { parts } = decodeRange(msg, 0, [[1, 3]]);
    expect(parts[0]).toBeInstanceOf(BigInt64Array);
    expect(Array.from(parts[0] as BigInt64Array)).toEqual([-1n, 0n, 1n]);
  });

  it('accepts bigint offsets and counts', async () => {
    const values = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([8], 'uint8'), data: values },
    ]);
    const { parts } = decodeRange(msg, 0, [[2n, 3n]]);
    expect(Array.from(parts[0] as Uint8Array)).toEqual([30, 40, 50]);
  });

  it('returns empty parts[] for an empty ranges array', async () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data: values },
    ]);
    const { parts } = decodeRange(msg, 0, []);
    expect(parts).toHaveLength(0);
  });

  it('rejects out-of-range object index with ObjectError', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    expect(() => decodeRange(msg, 5, [[0, 1]])).toThrow(ObjectError);
  });

  it('rejects a filter=shuffle object with EncodingError', async () => {
    // Manually craft a descriptor with shuffle + an explicit element size
    // and round-trip through encode/decode_range.
    const values = new Float32Array([1, 2, 3, 4]);
    const desc = {
      ...makeDescriptor([4], 'float32'),
      filter: 'shuffle',
      shuffle_element_size: 4,
    } as const;
    const msg = encode(defaultMeta(), [{ descriptor: desc, data: values }]);
    expect(() => decodeRange(msg, 0, [[0, 2]])).toThrow(EncodingError);
  });

  it('rejects bitmask dtype with EncodingError', async () => {
    // ceil(16 / 8) = 2 bytes of packed bits
    const bits = new Uint8Array([0b11110000, 0b00001111]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([16], 'bitmask'), data: bits },
    ]);
    expect(() => decodeRange(msg, 0, [[0, 8]])).toThrow(EncodingError);
  });

  it('rejects malformed ranges (not a pair)', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    // @ts-expect-error intentional bad input
    expect(() => decodeRange(msg, 0, [[1, 2, 3]])).toThrow(InvalidArgumentError);
  });

  it('rejects negative offsets / counts', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    expect(() => decodeRange(msg, 0, [[-1, 2]])).toThrow(InvalidArgumentError);
    expect(() => decodeRange(msg, 0, [[0, -1]])).toThrow(InvalidArgumentError);
    expect(() => decodeRange(msg, 0, [[1.5, 2]])).toThrow(InvalidArgumentError);
  });

  it('rejects non-Uint8Array buf', () => {
    expect(() =>
      // @ts-expect-error intentional
      decodeRange([1, 2, 3], 0, [[0, 1]]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects non-array ranges argument', () => {
    expect(() =>
      // @ts-expect-error intentional: ranges must be an array
      decodeRange(new Uint8Array(100), 0, 'not an array'),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects range pair where the offset is neither number nor bigint', () => {
    expect(() =>
      // @ts-expect-error intentional: strings are not accepted
      decodeRange(new Uint8Array(100), 0, [['0', '1']]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects non-integer objectIndex', () => {
    expect(() =>
      decodeRange(new Uint8Array(100), 1.5, [[0, 1]]),
    ).toThrow(InvalidArgumentError);
    expect(() =>
      decodeRange(new Uint8Array(100), -1, [[0, 1]]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects objectIndex above u32 range', () => {
    // WASM's `usize` is u32 on wasm32 — values beyond would be
    // silently truncated.  We reject at the TS boundary.
    expect(() =>
      decodeRange(new Uint8Array(100), 2 ** 32, [[0, 1]]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects range values above u64 range', () => {
    expect(() =>
      decodeRange(new Uint8Array(100), 0, [[2n ** 65n, 1n]]),
    ).toThrow(InvalidArgumentError);
    expect(() =>
      decodeRange(new Uint8Array(100), 0, [[0n, -1n]]),
    ).toThrow(InvalidArgumentError);
  });

  it('matches a manual slice of the full decode', async () => {
    const n = 100;
    const data = new Float32Array(n).map((_, i) => Math.sin(i * 0.1));
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([n], 'float32'), data },
    ]);

    const full = decode(msg);
    try {
      const fullF32 = full.objects[0].data() as Float32Array;
      const { parts } = decodeRange(msg, 0, [[25, 10]]);
      const sliceF32 = parts[0] as Float32Array;
      expect(Array.from(sliceF32)).toEqual(Array.from(fullF32.subarray(25, 35)));
    } finally {
      full.close();
    }
  });

  it('verifyHash: true passes on an unmodified message', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    const { parts } = decodeRange(msg, 0, [[0, 4]], { verifyHash: true });
    expect(Array.from(parts[0] as Float32Array)).toEqual([1, 2, 3, 4]);
  });
});

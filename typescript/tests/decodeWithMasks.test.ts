// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for the advanced `decodeWithMasks` binding (Wave D / D-TS-MASKS).
// See `plans/INTERFACE_SYMMETRY.md` §5.4 / §8 and
// docs/src/guide/nan-inf-handling.md for the user contract.  Mirrors the
// Python `test_decode_with_masks_*` suite in
// python/tests/test_allow_nan_inf.py.

import { describe, expect, it } from 'vitest';
import { decodeWithMasks, encode, init } from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('decodeWithMasks', () => {
  it('returns 0.0-substituted payload plus per-kind masks', async () => {
    await init();
    // 128 finite values with a NaN, a +Inf and a -Inf punched in.
    const values = new Float64Array(128);
    for (let i = 0; i < 128; i++) values[i] = i;
    values[10] = NaN;
    values[50] = Infinity;
    values[100] = -Infinity;

    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([128], 'float64'), data: values }],
      { allowNan: true, allowInf: true, smallMaskThresholdBytes: 0 },
    );

    const decoded = decodeWithMasks(msg);
    try {
      expect(decoded.objects).toHaveLength(1);
      const obj = decoded.objects[0];

      // Payload is 0.0-substituted (restoration NOT applied).
      const payload = obj.data() as Float64Array;
      expect(payload[10]).toBe(0);
      expect(payload[50]).toBe(0);
      expect(payload[100]).toBe(0);
      // Finite values survive untouched.
      expect(payload[11]).toBe(11);
      expect(payload[0]).toBe(0); // genuine zero, not masked

      // Masks are byte-per-element Uint8Arrays of length n_elements.
      const { nan, posInf, negInf } = obj.masks;
      expect(nan).toBeInstanceOf(Uint8Array);
      expect(posInf).toBeInstanceOf(Uint8Array);
      expect(negInf).toBeInstanceOf(Uint8Array);
      expect(nan?.length).toBe(128);
      expect(posInf?.length).toBe(128);
      expect(negInf?.length).toBe(128);

      // Exactly one bit set per kind, at the right index.
      expect(nan?.[10]).toBe(1);
      expect(posInf?.[50]).toBe(1);
      expect(negInf?.[100]).toBe(1);
      expect(sum(nan!)).toBe(1);
      expect(sum(posInf!)).toBe(1);
      expect(sum(negInf!)).toBe(1);

      // Kinds never overlap at the same index.
      for (let i = 0; i < 128; i++) {
        expect((nan![i] ?? 0) + (posInf![i] ?? 0) + (negInf![i] ?? 0)).toBeLessThanOrEqual(1);
      }
    } finally {
      decoded.close();
    }
  });

  it('returns all-null masks when the frame carried no mask companion', async () => {
    await init();
    const values = new Float32Array([1, 2, 3, 4]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data: values },
    ]);

    const decoded = decodeWithMasks(msg);
    try {
      const obj = decoded.objects[0];
      expect(obj.masks.nan).toBeNull();
      expect(obj.masks.posInf).toBeNull();
      expect(obj.masks.negInf).toBeNull();
      // Payload round-trips exactly.
      expect(Array.from(obj.data() as Float32Array)).toEqual([1, 2, 3, 4]);
    } finally {
      decoded.close();
    }
  });

  it('all three kinds produce independent masks', async () => {
    await init();
    const values = new Float64Array(64).fill(1);
    values[0] = NaN;
    values[1] = NaN;
    values[2] = Infinity;
    values[3] = -Infinity;
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([64], 'float64'), data: values }],
      { allowNan: true, allowInf: true, smallMaskThresholdBytes: 0 },
    );
    const decoded = decodeWithMasks(msg);
    try {
      const { nan, posInf, negInf } = decoded.objects[0].masks;
      expect(sum(nan!)).toBe(2);
      expect(nan?.[0]).toBe(1);
      expect(nan?.[1]).toBe(1);
      expect(sum(posInf!)).toBe(1);
      expect(posInf?.[2]).toBe(1);
      expect(sum(negInf!)).toBe(1);
      expect(negInf?.[3]).toBe(1);
    } finally {
      decoded.close();
    }
  });

  it('masks survive after close() (JS-heap copies)', async () => {
    await init();
    const values = new Float64Array([NaN, 1, 2]);
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([3], 'float64'), data: values }],
      { allowNan: true, smallMaskThresholdBytes: 0 },
    );
    const decoded = decodeWithMasks(msg);
    const nan = decoded.objects[0].masks.nan;
    decoded.close();
    // The mask is a JS-heap copy, still readable after close().
    expect(nan?.[0]).toBe(1);
    expect(sum(nan!)).toBe(1);
  });

  it('honours nativeByteOrder=false (wire byte order retained)', async () => {
    await init();
    // Big-endian descriptor; with nativeByteOrder=false the payload
    // bytes stay big-endian on this little-endian host.
    const desc = { ...makeDescriptor([2], 'int16'), byte_order: 'big' as const };
    const data = new Uint8Array([0x12, 0x34, 0x56, 0x78]);
    const msg = encode(defaultMeta(), [{ descriptor: desc, data }]);

    const decoded = decodeWithMasks(msg, { nativeByteOrder: false });
    try {
      const raw = decoded.objects[0].dataView() as Int16Array;
      // Bytes untouched → still the big-endian byte pattern.
      const bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
      expect(Array.from(bytes)).toEqual([0x12, 0x34, 0x56, 0x78]);
    } finally {
      decoded.close();
    }
  });

  it('throws on non-Uint8Array input', async () => {
    await init();
    // @ts-expect-error deliberate wrong type
    expect(() => decodeWithMasks([1, 2, 3])).toThrow(/Uint8Array/);
  });

  it('accessing payload after close() throws', async () => {
    await init();
    const values = new Float64Array([1, 2, 3]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([3], 'float64'), data: values },
    ]);
    const decoded = decodeWithMasks(msg);
    const obj = decoded.objects[0];
    decoded.close();
    expect(() => obj.data()).toThrow(/closed/);
  });

  it('works across multiple objects with mixed masks', async () => {
    await init();
    const a = new Float64Array([NaN, 1]);
    const b = new Float32Array([2, 3, 4]);
    const msg = encode(
      defaultMeta(),
      [
        { descriptor: makeDescriptor([2], 'float64'), data: a },
        { descriptor: makeDescriptor([3], 'float32'), data: b },
      ],
      { allowNan: true, smallMaskThresholdBytes: 0 },
    );
    const decoded = decodeWithMasks(msg);
    try {
      expect(decoded.objects).toHaveLength(2);
      expect(decoded.objects[0].masks.nan?.[0]).toBe(1);
      // Second object had no NaN → no mask companion.
      expect(decoded.objects[1].masks.nan).toBeNull();
      expect(Array.from(decoded.objects[1].data() as Float32Array)).toEqual([2, 3, 4]);
    } finally {
      decoded.close();
    }
  });
});

function sum(a: Uint8Array): number {
  let s = 0;
  for (const v of a) s += v;
  return s;
}

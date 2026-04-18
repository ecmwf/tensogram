// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `Bfloat16Array` — behaviour tests.  Same shape as the float16 suite,
 * adapted for the brain-float layout (1-8-7).  The conversion is
 * close to a byte copy so the tolerance is tighter than float16 in
 * ulp terms, but the dynamic range is effectively float32 (same
 * 8-bit exponent).
 */

import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  bfloat16FromBytes,
  Bfloat16Array,
  decode,
  encode,
} from '../src/index.js';
// The bit-conversion helpers are `@internal` — pulled from the concrete
// module so the public barrel stays lean.
import { bfloat16BitsToFloat, floatToBfloat16Bits } from '../src/bfloat16.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.2 — Bfloat16 bit conversions', () => {
  it('round-trips ±0 with sign', () => {
    expect(bfloat16BitsToFloat(0x0000)).toBe(0);
    expect(Object.is(bfloat16BitsToFloat(0x8000), -0)).toBe(true);
    expect(floatToBfloat16Bits(0)).toBe(0x0000);
    expect(floatToBfloat16Bits(-0)).toBe(0x8000);
  });

  it('round-trips ±Inf', () => {
    expect(bfloat16BitsToFloat(0x7f80)).toBe(Infinity);
    expect(bfloat16BitsToFloat(0xff80)).toBe(-Infinity);
    expect(floatToBfloat16Bits(Infinity)).toBe(0x7f80);
    expect(floatToBfloat16Bits(-Infinity)).toBe(0xff80);
  });

  it('preserves NaN', () => {
    expect(Number.isNaN(bfloat16BitsToFloat(0x7fc0))).toBe(true);
    const bits = floatToBfloat16Bits(NaN);
    expect((bits >>> 7) & 0xff).toBe(0xff); // exponent all-ones
    expect(bits & 0x7f).not.toBe(0); // mantissa ≠ 0
  });

  it('round-trips exactly for integers representable in 8 mantissa bits', () => {
    for (const v of [1, -1, 2, -2, 4, 100, -100]) {
      const back = bfloat16BitsToFloat(floatToBfloat16Bits(v));
      expect(back).toBe(v);
    }
  });

  it('narrows via round-to-nearest-even', () => {
    // 1.5 is exactly representable (mantissa = 1.1₂ → 0x3fc0 in bfloat16).
    expect(floatToBfloat16Bits(1.5)).toBe(0x3fc0);
    expect(bfloat16BitsToFloat(0x3fc0)).toBe(1.5);
  });

  it('property: random float32 survives round-trip within bfloat16 ulp', () => {
    fc.assert(
      fc.property(
        fc.float({ min: Math.fround(-1e4), max: Math.fround(1e4), noNaN: true }),
        (v) => {
          const back = bfloat16BitsToFloat(floatToBfloat16Bits(v));
          // Bfloat16 ulp ≈ 2^-7 × magnitude.
          const tol = Math.max(Math.abs(v) * 0.01, 1e-3);
          expect(Math.abs(back - v)).toBeLessThan(tol);
        },
      ),
      { numRuns: 200 },
    );
  });
});

describe('Scope C.2 — Bfloat16Array array API', () => {
  it('constructs from length', () => {
    const a = new Bfloat16Array(4);
    expect(a.length).toBe(4);
    expect(a.byteLength).toBe(8);
    expect(Array.from(a)).toEqual([0, 0, 0, 0]);
  });

  it('constructs from array-like with narrowing', () => {
    const a = new Bfloat16Array([1, 2, 3, 4]);
    expect(Array.from(a)).toEqual([1, 2, 3, 4]);
  });

  it('exposes raw bits via .bits', () => {
    const a = new Bfloat16Array([1, 2, 3]);
    expect(a.bits).toBeInstanceOf(Uint16Array);
    expect(a.bits.length).toBe(3);
  });

  it('set() narrows per-element, accepts Bfloat16Array verbatim', () => {
    const src = new Bfloat16Array([7, 8, 9]);
    const dst = new Bfloat16Array(3);
    dst.set(src);
    expect(Array.from(dst)).toEqual([7, 8, 9]);
    dst.set([1.5, 2.5, 3.5], 0);
    expect(Array.from(dst)).toEqual([1.5, 2.5, 3.5]);
  });

  it('slice / subarray semantics mirror TypedArray', () => {
    const a = new Bfloat16Array([1, 2, 3, 4]);
    expect(Array.from(a.slice(1, 3))).toEqual([2, 3]);
    expect(Array.from(a.subarray(1, 3))).toEqual([2, 3]);
  });

  it('toFloat32Array widens precisely', () => {
    const a = new Bfloat16Array([1.5, 2.5]);
    expect(Array.from(a.toFloat32Array())).toEqual([1.5, 2.5]);
  });

  it('fill / iterate', () => {
    const a = new Bfloat16Array(3);
    a.fill(42);
    expect(Array.from(a)).toEqual([42, 42, 42]);
  });
});

describe('Scope C.2 — bfloat16 end-to-end', () => {
  initOnce();

  it('encode → decode preserves bit-exact values', () => {
    const values = new Bfloat16Array([1, -1, 0, 2.5, -3.5]);
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([values.length], 'bfloat16'),
        data: new Uint8Array(values.bits.buffer, values.bits.byteOffset, values.bits.byteLength),
      },
    ]);
    const decoded = decode(msg);
    try {
      const out = decoded.objects[0].data() as Bfloat16Array;
      expect(out).toBeInstanceOf(Bfloat16Array);
      expect(Array.from(out.bits)).toEqual(Array.from(values.bits));
    } finally {
      decoded.close();
    }
  });

  it('bfloat16FromBytes is zero-copy for aligned input', () => {
    const aligned = new Uint8Array(8);
    const view = bfloat16FromBytes(aligned);
    expect(view.bits.buffer).toBe(aligned.buffer);
  });

  it('bfloat16FromBytes rejects odd byte lengths', () => {
    expect(() => bfloat16FromBytes(new Uint8Array(3))).toThrow(RangeError);
    expect(() => bfloat16FromBytes(new Uint8Array(1))).toThrow(RangeError);
  });
});

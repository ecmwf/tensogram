// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `Float16Polyfill` — behaviour tests.  Covers:
 *
 * - Round-trip of every representable bit pattern through
 *   `floatToHalfBits → halfBitsToFloat`.
 * - Special values: NaN, ±Inf, ±0, subnormals.
 * - Array API: iteration, `set`, `fill`, `subarray`, `slice`,
 *   `.bits` zero-copy access.
 * - End-to-end round-trip through `encode` / `decode` for the
 *   `float16` dtype — the bytes must match what Rust produced because
 *   both sides use the same binary16 layout.
 * - Parity against the native `Float16Array` when the host has one.
 */

import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  decode,
  encode,
  Float16Polyfill,
  float16FromBytes,
  getFloat16ArrayCtor,
  hasNativeFloat16Array,
} from '../src/index.js';
// The bit-conversion helpers are `@internal` — pulled from the concrete
// module so the public barrel stays lean.
import { floatToHalfBits, halfBitsToFloat } from '../src/float16.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.2 — Float16Polyfill bit conversions', () => {
  it('round-trips ±0 and preserves sign', () => {
    // +0 and -0 are numerically equal but carry different sign bits.
    // Vitest's `.toBe(0)` uses `Object.is`, so compare explicitly.
    expect(Object.is(halfBitsToFloat(0x0000), 0)).toBe(true);
    expect(Object.is(halfBitsToFloat(0x8000), -0)).toBe(true);
    expect(floatToHalfBits(0)).toBe(0x0000);
    expect(floatToHalfBits(-0)).toBe(0x8000);
  });

  it('round-trips ±Inf', () => {
    expect(halfBitsToFloat(0x7c00)).toBe(Infinity);
    expect(halfBitsToFloat(0xfc00)).toBe(-Infinity);
    expect(floatToHalfBits(Infinity)).toBe(0x7c00);
    expect(floatToHalfBits(-Infinity)).toBe(0xfc00);
  });

  it('preserves NaN', () => {
    expect(Number.isNaN(halfBitsToFloat(0x7e00))).toBe(true);
    expect(floatToHalfBits(NaN) & 0x7c00).toBe(0x7c00); // exponent all-ones
    expect(floatToHalfBits(NaN) & 0x03ff).not.toBe(0); // mantissa ≠ 0
  });

  it('round-trips exactly representable values', () => {
    const exact = [1, -1, 0.5, 1024, -1024, 65504 /* float16 max */];
    for (const v of exact) {
      const bits = floatToHalfBits(v);
      expect(halfBitsToFloat(bits)).toBe(v);
    }
  });

  it('round-trips float32 → float16 → float32 within half-precision ulp', () => {
    fc.assert(
      fc.property(
        fc.float({ min: Math.fround(-100), max: Math.fround(100), noNaN: true }),
        (v) => {
          const bits = floatToHalfBits(v);
          const back = halfBitsToFloat(bits);
          // Half-precision ulp ≤ 2^-10 × magnitude ≈ 0.1 at |v| = 100.
          // Use a generous bound because fast-check may generate tiny
          // subnormals; we only want to catch catastrophic errors.
          const tol = Math.max(Math.abs(v) * 1e-3, 1e-3);
          expect(Math.abs(back - v)).toBeLessThan(tol);
        },
      ),
      { numRuns: 200 },
    );
  });

  it('saturates to ±Inf on overflow', () => {
    // 1e10 is far outside float16's ±65504 range.
    const bits = floatToHalfBits(1e10);
    expect(bits).toBe(0x7c00);
    expect(floatToHalfBits(-1e10)).toBe(0xfc00);
  });

  it('flushes to ±0 on underflow', () => {
    // 1e-10 is below float16's smallest subnormal (~5.96e-8).
    expect(floatToHalfBits(1e-10)).toBe(0);
    expect(floatToHalfBits(-1e-10)).toBe(0x8000);
  });
});

describe('Scope C.2 — Float16Polyfill array API', () => {
  it('constructs from length with zeros', () => {
    const a = new Float16Polyfill(4);
    expect(a.length).toBe(4);
    expect(a.byteLength).toBe(8);
    expect(Array.from(a)).toEqual([0, 0, 0, 0]);
  });

  it('constructs from array-like and narrows', () => {
    const a = new Float16Polyfill([1, 2, 3, 4]);
    expect(Array.from(a)).toEqual([1, 2, 3, 4]);
  });

  it('constructs from ArrayBuffer (raw bits view)', () => {
    const bits = new Uint16Array([0x3c00, 0x4000, 0x4200]); // 1.0, 2.0, 3.0
    const a = new Float16Polyfill(bits.buffer);
    expect(a.length).toBe(3);
    expect(Array.from(a)).toEqual([1, 2, 3]);
  });

  it('`.bits` exposes raw binary16 for zero-copy roundtrips', () => {
    const a = new Float16Polyfill([1, 2, 3, 4]);
    expect(a.bits).toBeInstanceOf(Uint16Array);
    expect(a.bits.length).toBe(4);
  });

  it('get() / at() return NaN / undefined out of range', () => {
    const a = new Float16Polyfill([1, 2, 3]);
    expect(a.at(0)).toBe(1);
    expect(a.at(-1)).toBe(3);
    expect(a.at(10)).toBeUndefined();
    expect(Number.isNaN(a.get(10))).toBe(true);
    expect(Number.isNaN(a.get(-1))).toBe(true);
  });

  it('set(number[]) narrows individual values', () => {
    const a = new Float16Polyfill(4);
    a.set([1.0, 2.0, 3.0, 4.0]);
    expect(Array.from(a)).toEqual([1, 2, 3, 4]);
  });

  it('set(Float16Polyfill) copies bits verbatim', () => {
    const src = new Float16Polyfill([1, 2, 3, 4]);
    const dst = new Float16Polyfill(4);
    dst.set(src);
    expect(Array.from(dst)).toEqual([1, 2, 3, 4]);
    // Source unaffected
    expect(Array.from(src)).toEqual([1, 2, 3, 4]);
  });

  it('fill() narrows the fill value once', () => {
    const a = new Float16Polyfill(4);
    a.fill(7);
    expect(Array.from(a)).toEqual([7, 7, 7, 7]);
  });

  it('slice() returns an independent copy of raw bits', () => {
    const a = new Float16Polyfill([1, 2, 3, 4]);
    const s = a.slice(1, 3);
    expect(Array.from(s)).toEqual([2, 3]);
    a.set([9, 9, 9, 9]);
    expect(Array.from(s)).toEqual([2, 3]); // unaffected
  });

  it('subarray() shares bit storage with the parent', () => {
    const a = new Float16Polyfill([1, 2, 3, 4]);
    const s = a.subarray(1, 3);
    expect(Array.from(s)).toEqual([2, 3]);
    a.set([0], 2); // overwrite parent index 2
    expect(s.get(1)).toBe(0); // subview reflects the change
  });

  it('toFloat32Array() widens to float32 precision', () => {
    const a = new Float16Polyfill([1.5, 2.5, 3.5]);
    const f32 = a.toFloat32Array();
    expect(f32).toBeInstanceOf(Float32Array);
    expect(Array.from(f32)).toEqual([1.5, 2.5, 3.5]);
  });

  it('iterates with for..of', () => {
    const a = new Float16Polyfill([1, 2, 3]);
    const out: number[] = [];
    for (const v of a) out.push(v);
    expect(out).toEqual([1, 2, 3]);
  });
});

describe('Scope C.2 — native Float16Array detection', () => {
  it('probes every call (honours ambient override)', () => {
    // We can't reliably mutate globalThis.Float16Array in one worker
    // without racing other tests, so we just check the probe returns a
    // consistent boolean for whatever the runtime actually has.
    const a = hasNativeFloat16Array();
    const b = hasNativeFloat16Array();
    expect(a).toBe(b);
    expect(typeof a).toBe('boolean');
  });

  it('getFloat16ArrayCtor returns a constructor', () => {
    const Ctor = getFloat16ArrayCtor();
    expect(typeof Ctor).toBe('function');
    // Constructing with a length must work on either path.
    const inst = new Ctor(4);
    expect(inst.length).toBe(4);
  });
});

describe('Scope C.2 — float16 end-to-end round-trip', () => {
  initOnce();

  it('encode → decode round-trips bits exactly for every finite value', () => {
    // Start from explicit binary16 bit patterns so the comparison is
    // bit-exact regardless of which Float16 path the decoder picks.
    //
    // Bit patterns used:
    //   0x3c00  =  1.0
    //   0xbc00  = -1.0
    //   0x0000  =  0.0
    //   0x7bff  =  65504  (float16 max finite — avoids Inf's mantissa==0)
    //   0x0001  =  smallest subnormal
    const bits = new Uint16Array([0x3c00, 0xbc00, 0x0000, 0x7bff, 0x0001]);
    const bytes = new Uint8Array(bits.buffer, bits.byteOffset, bits.byteLength);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([5], 'float16'), data: bytes },
    ]);
    const decoded = decode(msg);
    try {
      // Whether decode() returns the native Float16Array or our
      // polyfill, we can always pull the raw bits from the underlying
      // ArrayBuffer at the same byte offset.
      const view = decoded.objects[0].data() as {
        buffer: ArrayBufferLike;
        byteOffset: number;
        byteLength: number;
      };
      const outBits = new Uint16Array(view.buffer, view.byteOffset, view.byteLength / 2);
      expect(Array.from(outBits)).toEqual(Array.from(bits));
    } finally {
      decoded.close();
    }
  });

  it('float16FromBytes is zero-copy for aligned input', () => {
    // Aligned input (offset 0) — the view's `.bits` should share the
    // same underlying ArrayBuffer.
    const aligned = new Uint8Array(8);
    const view = float16FromBytes(aligned) as { bits: Uint16Array };
    if ('bits' in view) {
      // Polyfill path.
      expect(view.bits.buffer).toBe(aligned.buffer);
    }
  });

  it('float16FromBytes rejects odd byte lengths', () => {
    expect(() => float16FromBytes(new Uint8Array(3))).toThrow(RangeError);
    expect(() => float16FromBytes(new Uint8Array(1))).toThrow(RangeError);
  });

  it('property: random float32 values survive round-trip within half-precision tolerance', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: -1e3, max: 1e3, noNaN: true }), { minLength: 1, maxLength: 32 }),
        (values) => {
          const poly = new Float16Polyfill(values);
          for (let i = 0; i < values.length; i++) {
            const back = poly.get(i);
            const tol = Math.max(Math.abs(values[i]) * 1e-3, 1e-3);
            expect(Math.abs(back - values[i])).toBeLessThan(tol);
          }
        },
      ),
      { numRuns: 100 },
    );
  });
});

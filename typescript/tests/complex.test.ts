// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `ComplexArray` — behaviour tests for the interleaved-storage view.
 * Covers: field accessors, iteration, out-of-range behaviour, end-to-
 * end round-trip through `encode` / `decode` for complex64 / complex128.
 */

import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  ComplexArray,
  complexFromBytes,
  decode,
  encode,
} from '../src/index.js';
// `isComplexDtype` is `@internal` — pulled from the concrete module so
// the public barrel stays lean.
import { isComplexDtype } from '../src/complex.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.2 — ComplexArray basics', () => {
  it('real/imag/get accessors unpack an interleaved Float32Array (complex64)', () => {
    const storage = new Float32Array([1, 2, 3, 4, 5, 6]);
    const c = new ComplexArray('complex64', storage);
    expect(c.length).toBe(3);
    expect(c.real(0)).toBe(1);
    expect(c.imag(0)).toBe(2);
    expect(c.real(1)).toBe(3);
    expect(c.imag(1)).toBe(4);
    expect(c.get(2)).toEqual({ re: 5, im: 6 });
  });

  it('real/imag/get accessors work for complex128 on Float64Array', () => {
    const storage = new Float64Array([0.5, 1.5, 2.5, 3.5]);
    const c = new ComplexArray('complex128', storage);
    expect(c.length).toBe(2);
    expect(c.get(0)).toEqual({ re: 0.5, im: 1.5 });
    expect(c.get(1)).toEqual({ re: 2.5, im: 3.5 });
  });

  it('out-of-range returns NaN on real/imag/get', () => {
    const c = new ComplexArray('complex64', new Float32Array([1, 2]));
    expect(Number.isNaN(c.real(-1))).toBe(true);
    expect(Number.isNaN(c.imag(2))).toBe(true);
    expect(Number.isNaN(c.get(2).re)).toBe(true);
    expect(Number.isNaN(c.get(2).im)).toBe(true);
  });

  it('set writes pair; throws on out-of-range index', () => {
    const c = new ComplexArray('complex64', new Float32Array(4));
    c.set(0, 10, 20);
    c.set(1, 30, 40);
    expect(c.get(0)).toEqual({ re: 10, im: 20 });
    expect(c.get(1)).toEqual({ re: 30, im: 40 });
    expect(() => c.set(5, 0, 0)).toThrow(RangeError);
    expect(() => c.set(-1, 0, 0)).toThrow(RangeError);
  });

  it('data is the zero-copy interleaved storage', () => {
    const storage = new Float32Array([1, 2, 3, 4]);
    const c = new ComplexArray('complex64', storage);
    expect(c.data).toBe(storage);
  });

  it('iteration yields { re, im } per pair', () => {
    const c = new ComplexArray('complex64', new Float32Array([1, 2, 3, 4]));
    const collected: Array<{ re: number; im: number }> = [];
    for (const pair of c) collected.push(pair);
    expect(collected).toEqual([
      { re: 1, im: 2 },
      { re: 3, im: 4 },
    ]);
  });

  it('toArray() returns a plain [re, im] list', () => {
    const c = new ComplexArray('complex64', new Float32Array([1, 2, 3, 4]));
    expect(c.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it('constructor validates dtype/storage pair', () => {
    expect(
      () => new ComplexArray('complex64', new Float64Array(4)),
    ).toThrow(/complex64 requires a Float32Array/);
    expect(
      () => new ComplexArray('complex128', new Float32Array(4)),
    ).toThrow(/complex128 requires a Float64Array/);
    expect(
      () =>
        // @ts-expect-error intentional bad dtype
        new ComplexArray('nope', new Float32Array(4)),
    ).toThrow();
  });

  it('constructor rejects odd-length storage', () => {
    expect(() => new ComplexArray('complex64', new Float32Array(3))).toThrow(
      /length 3 must be even/,
    );
  });

  it('isComplexDtype guard', () => {
    expect(isComplexDtype('complex64')).toBe(true);
    expect(isComplexDtype('complex128')).toBe(true);
    expect(isComplexDtype('float32')).toBe(false);
  });
});

describe('Scope C.2 — complex end-to-end round-trip', () => {
  initOnce();

  it('complex64 survives encode → decode with correct re/im interleaving', () => {
    const interleaved = new Float32Array([1, 2, 3, 4, 5, 6]);
    const bytes = new Uint8Array(interleaved.buffer, interleaved.byteOffset, interleaved.byteLength);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([3], 'complex64'), data: bytes },
    ]);
    const decoded = decode(msg);
    try {
      const c = decoded.objects[0].data() as ComplexArray;
      expect(c).toBeInstanceOf(ComplexArray);
      expect(c.length).toBe(3);
      expect(c.get(0)).toEqual({ re: 1, im: 2 });
      expect(c.get(1)).toEqual({ re: 3, im: 4 });
      expect(c.get(2)).toEqual({ re: 5, im: 6 });
    } finally {
      decoded.close();
    }
  });

  it('complex128 survives encode → decode', () => {
    const interleaved = new Float64Array([0.1, 0.2, 0.3, 0.4]);
    const bytes = new Uint8Array(interleaved.buffer, interleaved.byteOffset, interleaved.byteLength);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([2], 'complex128'), data: bytes },
    ]);
    const decoded = decode(msg);
    try {
      const c = decoded.objects[0].data() as ComplexArray;
      expect(c.dtype).toBe('complex128');
      expect(c.length).toBe(2);
      const p0 = c.get(0);
      const p1 = c.get(1);
      // Compare within a tiny epsilon — float64 round-trip is exact for
      // these constants but we accept 1e-15 to keep the assertion robust.
      expect(Math.abs(p0.re - 0.1)).toBeLessThan(1e-15);
      expect(Math.abs(p0.im - 0.2)).toBeLessThan(1e-15);
      expect(Math.abs(p1.re - 0.3)).toBeLessThan(1e-15);
      expect(Math.abs(p1.im - 0.4)).toBeLessThan(1e-15);
    } finally {
      decoded.close();
    }
  });

  it('complexFromBytes is zero-copy for aligned input', () => {
    const f32 = new Float32Array(8); // 4 complex pairs
    const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    const c = complexFromBytes('complex64', bytes);
    expect(c.data.buffer).toBe(f32.buffer);
  });

  it('complexFromBytes rejects lengths that are not a multiple of the pair size', () => {
    // complex64 pair = 8 bytes
    expect(() => complexFromBytes('complex64', new Uint8Array(12))).toThrow(RangeError);
    expect(() => complexFromBytes('complex64', new Uint8Array(4))).toThrow(RangeError);
    // complex128 pair = 16 bytes
    expect(() => complexFromBytes('complex128', new Uint8Array(24))).toThrow(RangeError);
    expect(() => complexFromBytes('complex128', new Uint8Array(8))).toThrow(RangeError);
  });

  it('property: random complex64 payloads round-trip bit-exactly', () => {
    initOnce();
    fc.assert(
      fc.property(
        fc.array(fc.float({ noNaN: true }), { minLength: 2, maxLength: 16 }).filter((a) => a.length % 2 === 0),
        (flat) => {
          const data = new Float32Array(flat);
          const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          const msg = encode(defaultMeta(), [
            {
              descriptor: makeDescriptor([data.length / 2], 'complex64'),
              data: bytes,
            },
          ]);
          const decoded = decode(msg);
          try {
            const c = decoded.objects[0].data() as ComplexArray;
            for (let i = 0; i < c.length; i++) {
              expect(c.real(i)).toBe(flat[2 * i]);
              expect(c.imag(i)).toBe(flat[2 * i + 1]);
            }
          } finally {
            decoded.close();
          }
        },
      ),
      { numRuns: 50 },
    );
  });
});

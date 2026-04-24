// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `simplePackingComputeParams` — params computation tests.  Focused on
 * (a) well-formed returns for typical inputs, (b) integration with
 * encoding-pipeline round-trips, and (c) error-boundary checks.
 */

import { describe, expect, it } from 'vitest';
import {
  decode,
  encode,
  InvalidArgumentError,
  simplePackingComputeParams,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.1 — simplePackingComputeParams', () => {
  initOnce();

  it('returns the expected snake-case fields', () => {
    const p = simplePackingComputeParams(
      new Float64Array([200, 210, 220, 230, 240]),
      16,
      0,
    );
    expect(p.sp_reference_value).toBe(200);
    expect(p.sp_bits_per_value).toBe(16);
    expect(p.sp_decimal_scale_factor).toBe(0);
    expect(typeof p.sp_binary_scale_factor).toBe('number');
  });

  it('accepts a plain number[] input', () => {
    const p = simplePackingComputeParams([0, 1, 2, 3, 4, 5], 8);
    expect(p.sp_reference_value).toBe(0);
    expect(p.sp_bits_per_value).toBe(8);
  });

  it('handles zero-bit constant field', () => {
    const p = simplePackingComputeParams([42, 42, 42], 0);
    expect(p.sp_bits_per_value).toBe(0);
    expect(p.sp_reference_value).toBe(42);
  });

  it('rejects NaN values via InvalidArgumentError', () => {
    // The TS wrapper rejects non-finite values before the WASM call
    // so callers see a clear early error (the Rust core would
    // surface an EncodingError later with the same meaning).
    expect(() =>
      simplePackingComputeParams([1, NaN, 3], 16),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects +Infinity via InvalidArgumentError', () => {
    expect(() =>
      simplePackingComputeParams([1, Infinity, 3], 16),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects -Infinity via InvalidArgumentError', () => {
    expect(() =>
      simplePackingComputeParams([1, -Infinity, 3], 16),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects out-of-range decimalScaleFactor (i32 overflow)', () => {
    expect(() =>
      simplePackingComputeParams([1, 2, 3], 16, 2_147_483_648),
    ).toThrow(InvalidArgumentError);
    expect(() =>
      simplePackingComputeParams([1, 2, 3], 16, -2_147_483_649),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects out-of-range bitsPerValue', () => {
    expect(() =>
      simplePackingComputeParams([1, 2, 3], 65),
    ).toThrow(InvalidArgumentError);
    expect(() =>
      simplePackingComputeParams([1, 2, 3], -1),
    ).toThrow(InvalidArgumentError);
    expect(() =>
      simplePackingComputeParams([1, 2, 3], 1.5),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects non-integer decimalScaleFactor', () => {
    expect(() =>
      simplePackingComputeParams([1, 2, 3], 16, 0.5),
    ).toThrow(InvalidArgumentError);
  });

  it('spreads into a descriptor to drive a lossy round-trip', async () => {
    // End-to-end: compute params on a float field, feed into a
    // simple_packing descriptor, encode / decode, and verify the
    // recovered values are within the ±half-step quantisation bound.
    const values = new Float64Array(64);
    for (let i = 0; i < values.length; i++) {
      values[i] = 273.15 + i * 0.01;
    }
    const params = simplePackingComputeParams(values, 16, 0);
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const desc = {
      ...makeDescriptor([values.length], 'float64'),
      encoding: 'simple_packing' as const,
      ...params,
    };
    const msg = encode(defaultMeta(), [{ descriptor: desc, data: bytes }]);
    const decoded = decode(msg);
    try {
      const recovered = decoded.objects[0].data() as Float64Array;
      expect(recovered.length).toBe(values.length);
      // Quantisation step upper bound: (max - min) / (2^bits - 1) ≈ 9.6e-6
      const step = (values[values.length - 1] - values[0]) / (Math.pow(2, 16) - 1);
      for (let i = 0; i < values.length; i++) {
        expect(Math.abs(recovered[i] - values[i])).toBeLessThan(step * 2);
      }
    } finally {
      decoded.close();
    }
  });
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// TypeScript parity tests for the `rejectNan` / `rejectInf` encode
// options.  Cross-references:
//   - Rust:   rust/tensogram/tests/strict_finite.rs
//   - Python: python/tests/test_strict_finite.py
//   - Memo:   plans/RESEARCH_NAN_HANDLING.md §4.1

import { describe, expect, it } from 'vitest';
import { encode, EncodingError, init } from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('strict-finite — rejectNan / rejectInf', () => {
  // ── Default behaviour is unchanged ─────────────────────────────────────

  it('default options accept NaN in float32 passthrough', async () => {
    await init();
    const data = new Float32Array([1.0, NaN, 3.0]);
    expect(() =>
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([3], 'float32'), data },
      ]),
    ).not.toThrow();
  });

  it('default options accept Inf in float64 passthrough', async () => {
    await init();
    const data = new Float64Array([1.0, Infinity, -Infinity, 4.0]);
    expect(() =>
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([4], 'float64'), data },
      ]),
    ).not.toThrow();
  });

  // ── rejectNan rejects NaN across float dtypes ─────────────────────────

  it('rejectNan rejects float32 NaN', async () => {
    await init();
    const data = new Float32Array([1.0, 2.0, NaN, 4.0]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([4], 'float32'), data }],
        { rejectNan: true },
      ),
    ).toThrow(EncodingError);
  });

  it('rejectNan rejects float64 NaN with element index in the message', async () => {
    await init();
    const data = new Float64Array([1.0, 2.0, 3.0, NaN]);
    try {
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([4], 'float64'), data }],
        { rejectNan: true },
      );
      throw new Error('expected throw');
    } catch (e) {
      expect(e).toBeInstanceOf(EncodingError);
      const msg = (e as Error).message;
      expect(msg).toMatch(/NaN/i);
      expect(msg).toMatch(/element 3/);
      expect(msg).toMatch(/float64/);
    }
  });

  it('rejectNan rejects float16 NaN via bit-level inspection', async () => {
    await init();
    // IEEE half NaN: exp=0x1F, mantissa != 0 — 0x7E00.
    const bits = new Uint16Array([0x3c00, 0x7e00, 0x4000, 0x4200]);
    expect(() =>
      encode(
        defaultMeta(),
        [
          {
            descriptor: makeDescriptor([4], 'float16'),
            data: bits,
          },
        ],
        { rejectNan: true },
      ),
    ).toThrow(EncodingError);
  });

  it('rejectNan rejects complex64 NaN in real component', async () => {
    await init();
    // 3 complex64 elements = 6 f32 slots. Real of element 1 is NaN.
    const data = new Float32Array([1.0, 2.0, NaN, 3.0, 4.0, 5.0]);
    try {
      encode(
        defaultMeta(),
        [
          {
            descriptor: makeDescriptor([3], 'complex64'),
            data,
          },
        ],
        { rejectNan: true },
      );
      throw new Error('expected throw');
    } catch (e) {
      expect(e).toBeInstanceOf(EncodingError);
      expect((e as Error).message).toMatch(/real/i);
    }
  });

  // ── rejectInf ─────────────────────────────────────────────────────────

  it('rejectInf rejects +Infinity', async () => {
    await init();
    const data = new Float32Array([1.0, Infinity, 3.0]);
    try {
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([3], 'float32'), data }],
        { rejectInf: true },
      );
      throw new Error('expected throw');
    } catch (e) {
      expect(e).toBeInstanceOf(EncodingError);
      expect((e as Error).message).toMatch(/inf/i);
    }
  });

  it('rejectInf rejects -Infinity', async () => {
    await init();
    const data = new Float64Array([1.0, -Infinity]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([2], 'float64'), data }],
        { rejectInf: true },
      ),
    ).toThrow(EncodingError);
  });

  // ── Orthogonality ─────────────────────────────────────────────────────

  it('rejectInf does not reject NaN', async () => {
    await init();
    const data = new Float32Array([1.0, NaN]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([2], 'float32'), data }],
        { rejectInf: true },
      ),
    ).not.toThrow();
  });

  it('rejectNan does not reject Inf', async () => {
    await init();
    const data = new Float32Array([1.0, Infinity]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([2], 'float32'), data }],
        { rejectNan: true },
      ),
    ).not.toThrow();
  });

  it('rejectNan + rejectInf catches either', async () => {
    await init();
    const nanData = new Float32Array([1.0, NaN]);
    const infData = new Float32Array([1.0, Infinity]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([2], 'float32'), data: nanData }],
        { rejectNan: true, rejectInf: true },
      ),
    ).toThrow(/nan/i);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([2], 'float32'), data: infData }],
        { rejectNan: true, rejectInf: true },
      ),
    ).toThrow(/inf/i);
  });

  // ── Integer dtypes skip the scan ──────────────────────────────────────

  it('rejectNan skips uint32 data (zero-cost guarantee)', async () => {
    await init();
    const data = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([3], 'uint32'), data }],
        { rejectNan: true, rejectInf: true },
      ),
    ).not.toThrow();
  });

  // ── Edge cases ────────────────────────────────────────────────────────

  it('negative zero is not treated as NaN', async () => {
    await init();
    const data = new Float64Array([1.0, -0.0, 2.0]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([3], 'float64'), data }],
        { rejectNan: true, rejectInf: true },
      ),
    ).not.toThrow();
  });

  it('empty array passes', async () => {
    await init();
    const data = new Float32Array();
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([0], 'float32'), data }],
        { rejectNan: true, rejectInf: true },
      ),
    ).not.toThrow();
  });

  // ── Interaction with compression ──────────────────────────────────────

  it('rejectNan fires before lz4 compression', async () => {
    await init();
    const data = new Float64Array([1.0, NaN, 3.0]);
    const desc = { ...makeDescriptor([3], 'float64'), compression: 'lz4' };
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: desc, data }],
        { rejectNan: true },
      ),
    ).toThrow(EncodingError);
  });

  it('NaN passes lz4 when flag is off (byte-level lossless)', async () => {
    await init();
    const data = new Float64Array([1.0, NaN, 3.0]);
    const desc = { ...makeDescriptor([3], 'float64'), compression: 'lz4' };
    expect(() =>
      encode(defaultMeta(), [{ descriptor: desc, data }]),
    ).not.toThrow();
  });

  // ── Standalone-API safety net — see plans/RESEARCH_NAN_HANDLING.md §4.2.3 ──

  describe('simple_packing params safety net', () => {
    const simplePackingDesc = (
      binaryScaleFactor: number,
      bitsPerValue: number = 16,
    ) => ({
      ...makeDescriptor([4], 'float64'),
      encoding: 'simple_packing',
      reference_value: 273.15,
      binary_scale_factor: binaryScaleFactor,
      decimal_scale_factor: 0,
      bits_per_value: bitsPerValue,
    });

    it('rejects binary_scale_factor = i32::MAX (silent-corruption fingerprint)', async () => {
      await init();
      const data = new Float64Array([273.15, 283.0, 293.0, 303.0]);
      const desc = simplePackingDesc(2 ** 31 - 1);
      expect(() => encode(defaultMeta(), [{ descriptor: desc, data }])).toThrow(
        /binary_scale_factor/,
      );
    });

    it('256 is accepted, 257 is rejected (threshold boundary)', async () => {
      await init();
      const data = new Float64Array([1.0, 2.0, 3.0, 4.0]);
      expect(() =>
        encode(defaultMeta(), [{ descriptor: simplePackingDesc(256), data }]),
      ).not.toThrow();
      expect(() =>
        encode(defaultMeta(), [{ descriptor: simplePackingDesc(257), data }]),
      ).toThrow(/binary_scale_factor/);
    });

    it('accepts realistic weather-data binary_scale_factor values', async () => {
      await init();
      const data = new Float64Array([273.15, 283.0, 293.0, 303.0]);
      for (const bsf of [-60, -20, 0, 20, 60]) {
        expect(() =>
          encode(defaultMeta(), [{ descriptor: simplePackingDesc(bsf), data }]),
        ).not.toThrow();
      }
    });

    it('accepts bits_per_value = 0 for constant-field encoding', async () => {
      await init();
      const data = new Float64Array([42.0, 42.0, 42.0, 42.0]);
      const desc = {
        ...makeDescriptor([4], 'float64'),
        encoding: 'simple_packing',
        reference_value: 42.0,
        binary_scale_factor: 0,
        decimal_scale_factor: 0,
        bits_per_value: 0,
      };
      expect(() =>
        encode(defaultMeta(), [{ descriptor: desc, data }]),
      ).not.toThrow();
    });
  });
});

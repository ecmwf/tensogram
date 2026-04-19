// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for the allow_nan / allow_inf TypeScript bindings (Commit 10
// of plans/BITMASK_FRAME.md).

import { describe, expect, it } from 'vitest';
import { decode, encode, init, StreamingEncoder } from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('allow_nan / allow_inf bindings', () => {
  it('rejects NaN by default', async () => {
    await init();
    const data = new Float64Array([1, NaN, 3]);
    expect(() =>
      encode(defaultMeta(), [{ descriptor: makeDescriptor([3], 'float64'), data }]),
    ).toThrow(/NaN/);
  });

  it('rejects positive Inf by default', async () => {
    await init();
    const data = new Float32Array([1, Infinity]);
    expect(() =>
      encode(defaultMeta(), [{ descriptor: makeDescriptor([2], 'float32'), data }]),
    ).toThrow(/Inf/);
  });

  it('allow_nan round-trips NaN via default decode (restoreNonFinite=true)', async () => {
    await init();
    const data = new Float64Array([1, NaN, 3, NaN, 5]);
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([5], 'float64'), data }],
      { allowNan: true, smallMaskThresholdBytes: 0 },
    );
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as Float64Array;
    expect(out[0]).toBe(1);
    expect(Number.isNaN(out[1])).toBe(true);
    expect(out[2]).toBe(3);
    expect(Number.isNaN(out[3])).toBe(true);
    expect(out[4]).toBe(5);
    decoded.close();
  });

  it('restoreNonFinite=false returns 0.0 at masked positions', async () => {
    await init();
    const data = new Float64Array([1, NaN, 3]);
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([3], 'float64'), data }],
      { allowNan: true },
    );
    const decoded = decode(msg, { restoreNonFinite: false });
    const out = decoded.objects[0].data() as Float64Array;
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(0);
    expect(out[2]).toBe(3);
    decoded.close();
  });

  it('allow_inf restores +Inf and -Inf', async () => {
    await init();
    const data = new Float32Array([1, Infinity, -Infinity, 2]);
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([4], 'float32'), data }],
      { allowInf: true, smallMaskThresholdBytes: 0 },
    );
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as Float32Array;
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(Infinity);
    expect(out[2]).toBe(-Infinity);
    expect(out[3]).toBe(2);
    decoded.close();
  });

  it('supports mask methods available in WASM build', async () => {
    await init();
    // WASM ships with lz4 but not full zstd (zstd-pure is decode-only).
    // The zstd mask-method path errors cleanly at encode — covered by
    // the "zstd raises FeatureDisabled" test below.
    for (const method of ['none', 'rle', 'roaring', 'lz4'] as const) {
      const values = new Float64Array(128);
      for (let i = 0; i < 128; i++) values[i] = i;
      values[10] = NaN;
      values[50] = NaN;
      values[100] = NaN;
      const msg = encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([128], 'float64'), data: values }],
        { allowNan: true, nanMaskMethod: method, smallMaskThresholdBytes: 0 },
      );
      const decoded = decode(msg);
      const out = decoded.objects[0].data() as Float64Array;
      expect(Number.isNaN(out[10])).toBe(true);
      expect(Number.isNaN(out[50])).toBe(true);
      expect(Number.isNaN(out[100])).toBe(true);
      expect(out[11]).toBe(11);
      expect(out[99]).toBe(99);
      decoded.close();
    }
  });

  it('unknown mask method name raises a JsError (no silent fallback)', async () => {
    await init();
    const data = new Float64Array([1, NaN, 3]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([3], 'float64'), data }],
        {
          allowNan: true,
          // Cast to bypass the TypeScript type guard — we want to
          // exercise the WASM-side validation on arbitrary user
          // input (e.g. runtime strings from CLI flags).
          nanMaskMethod: 'totally-bogus' as never,
          smallMaskThresholdBytes: 0,
        },
      ),
    ).toThrow(/unknown mask method/);
  });

  it('zstd mask method raises feature-disabled error on WASM', async () => {
    await init();
    const data = new Float64Array([NaN, 1, 2]);
    expect(() =>
      encode(
        defaultMeta(),
        [{ descriptor: makeDescriptor([3], 'float64'), data }],
        { allowNan: true, nanMaskMethod: 'zstd', smallMaskThresholdBytes: 0 },
      ),
    ).toThrow(/zstd/i);
  });

  it('StreamingEncoder honours allow_nan', async () => {
    await init();
    const data = new Float64Array([1, NaN, 3]);
    const enc = new StreamingEncoder(defaultMeta(), {
      allowNan: true,
      smallMaskThresholdBytes: 0,
    });
    enc.writeObject(makeDescriptor([3], 'float64'), data);
    const msg = enc.finish();
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as Float64Array;
    expect(out[0]).toBe(1);
    expect(Number.isNaN(out[1])).toBe(true);
    expect(out[2]).toBe(3);
    decoded.close();
  });
});

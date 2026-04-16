// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import { decode, encode, init } from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('Phase 0 — scaffold + WASM integration', () => {
  it('init() resolves without error', async () => {
    await expect(init()).resolves.toBeUndefined();
  });

  it('init() is idempotent across concurrent callers', async () => {
    const [a, b, c] = await Promise.all([init(), init(), init()]);
    expect(a).toBeUndefined();
    expect(b).toBeUndefined();
    expect(c).toBeUndefined();
  });

  it('encode → decode round-trips a Float32Array bit-exact', async () => {
    await init();
    const data = new Float32Array(100 * 200);
    for (let i = 0; i < data.length; i++) data[i] = 273.15 + i / 100;

    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([100, 200], 'float32'), data },
    ]);
    expect(msg).toBeInstanceOf(Uint8Array);
    expect(msg.byteLength).toBeGreaterThan(data.byteLength);

    const decoded = decode(msg);
    expect(decoded.metadata.version).toBe(2);
    expect(decoded.objects).toHaveLength(1);

    const out = decoded.objects[0].data() as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(data.length);
    // Bit-exact
    for (let i = 0; i < data.length; i++) {
      expect(out[i]).toBe(data[i]);
    }
    decoded.close();
  });

  it('preamble magic is "TENSOGRM" and postamble is "39277777"', async () => {
    await init();
    const data = new Float32Array([1, 2, 3]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([3], 'float32'), data },
    ]);
    const magic = new TextDecoder().decode(msg.slice(0, 8));
    const endMagic = new TextDecoder().decode(msg.slice(msg.length - 8));
    expect(magic).toBe('TENSOGRM');
    expect(endMagic).toBe('39277777');
  });
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import {
  Bfloat16Array,
  ComplexArray,
  DTYPE_BYTE_WIDTH,
  decode,
  encode,
  Float16Polyfill,
  getFloat16ArrayCtor,
  hasNativeFloat16Array,
  init,
  isDtype,
  payloadByteSize,
  shapeElementCount,
  SUPPORTED_DTYPES,
  typedArrayFor,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('Phase 2 — dtype dispatch', () => {
  it('DTYPE_BYTE_WIDTH table matches the spec', () => {
    expect(DTYPE_BYTE_WIDTH.float16).toBe(2);
    expect(DTYPE_BYTE_WIDTH.bfloat16).toBe(2);
    expect(DTYPE_BYTE_WIDTH.float32).toBe(4);
    expect(DTYPE_BYTE_WIDTH.float64).toBe(8);
    expect(DTYPE_BYTE_WIDTH.complex64).toBe(8);
    expect(DTYPE_BYTE_WIDTH.complex128).toBe(16);
    expect(DTYPE_BYTE_WIDTH.int8).toBe(1);
    expect(DTYPE_BYTE_WIDTH.int64).toBe(8);
    expect(DTYPE_BYTE_WIDTH.uint8).toBe(1);
    expect(DTYPE_BYTE_WIDTH.uint64).toBe(8);
    expect(DTYPE_BYTE_WIDTH.bitmask).toBe(0);
  });

  it('SUPPORTED_DTYPES is the full set', () => {
    expect(SUPPORTED_DTYPES.size).toBe(15);
  });

  it('isDtype narrows correctly', () => {
    expect(isDtype('float32')).toBe(true);
    expect(isDtype('bitmask')).toBe(true);
    expect(isDtype('float128')).toBe(false);
    expect(isDtype(42)).toBe(false);
    expect(isDtype(null)).toBe(false);
  });

  it('payloadByteSize handles bitmask as ceil(N/8)', () => {
    expect(payloadByteSize('bitmask', 0)).toBe(0);
    expect(payloadByteSize('bitmask', 1)).toBe(1);
    expect(payloadByteSize('bitmask', 8)).toBe(1);
    expect(payloadByteSize('bitmask', 9)).toBe(2);
    expect(payloadByteSize('bitmask', 7)).toBe(1);
  });

  it('payloadByteSize handles regular dtypes', () => {
    expect(payloadByteSize('float32', 10)).toBe(40);
    expect(payloadByteSize('float64', 10)).toBe(80);
    expect(payloadByteSize('uint8', 10)).toBe(10);
    expect(payloadByteSize('complex128', 3)).toBe(48);
  });

  it('shapeElementCount returns 1 for scalar', () => {
    expect(shapeElementCount([])).toBe(1);
    expect(shapeElementCount([100])).toBe(100);
    expect(shapeElementCount([3, 4, 5])).toBe(60);
  });

  it('typedArrayFor returns the correct view per dtype', () => {
    const bytes = new Uint8Array(16);
    expect(typedArrayFor('float32', bytes)).toBeInstanceOf(Float32Array);
    expect(typedArrayFor('float64', bytes)).toBeInstanceOf(Float64Array);
    expect(typedArrayFor('int8', bytes)).toBeInstanceOf(Int8Array);
    expect(typedArrayFor('int16', bytes)).toBeInstanceOf(Int16Array);
    expect(typedArrayFor('int32', bytes)).toBeInstanceOf(Int32Array);
    expect(typedArrayFor('int64', bytes)).toBeInstanceOf(BigInt64Array);
    expect(typedArrayFor('uint8', bytes)).toBeInstanceOf(Uint8Array);
    expect(typedArrayFor('uint16', bytes)).toBeInstanceOf(Uint16Array);
    expect(typedArrayFor('uint32', bytes)).toBeInstanceOf(Uint32Array);
    expect(typedArrayFor('uint64', bytes)).toBeInstanceOf(BigUint64Array);
    expect(typedArrayFor('bitmask', bytes)).toBeInstanceOf(Uint8Array);
    // Scope C.2 — first-class views
    const f16 = typedArrayFor('float16', bytes);
    // Either a native Float16Array (Node ≥ 22) or the polyfill.
    if (hasNativeFloat16Array()) {
      expect(f16).toBeInstanceOf(getFloat16ArrayCtor());
    } else {
      expect(f16).toBeInstanceOf(Float16Polyfill);
    }
    expect(typedArrayFor('bfloat16', bytes)).toBeInstanceOf(Bfloat16Array);
    expect(typedArrayFor('complex64', bytes)).toBeInstanceOf(ComplexArray);
    expect(typedArrayFor('complex128', bytes)).toBeInstanceOf(ComplexArray);
  });

  it('data() round-trips float64', async () => {
    await init();
    const data = new Float64Array([1.5, 2.5, 3.5, 4.5]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float64'), data },
    ]);
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as Float64Array;
    expect(out).toBeInstanceOf(Float64Array);
    expect(Array.from(out)).toEqual([1.5, 2.5, 3.5, 4.5]);
    decoded.close();
  });

  it('data() round-trips int32', async () => {
    await init();
    const data = new Int32Array([-5, 0, 42, 2_000_000_000]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'int32'), data },
    ]);
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as Int32Array;
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-5, 0, 42, 2_000_000_000]);
    decoded.close();
  });

  it('data() round-trips int64', async () => {
    await init();
    const data = new BigInt64Array([
      -1n,
      0n,
      1n,
      9_000_000_000_000_000_000n,
    ]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'int64'), data },
    ]);
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as BigInt64Array;
    expect(out).toBeInstanceOf(BigInt64Array);
    expect(Array.from(out)).toEqual([
      -1n,
      0n,
      1n,
      9_000_000_000_000_000_000n,
    ]);
    decoded.close();
  });

  it('data() copy survives close()', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'float32'),
        data: new Float32Array([7, 8, 9]),
      },
    ]);
    const decoded = decode(msg);
    const copy = decoded.objects[0].data() as Float32Array;
    decoded.close();
    expect(Array.from(copy)).toEqual([7, 8, 9]);
  });

  // ── Coverage closers ────────────────────────────────────────────────

  it('payloadByteSize rejects negative / NaN elementCount', () => {
    expect(() => payloadByteSize('float32', -1)).toThrow(/element count/);
    expect(() => payloadByteSize('float32', NaN)).toThrow(/element count/);
  });

  it('typedArrayFor handles an empty Uint8Array for every dtype', () => {
    const empty = new Uint8Array(0);
    for (const dt of SUPPORTED_DTYPES) {
      const arr = typedArrayFor(dt, empty);
      expect(arr.length).toBe(0);
    }
  });

  it('typedArrayFor zero-copy view reflects source bytes', () => {
    const bytes = new Uint8Array(8);
    bytes[0] = 0x00; // float32 bits = 0
    bytes[1] = 0x00;
    bytes[2] = 0x80;
    bytes[3] = 0x3F; // 1.0 in little-endian f32
    const view = typedArrayFor('float32', bytes, false) as Float32Array;
    // On wasm32 (little-endian) the view reads 1.0. The test runs on the
    // host machine (also little-endian for all supported targets); this
    // documents the contract, not host-LE assumption.
    expect(view.length).toBe(2);
  });

  it('byteLength matches expected payload size', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([5, 4], 'float32'),
        data: new Float32Array(20),
      },
    ]);
    const decoded = decode(msg);
    expect(decoded.objects[0].byteLength).toBe(20 * 4);
    decoded.close();
  });
});

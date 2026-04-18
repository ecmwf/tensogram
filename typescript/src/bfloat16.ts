// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `Bfloat16Array` — brain-float-16, the 8-bit-exponent / 7-bit-mantissa
 * floating-point format popular in ML frameworks.
 *
 * Bit layout: `sign:1 | exp:8 | mantissa:7` — identical to `float32`'s
 * top 16 bits.  This makes conversion close to a byte copy: widening
 * is "shift left by 16", narrowing is "take the top 16 bits (with
 * round-to-nearest-even) of the float32 representation".
 *
 * The API surface matches {@link Float16Polyfill}: array-like
 * accessors, iteration, `.bits`, `.toFloat32Array()`.  There is no
 * native JS `Bfloat16Array` today; this class is always used.
 */

const _f32_scratch = new Float32Array(1);
const _u32_scratch = new Uint32Array(_f32_scratch.buffer);

/**
 * Widen a bfloat16 bit pattern to a `number`.
 *
 * @internal — not re-exported from the package barrel.
 */
export function bfloat16BitsToFloat(bits: number): number {
  _u32_scratch[0] = (bits & 0xffff) << 16;
  return _f32_scratch[0];
}

/**
 * Narrow a `number` to a bfloat16 bit pattern.  Uses round-to-nearest-
 * even on the 16 discarded float32 mantissa bits.  NaN propagates with
 * the sign / quiet bit preserved; ±Inf saturates.
 *
 * @internal — not re-exported from the package barrel.
 */
export function floatToBfloat16Bits(value: number): number {
  _f32_scratch[0] = value;
  const x = _u32_scratch[0];
  const high = x >>> 16;

  // NaN: force mantissa ≠ 0 in the 7 retained bits, preserve sign.
  const exp = (x >>> 23) & 0xff;
  const mantissa = x & 0x7fffff;
  if (exp === 0xff) {
    if (mantissa === 0) return high; // ±Inf
    // Keep the sign bit + all-ones exponent, and guarantee at least
    // one set mantissa bit.  Preserving the original high-mantissa
    // bits is good enough for nearly-transparent NaN propagation.
    return (high | 0x40) & 0xffff;
  }

  // Round-to-nearest-even on the dropped bits.
  const dropped = x & 0xffff;
  const halfway = 0x8000;
  const lsb = high & 1;
  if (dropped > halfway || (dropped === halfway && lsb === 1)) {
    const rounded = (high + 1) & 0xffff;
    // Propagate exponent overflow into ±Inf.
    if (((rounded >>> 7) & 0xff) === 0xff && (rounded & 0x7f) !== 0) {
      return (rounded & 0x8000) | 0x7f80; // saturate to ±Inf
    }
    return rounded;
  }
  return high;
}

const STORAGE = new WeakMap<object, Uint16Array>();

/** Brain-float-16 array.  API-compatible with {@link Float16Polyfill}. */
export class Bfloat16Array {
  readonly BYTES_PER_ELEMENT = 2 as const;

  constructor(
    a?: number | ArrayLike<number> | Iterable<number> | ArrayBufferLike,
    b?: number,
    c?: number,
  ) {
    let bits: Uint16Array;
    if (a === undefined) {
      bits = new Uint16Array(0);
    } else if (typeof a === 'number') {
      bits = new Uint16Array(a);
    } else if (
      a instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && a instanceof SharedArrayBuffer)
    ) {
      bits = new Uint16Array(a as ArrayBufferLike, b, c);
    } else if (ArrayBuffer.isView(a)) {
      const src = a as unknown as ArrayLike<number>;
      bits = new Uint16Array(src.length);
      for (let i = 0; i < src.length; i++) bits[i] = floatToBfloat16Bits(src[i]);
    } else {
      const arr = Array.from(a as Iterable<number>);
      bits = new Uint16Array(arr.length);
      for (let i = 0; i < arr.length; i++) bits[i] = floatToBfloat16Bits(arr[i]);
    }
    STORAGE.set(this, bits);
  }

  // STORAGE is populated by the constructor (or `wrapBits`) before
  // the instance is exposed — the non-null assertion documents this
  // total invariant without a redundant runtime check.
  private get _bits(): Uint16Array {
    return STORAGE.get(this)!;
  }

  get length(): number {
    return this._bits.length;
  }
  get byteLength(): number {
    return this._bits.byteLength;
  }
  get byteOffset(): number {
    return this._bits.byteOffset;
  }
  get buffer(): ArrayBufferLike {
    return this._bits.buffer;
  }
  /** Raw bfloat16 bit patterns, zero-copy. */
  get bits(): Uint16Array {
    return this._bits;
  }

  at(index: number): number | undefined {
    const b = this._bits;
    if (index < 0) index += b.length;
    if (index < 0 || index >= b.length) return undefined;
    return bfloat16BitsToFloat(b[index]);
  }

  get(index: number): number {
    const b = this._bits;
    if (!Number.isInteger(index) || index < 0 || index >= b.length) return NaN;
    return bfloat16BitsToFloat(b[index]);
  }

  set(source: ArrayLike<number> | Bfloat16Array, offset = 0): void {
    const b = this._bits;
    if (source instanceof Bfloat16Array) {
      b.set(source.bits, offset);
      return;
    }
    const arr = source as ArrayLike<number>;
    for (let i = 0; i < arr.length; i++) b[offset + i] = floatToBfloat16Bits(arr[i]);
  }

  fill(value: number, start?: number, end?: number): this {
    this._bits.fill(floatToBfloat16Bits(value), start, end);
    return this;
  }

  slice(start?: number, end?: number): Bfloat16Array {
    return wrapBits(this._bits.slice(start, end));
  }

  subarray(start?: number, end?: number): Bfloat16Array {
    return wrapBits(this._bits.subarray(start, end));
  }

  toArray(): number[] {
    const b = this._bits;
    const out = new Array<number>(b.length);
    for (let i = 0; i < b.length; i++) out[i] = bfloat16BitsToFloat(b[i]);
    return out;
  }

  toFloat32Array(): Float32Array {
    const b = this._bits;
    const out = new Float32Array(b.length);
    for (let i = 0; i < b.length; i++) out[i] = bfloat16BitsToFloat(b[i]);
    return out;
  }

  *[Symbol.iterator](): IterableIterator<number> {
    const b = this._bits;
    for (let i = 0; i < b.length; i++) yield bfloat16BitsToFloat(b[i]);
  }
}

function wrapBits(bits: Uint16Array): Bfloat16Array {
  const inst = Object.create(Bfloat16Array.prototype) as Bfloat16Array;
  STORAGE.set(inst, bits);
  return inst;
}

/**
 * Build a `Bfloat16Array` from raw binary bytes (2 bytes per element).
 *
 * @throws {RangeError} when `bytes.byteLength` is not a multiple of 2.
 */
export function bfloat16FromBytes(bytes: Uint8Array): Bfloat16Array {
  if (!Number.isInteger(bytes.byteLength / 2)) {
    throw new RangeError(
      `bfloat16FromBytes: byte length ${bytes.byteLength} is not a multiple of 2`,
    );
  }
  const aligned = bytes.byteOffset % 2 === 0 ? bytes : new Uint8Array(bytes);
  const u16 = new Uint16Array(
    aligned.buffer,
    aligned.byteOffset,
    aligned.byteLength / 2,
  );
  return wrapBits(u16);
}

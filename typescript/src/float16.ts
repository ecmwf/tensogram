// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `Float16Array` polyfill — IEEE 754 binary16 ("half-precision").
 *
 * Used by {@link typedArrayFor} to wrap `float16` payloads when the
 * host does not ship a native `Float16Array`.  The observable
 * behaviour matches the TC39 Stage-3 proposal:
 *
 * - Constructor accepts `length`, an iterable/array-like of numbers
 *   (narrowed to binary16), or an `ArrayBufferLike` view (raw bits).
 * - `.length`, `.byteLength`, `.byteOffset`, `.buffer` mirror a
 *   `Uint16Array` of the underlying bits.
 * - `.at(i)`, `.get(i)` widen the stored bits back to a `number`.
 * - `.set(src, offset?)` narrows each incoming value to binary16
 *   using round-ties-to-even; NaN / ±Inf / ±0 / subnormals survive
 *   where binary16 can represent them.
 *
 * The backing storage is a `Uint16Array` of raw bits, accessible via
 * {@link Float16Polyfill.bits} for zero-copy passes back to the
 * wire or to a native `Float16Array` when one becomes available.
 */

// ── Bit-level conversion helpers ───────────────────────────────────────────

const _f32_scratch = new Float32Array(1);
const _u32_scratch = new Uint32Array(_f32_scratch.buffer);

/**
 * Convert a binary16 bit pattern to a `number` (float32 precision).
 *
 * Handles every case required by the TC39 proposal: ±0, subnormals,
 * normal range, ±Inf, NaN (with mantissa preservation where the lower
 * 10 bits fit into binary16's representation).
 *
 * @internal — not re-exported from the package barrel.  Callers that
 * need raw bit access should use a view's `.bits` accessor.
 */
export function halfBitsToFloat(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;

  if (exp === 0x1f) {
    // Inf / NaN
    if (mant === 0) {
      _u32_scratch[0] = sign | 0x7f800000;
    } else {
      _u32_scratch[0] = sign | 0x7fc00000 | (mant << 13);
    }
    return _f32_scratch[0];
  }

  if (exp === 0) {
    if (mant === 0) {
      _u32_scratch[0] = sign;
      return _f32_scratch[0];
    }
    // Subnormal: normalise into float32's normal range.
    let m = mant;
    let e = 1;
    while ((m & 0x400) === 0) {
      m <<= 1;
      e++;
    }
    m &= 0x3ff;
    _u32_scratch[0] = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
    return _f32_scratch[0];
  }

  _u32_scratch[0] = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  return _f32_scratch[0];
}

/**
 * Convert a JS `number` to its binary16 bit pattern, using
 * round-ties-to-even as specified by IEEE 754.
 *
 * @internal — not re-exported from the package barrel.
 */
export function floatToHalfBits(value: number): number {
  _f32_scratch[0] = value;
  const x = _u32_scratch[0];
  const sign = (x >>> 16) & 0x8000;
  const exp32 = (x >>> 23) & 0xff;
  const mant32 = x & 0x7fffff;

  if (exp32 === 0xff) {
    if (mant32 === 0) return sign | 0x7c00; // ±Inf
    // NaN — preserve the quiet-bit into binary16's mantissa MSB.
    const mant16 = ((mant32 >>> 13) | 0x200) & 0x3ff;
    return sign | 0x7c00 | mant16;
  }

  const e = exp32 - 112; // exponent re-biased for binary16

  if (e >= 0x1f) return sign | 0x7c00; // overflow → ±Inf

  if (e <= 0) {
    if (e < -10) return sign; // underflow → ±0
    // Subnormal: add implicit leading 1, shift, round-to-nearest-even.
    const m = mant32 | 0x800000;
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    const mask = (1 << shift) - 1;
    const rounded = m + half;
    if ((m & mask) === half && ((rounded >>> shift) & 1) === 1) {
      return sign | ((rounded - (1 << shift)) >>> shift);
    }
    return sign | (rounded >>> shift);
  }

  // Normal: round-to-nearest-even on the low 13 bits.
  const half = 0x1000;
  const mask = 0x1fff;
  const rounded = x + half;
  const newExp = ((rounded >>> 23) & 0xff) - 112;
  const newMant = (rounded >>> 13) & 0x3ff;

  // Exact tie → prefer even LSB (drop the rounding increment).
  if ((x & mask) === half && (newMant & 1) === 1) {
    const exact = x & ~mask;
    const origExp = ((exact >>> 23) & 0xff) - 112;
    const origMant = (exact >>> 13) & 0x3ff;
    if (origExp >= 0x1f) return sign | 0x7c00;
    return sign | (origExp << 10) | origMant;
  }

  if (newExp >= 0x1f) return sign | 0x7c00;
  return sign | (newExp << 10) | newMant;
}

// ── Polyfill class ─────────────────────────────────────────────────────────

// The "private bits" slot is held in a module-scope WeakMap so that
// `wrapBits()` can build an instance that *shares* storage with an
// existing Uint16Array — impossible with a # private field because
// there is no external `new` overload that skips the
// narrow-every-element path.
const STORAGE = new WeakMap<object, Uint16Array>();

/** JS-land polyfill for the TC39 `Float16Array` proposal. */
export class Float16Polyfill {
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
      // Raw-bits view into an existing buffer — matches the native
      // `new Float16Array(buffer, byteOffset?, length?)` shape.
      bits = new Uint16Array(a as ArrayBufferLike, b, c);
    } else if (ArrayBuffer.isView(a)) {
      const src = a as unknown as ArrayLike<number>;
      bits = new Uint16Array(src.length);
      for (let i = 0; i < src.length; i++) bits[i] = floatToHalfBits(src[i]);
    } else {
      const arr = Array.from(a as Iterable<number>);
      bits = new Uint16Array(arr.length);
      for (let i = 0; i < arr.length; i++) bits[i] = floatToHalfBits(arr[i]);
    }
    STORAGE.set(this, bits);
  }

  // ── views into the backing Uint16Array ──
  //
  // STORAGE is populated by the constructor (or `wrapBits`) before the
  // instance is exposed to user code, and the key is never removed —
  // so the getter's invariant is total.  The non-null assertion
  // documents this; the `!` throws a TypeError if it ever became
  // untrue, catching refactor bugs without shipping an extra
  // conditional on every indexed read.
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
  /** Raw half-precision bit patterns, zero-copy. */
  get bits(): Uint16Array {
    return this._bits;
  }

  // ── numeric accessors ──
  at(index: number): number | undefined {
    const b = this._bits;
    if (index < 0) index += b.length;
    if (index < 0 || index >= b.length) return undefined;
    return halfBitsToFloat(b[index]);
  }

  /** Out-of-range → `NaN` (mirrors TypedArray out-of-range behaviour). */
  get(index: number): number {
    const b = this._bits;
    if (!Number.isInteger(index) || index < 0 || index >= b.length) return NaN;
    return halfBitsToFloat(b[index]);
  }

  set(source: ArrayLike<number> | Float16Polyfill, offset = 0): void {
    const b = this._bits;
    if (source instanceof Float16Polyfill) {
      // Same format — copy raw bits, no re-narrowing.
      b.set(source.bits, offset);
      return;
    }
    const arr = source as ArrayLike<number>;
    for (let i = 0; i < arr.length; i++) b[offset + i] = floatToHalfBits(arr[i]);
  }

  fill(value: number, start?: number, end?: number): this {
    this._bits.fill(floatToHalfBits(value), start, end);
    return this;
  }

  slice(start?: number, end?: number): Float16Polyfill {
    return wrapBits(this._bits.slice(start, end));
  }

  subarray(start?: number, end?: number): Float16Polyfill {
    return wrapBits(this._bits.subarray(start, end));
  }

  /** Plain `number[]` of widened float32 values. */
  toArray(): number[] {
    const b = this._bits;
    const out = new Array<number>(b.length);
    for (let i = 0; i < b.length; i++) out[i] = halfBitsToFloat(b[i]);
    return out;
  }

  /** Widened copy into a `Float32Array`. */
  toFloat32Array(): Float32Array {
    const b = this._bits;
    const out = new Float32Array(b.length);
    for (let i = 0; i < b.length; i++) out[i] = halfBitsToFloat(b[i]);
    return out;
  }

  *[Symbol.iterator](): IterableIterator<number> {
    const b = this._bits;
    for (let i = 0; i < b.length; i++) yield halfBitsToFloat(b[i]);
  }
}

/**
 * Build a `Float16Polyfill` from an existing `Uint16Array` of binary16
 * bits without re-narrowing.  Internal helper used by
 * {@link float16FromBytes}.
 */
function wrapBits(bits: Uint16Array): Float16Polyfill {
  const inst = Object.create(Float16Polyfill.prototype) as Float16Polyfill;
  STORAGE.set(inst, bits);
  return inst;
}

// ── Native detection ───────────────────────────────────────────────────────

/**
 * Minimal structural description of a native `Float16Array`.  Covers
 * everything the wrapper needs at the type level without dragging in
 * the full lib.dom definition (which is Stage-3 and not stable in
 * every TS release).
 */
interface NativeFloat16Array {
  readonly BYTES_PER_ELEMENT: 2;
  readonly length: number;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly buffer: ArrayBufferLike;
  at(index: number): number | undefined;
  [i: number]: number;
  [Symbol.iterator](): IterableIterator<number>;
}

/**
 * Constructor for a native `Float16Array`.  Supports the three
 * shapes the TC39 proposal and the polyfill both accept: length,
 * array-like of numbers (narrowed on write), or an `ArrayBuffer` +
 * offset + length bit-view.
 */
interface NativeFloat16Ctor {
  new (length: number): NativeFloat16Array;
  new (source: ArrayLike<number> | Iterable<number>): NativeFloat16Array;
  new (
    buffer: ArrayBufferLike,
    byteOffset?: number,
    length?: number,
  ): NativeFloat16Array;
}

/** Constructor type covering both native and polyfill. */
export type Float16Ctor = typeof Float16Polyfill | NativeFloat16Ctor;

/** Instance type covering both native and polyfill. */
export type Float16ArrayLike = Float16Polyfill | NativeFloat16Array;

function nativeCtor(): NativeFloat16Ctor | undefined {
  const g = globalThis as unknown as { Float16Array?: NativeFloat16Ctor };
  return typeof g.Float16Array === 'function' ? g.Float16Array : undefined;
}

/**
 * True iff the host ships a native `Float16Array` (TC39 Stage-3).
 * Probed every call so test-time overrides are honoured.
 */
export function hasNativeFloat16Array(): boolean {
  return nativeCtor() !== undefined;
}

/**
 * Return the preferred constructor: native when available, polyfill
 * otherwise.  Use to construct fresh instances (length-ctor path);
 * prefer {@link float16FromBytes} for zero-copy wrapping of existing
 * binary16 bytes.
 */
export function getFloat16ArrayCtor(): Float16Ctor {
  return nativeCtor() ?? Float16Polyfill;
}

/**
 * Wrap raw binary16 bytes as a `Float16Array`-like view.  Zero-copy on
 * 2-byte-aligned input; allocates an aligned copy when the offset is
 * odd (rare — our WASM output is aligned by construction).
 *
 * @throws {RangeError} when `bytes.byteLength` is not a multiple of 2
 *   — binary16 elements are exactly 2 bytes each, so a residual byte
 *   would mean a truncated / corrupted payload.
 */
export function float16FromBytes(bytes: Uint8Array): Float16ArrayLike {
  if (!Number.isInteger(bytes.byteLength / 2)) {
    throw new RangeError(
      `float16FromBytes: byte length ${bytes.byteLength} is not a multiple of 2`,
    );
  }
  const aligned = bytes.byteOffset % 2 === 0 ? bytes : new Uint8Array(bytes);
  const u16 = new Uint16Array(
    aligned.buffer,
    aligned.byteOffset,
    aligned.byteLength / 2,
  );
  const Native = nativeCtor();
  if (Native !== undefined) {
    // Native: bit-view construction via the (buffer, byteOffset, length) overload.
    return new Native(u16.buffer, u16.byteOffset, u16.length);
  }
  return wrapBits(u16);
}

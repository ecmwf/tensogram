// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `ComplexArray` — typed view over interleaved complex payloads.
 *
 * Matches the Tensogram wire contract: `complex64` stores two float32
 * components per element `[re₀, im₀, re₁, im₁, …]`, `complex128` stores
 * two float64 components.  This class exposes `numpy`-flavoured
 * accessors so callers don't have to remember the interleaving:
 *
 * ```ts
 * const cplx = complexFromBytes('complex64', bytes);
 * for (let i = 0; i < cplx.length; i++) {
 *   const { re, im } = cplx.get(i);
 *   // …
 * }
 * ```
 *
 * Accessors return primitives (`number`) — there is no `Complex` value
 * type.  That avoids the per-access allocation `cplx.get(i)` would
 * otherwise cost, and mirrors how ML frameworks and `numpy` expose
 * complex dtypes at the array boundary.
 */

import type { Dtype } from './types.js';

/** Which complex dtype a view covers.  The element-storage type flows from here. */
export type ComplexDtype = 'complex64' | 'complex128';

/** Underlying storage: `Float32Array` for complex64, `Float64Array` for complex128. */
export type ComplexStorage = Float32Array | Float64Array;

/** A single `{ re, im }` pair returned by {@link ComplexArray.get}. */
export interface ComplexPair {
  re: number;
  im: number;
}

/**
 * Read-only iteration over interleaved complex pairs.
 *
 * Instances are lightweight — a class-level wrapper around an existing
 * `Float32Array` / `Float64Array`.  Zero-copy when built via
 * {@link complexFromBytes} on an aligned `Uint8Array`.
 */
export class ComplexArray {
  readonly dtype: ComplexDtype;
  readonly #storage: ComplexStorage;

  /**
   * @param dtype   - Which complex variant this view represents.
   * @param storage - Interleaved `[re, im, re, im, …]` storage.  Length
   *                  must be even.
   */
  constructor(dtype: ComplexDtype, storage: ComplexStorage) {
    if (dtype !== 'complex64' && dtype !== 'complex128') {
      throw new TypeError(
        `ComplexArray: unknown complex dtype "${String(dtype)}"; expected "complex64" or "complex128"`,
      );
    }
    if (dtype === 'complex64' && !(storage instanceof Float32Array)) {
      throw new TypeError('ComplexArray: complex64 requires a Float32Array storage');
    }
    if (dtype === 'complex128' && !(storage instanceof Float64Array)) {
      throw new TypeError('ComplexArray: complex128 requires a Float64Array storage');
    }
    if (storage.length % 2 !== 0) {
      throw new RangeError(
        `ComplexArray: storage length ${storage.length} must be even (interleaved re/im pairs)`,
      );
    }
    this.dtype = dtype;
    this.#storage = storage;
  }

  /** Number of complex pairs stored. */
  get length(): number {
    return this.#storage.length / 2;
  }

  /** Raw interleaved storage, zero-copy. */
  get data(): ComplexStorage {
    return this.#storage;
  }

  /**
   * Real component of the `i`-th complex value.  Out-of-range reads
   * return `NaN`, matching {@link Float16Polyfill.get}.
   */
  real(i: number): number {
    return this.#inRange(i) ? this.#storage[2 * i] : NaN;
  }

  /** Imaginary component of the `i`-th complex value. */
  imag(i: number): number {
    return this.#inRange(i) ? this.#storage[2 * i + 1] : NaN;
  }

  /**
   * Both components as a fresh `{ re, im }` pair.  Does a single
   * bounds check rather than the two `real()` + `imag()` would cost.
   */
  get(i: number): ComplexPair {
    if (!this.#inRange(i)) return { re: NaN, im: NaN };
    return { re: this.#storage[2 * i], im: this.#storage[2 * i + 1] };
  }

  /**
   * Write a single complex pair.  Must lie within range — attempting to
   * set an out-of-range index throws, since silently dropping writes
   * would defy the normal TypedArray contract.
   */
  set(i: number, re: number, im: number): void {
    if (!this.#inRange(i)) {
      throw new RangeError(
        `ComplexArray.set: index ${i} out of range [0, ${this.length})`,
      );
    }
    this.#storage[2 * i] = re;
    this.#storage[2 * i + 1] = im;
  }

  #inRange(i: number): boolean {
    return Number.isInteger(i) && i >= 0 && i < this.length;
  }

  /** Iterate over all pairs — yields `{ re, im }` per element. */
  *[Symbol.iterator](): IterableIterator<ComplexPair> {
    for (let i = 0; i < this.length; i++) yield this.get(i);
  }

  /** Returns a copy: `[[re0, im0], [re1, im1], …]`. */
  toArray(): Array<[number, number]> {
    const out = new Array<[number, number]>(this.length);
    for (let i = 0; i < this.length; i++) out[i] = [this.real(i), this.imag(i)];
    return out;
  }
}

/**
 * Build a `ComplexArray` over raw interleaved bytes.  Zero-copy when
 * the input is scalar-aligned; allocates an aligned copy when it isn't.
 *
 * @throws {RangeError} when `bytes.byteLength` is not a multiple of
 *   `2 * scalar_bytes` — a complex element is one `re` + one `im`
 *   scalar, so a residual byte means a truncated payload.  `complex64`
 *   requires multiples of 8; `complex128` requires multiples of 16.
 */
export function complexFromBytes(dtype: ComplexDtype, bytes: Uint8Array): ComplexArray {
  const scalarBytes = dtype === 'complex64' ? 4 : 8;
  const pairBytes = 2 * scalarBytes;
  if (!Number.isInteger(bytes.byteLength / pairBytes)) {
    throw new RangeError(
      `complexFromBytes: byte length ${bytes.byteLength} is not a multiple of ${pairBytes} (${dtype} pair size)`,
    );
  }
  const aligned = bytes.byteOffset % scalarBytes === 0 ? bytes : new Uint8Array(bytes);
  const storage =
    dtype === 'complex64'
      ? new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4)
      : new Float64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
  return new ComplexArray(dtype, storage);
}

/**
 * Type guard for the two complex variants.
 *
 * @internal — used by `dtype.ts` to route complex types; not
 * re-exported from the package barrel.  Callers should pattern-match
 * on the descriptor's `dtype` string directly.
 */
export function isComplexDtype(dtype: Dtype): dtype is ComplexDtype {
  return dtype === 'complex64' || dtype === 'complex128';
}

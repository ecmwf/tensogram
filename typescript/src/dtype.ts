// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Dtype → byte width and TypedArray constructor dispatch.
 *
 * Mirrors `tensogram_core::dtype::Dtype::byte_width()`. Bitmask is a
 * sentinel with byte-width `0`; its payload size is
 * `ceil(num_elements / 8)`, computed from the descriptor.
 *
 * For dtypes that JS has no native `TypedArray` for (`float16`,
 * `bfloat16`, `complex*`) we expose a surrogate `TypedArray` and
 * leave higher-level conversion to the consumer:
 *
 * - `float16` / `bfloat16` → `Uint16Array` (raw half-precision bits)
 * - `complex64`  → `Float32Array` with interleaved `[re, im, re, im, ...]`
 * - `complex128` → `Float64Array` with interleaved `[re, im, re, im, ...]`
 */

import { InvalidArgumentError } from './errors.js';
import type { Dtype, TypedArray } from './types.js';

/** Byte width per scalar element. `0` for `bitmask` (sub-byte packed). */
export const DTYPE_BYTE_WIDTH: Readonly<Record<Dtype, number>> = Object.freeze({
  float16: 2,
  bfloat16: 2,
  float32: 4,
  float64: 8,
  complex64: 8,
  complex128: 16,
  int8: 1,
  int16: 2,
  int32: 4,
  int64: 8,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  uint64: 8,
  bitmask: 0,
});

/**
 * Compute the payload byte count for a given dtype and total element
 * count (product of shape). Handles the `bitmask` special case.
 */
export function payloadByteSize(dtype: Dtype, elementCount: number): number {
  if (!Number.isFinite(elementCount) || elementCount < 0) {
    throw new InvalidArgumentError(
      `invalid element count: ${elementCount}`,
      `invalid element count: ${elementCount}`,
    );
  }
  if (dtype === 'bitmask') {
    return Math.ceil(elementCount / 8);
  }
  return elementCount * DTYPE_BYTE_WIDTH[dtype];
}

/** Convenience: element count from a shape array. */
export function shapeElementCount(shape: readonly number[]): number {
  if (shape.length === 0) return 1; // scalar
  return shape.reduce((acc, s) => acc * s, 1);
}

/**
 * Given a dtype and raw payload bytes, construct the appropriate
 * `TypedArray` view.
 *
 * For complex dtypes the returned array contains interleaved real /
 * imaginary components — e.g. `complex64` yields a `Float32Array` of
 * length `2 × num_elements`. Consumers that want a pair of separate
 * real / imag arrays can destructure as they see fit.
 *
 * @param dtype - element type
 * @param bytes - raw payload bytes in native byte order
 * @param copy  - if true (default), copies into a fresh JS-heap buffer;
 *                if false, returns a zero-copy view over `bytes.buffer`
 *                that is invalidated when the underlying buffer moves
 */
export function typedArrayFor(dtype: Dtype, bytes: Uint8Array, copy = true): TypedArray {
  const source = copy ? new Uint8Array(bytes) : bytes;

  // Aligned construction requires offset divisible by element width.
  // wasm-bindgen already produces aligned payloads, but when callers
  // hand us a subarray of an unaligned buffer we fall back to a
  // byte-level copy into a fresh ArrayBuffer.
  const needAligned = (alignment: number): Uint8Array => {
    if (source.byteOffset % alignment === 0) return source;
    return new Uint8Array(source); // fresh ArrayBuffer, offset 0
  };

  switch (dtype) {
    case 'float16':
    case 'bfloat16': {
      const aligned = needAligned(2);
      return new Uint16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
    }
    case 'float32': {
      const aligned = needAligned(4);
      return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
    }
    case 'float64': {
      const aligned = needAligned(8);
      return new Float64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
    }
    case 'complex64': {
      const aligned = needAligned(4);
      return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
    }
    case 'complex128': {
      const aligned = needAligned(8);
      return new Float64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
    }
    case 'int8':
      return new Int8Array(source.buffer, source.byteOffset, source.byteLength);
    case 'int16': {
      const aligned = needAligned(2);
      return new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
    }
    case 'int32': {
      const aligned = needAligned(4);
      return new Int32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
    }
    case 'int64': {
      const aligned = needAligned(8);
      return new BigInt64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
    }
    case 'uint8':
    case 'bitmask':
      return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
    case 'uint16': {
      const aligned = needAligned(2);
      return new Uint16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
    }
    case 'uint32': {
      const aligned = needAligned(4);
      return new Uint32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
    }
    case 'uint64': {
      const aligned = needAligned(8);
      return new BigUint64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
    }
    default: {
      // Exhaustiveness guard.
      const _exhaustive: never = dtype;
      throw new InvalidArgumentError(
        `unknown dtype: ${String(_exhaustive)}`,
        `unknown dtype: ${String(_exhaustive)}`,
      );
    }
  }
}

/**
 * The set of all supported dtypes, as a frozen set.
 *
 * Useful for runtime validation at the input boundary.
 */
export const SUPPORTED_DTYPES: ReadonlySet<Dtype> = Object.freeze(
  new Set<Dtype>([
    'float16',
    'bfloat16',
    'float32',
    'float64',
    'complex64',
    'complex128',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'bitmask',
  ]),
);

/** Type guard that narrows an `unknown` to `Dtype`. */
export function isDtype(x: unknown): x is Dtype {
  return typeof x === 'string' && SUPPORTED_DTYPES.has(x as Dtype);
}

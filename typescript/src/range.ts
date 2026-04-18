// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `decodeRange` — partial sub-tensor decode.
 *
 * Mirrors Rust `tensogram::decode_range`, Python `file_decode_range`,
 * and `tgm_decode_range`.  For an uncompressed or szip-encoded object,
 * decoding a small range is dramatically cheaper than a full decode
 * because only the requested bytes are read through the pipeline.
 *
 * The result is always a `parts` array of dtype-typed views; when
 * `join: true`, parts are concatenated into a single buffer before
 * the dtype wrap (one element in `parts`).  `parts.length` is always
 * the effective number of output slabs — never a discriminated union.
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';
import { typedArrayFor } from './dtype.js';
import type {
  DataObjectDescriptor,
  DecodeRangeOptions,
  DecodeRangeResult,
  RangePair,
  TypedArray,
} from './types.js';

/**
 * Decode one or more sub-ranges from a single data object.
 *
 * @param buf         - Wire-format message bytes.
 * @param objectIndex - Zero-based index of the target object.
 * @param ranges      - Array of `[offset, count]` pairs in element units.
 *                      `number` and `bigint` are both accepted; values
 *                      above `Number.MAX_SAFE_INTEGER` must be `bigint`.
 * @param opts        - `verifyHash` and `join` (see {@link DecodeRangeOptions}).
 * @returns           - `{ descriptor, parts }`.  One entry in `parts`
 *                      per requested range (split mode — the default), or
 *                      a single concatenated entry when `join: true`.
 * @throws {ObjectError}    when `objectIndex` is out of range
 * @throws {EncodingError}  when the object has a filter (e.g. shuffle)
 *                          or a bitmask dtype — both unsupported by the
 *                          underlying range API.
 * @throws {HashMismatchError} if `verifyHash: true` and a payload hash
 *                             does not match the descriptor's recorded hash.
 * @throws {InvalidArgumentError} on malformed `ranges`.
 */
export function decodeRange(
  buf: Uint8Array,
  objectIndex: number,
  ranges: readonly RangePair[],
  opts?: DecodeRangeOptions,
): DecodeRangeResult {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`buf must be a Uint8Array, got ${typeof buf}`);
  }
  // WASM's `usize` is 32-bit on wasm32 — anything beyond u32 would be
  // silently truncated by the boundary, producing incorrect results.
  if (!Number.isInteger(objectIndex) || objectIndex < 0 || objectIndex > 0xffff_ffff) {
    throw new InvalidArgumentError(
      `objectIndex must be an unsigned 32-bit integer, got ${String(objectIndex)}`,
    );
  }
  if (!Array.isArray(ranges)) {
    throw new InvalidArgumentError('ranges must be an array of [offset, count] pairs');
  }

  const flat = flattenRanges(ranges);
  const wbg = getWbg();
  const result = rethrowTyped(
    () =>
      wbg.decode_range(buf, objectIndex, flat, opts?.verifyHash ?? false) as {
        descriptor: DataObjectDescriptor;
        parts: Uint8Array[];
      },
  );

  const rawParts = result.parts;
  if (opts?.join) {
    const joined = concat(rawParts);
    return {
      descriptor: result.descriptor,
      parts: [typedArrayFor(result.descriptor.dtype, joined, /* copy */ true)],
    };
  }

  const typedParts: TypedArray[] = rawParts.map((p) =>
    typedArrayFor(result.descriptor.dtype, p, /* copy */ true),
  );
  return { descriptor: result.descriptor, parts: typedParts };
}

/**
 * Pack `ranges` into the flat `BigUint64Array` expected by the WASM
 * boundary: `[off0, count0, off1, count1, ...]`.  Rejects negative /
 * fractional values; promotes `number` to `bigint` losslessly within
 * safe-integer range and lets `bigint` inputs flow through.
 */
function flattenRanges(ranges: readonly RangePair[]): BigUint64Array {
  const out = new BigUint64Array(ranges.length * 2);
  for (let i = 0; i < ranges.length; i++) {
    const pair = ranges[i];
    if (!Array.isArray(pair) || pair.length !== 2) {
      throw new InvalidArgumentError(
        `ranges[${i}] must be a [offset, count] pair, got ${JSON.stringify(pair)}`,
      );
    }
    out[2 * i] = toBigUint64(pair[0], i, 'offset');
    out[2 * i + 1] = toBigUint64(pair[1], i, 'count');
  }
  return out;
}

/** 2^64 − 1 — the inclusive upper bound of an unsigned 64-bit integer. */
const U64_MAX = 0xffff_ffff_ffff_ffffn;

function toBigUint64(v: number | bigint, i: number, field: string): bigint {
  if (typeof v === 'bigint') {
    if (v < 0n || v > U64_MAX) {
      throw new InvalidArgumentError(
        `ranges[${i}].${field} must fit in an unsigned 64-bit integer, got ${v}`,
      );
    }
    return v;
  }
  if (typeof v === 'number') {
    if (!Number.isFinite(v) || v < 0 || !Number.isInteger(v)) {
      throw new InvalidArgumentError(
        `ranges[${i}].${field} must be a non-negative integer, got ${v}`,
      );
    }
    if (v > Number.MAX_SAFE_INTEGER) {
      throw new InvalidArgumentError(
        `ranges[${i}].${field} exceeds Number.MAX_SAFE_INTEGER — pass a bigint`,
      );
    }
    return BigInt(v);
  }
  throw new InvalidArgumentError(
    `ranges[${i}].${field} must be a number or bigint, got ${typeof v}`,
  );
}

/** Concatenate a list of byte arrays into a single `Uint8Array`. */
function concat(parts: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

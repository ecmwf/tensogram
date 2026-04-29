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
import { concatBytes, flattenRangePairs } from './internal/rangePack.js';
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
 * Decode is a pure deserialisation in v3 — per-frame integrity
 * verification has moved to the validation layer.  Use
 * {@link validate} with `level: "checksum"` for the equivalent of
 * the legacy `verifyHash` option (which has been removed).
 *
 * @param buf         - Wire-format message bytes.
 * @param objectIndex - Zero-based index of the target object.
 * @param ranges      - Array of `[offset, count]` pairs in element units.
 *                      `number` and `bigint` are both accepted; values
 *                      above `Number.MAX_SAFE_INTEGER` must be `bigint`.
 * @param opts        - See {@link DecodeRangeOptions}.
 * @returns           - `{ descriptor, parts }`.  One entry in `parts`
 *                      per requested range (split mode — the default), or
 *                      a single concatenated entry when `join: true`.
 * @throws {ObjectError}    when `objectIndex` is out of range
 * @throws {EncodingError}  when the object has a filter (e.g. shuffle)
 *                          or a bitmask dtype — both unsupported by the
 *                          underlying range API.
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

  const flat = flattenRangePairs(ranges);
  const wbg = getWbg();
  const result = rethrowTyped(
    () =>
      wbg.decode_range(buf, objectIndex, flat) as {
        descriptor: DataObjectDescriptor;
        parts: Uint8Array[];
      },
  );

  // The WASM side already materialised each part via `Uint8Array::from`,
  // which copies the bytes from WASM linear memory into a JS-heap
  // `Uint8Array`.  We pass `copy: false` to `typedArrayFor` so it can
  // view those bytes zero-copy — an aligned copy is still made
  // internally when `byteOffset % element_width != 0`, which is rare
  // (the WASM side allocates at offset 0).
  const rawParts = result.parts;
  if (opts?.join) {
    const joined = concatBytes(rawParts);
    return {
      descriptor: result.descriptor,
      parts: [typedArrayFor(result.descriptor.dtype, joined, /* copy */ false)],
    };
  }

  const typedParts: TypedArray[] = rawParts.map((p) =>
    typedArrayFor(result.descriptor.dtype, p, /* copy */ false),
  );
  return { descriptor: result.descriptor, parts: typedParts };
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Range-pair packing helpers used by both buffer-level `decodeRange`
 * and the `messageObjectRange` per-message path.
 *
 * `@internal` — not re-exported from `index.ts`.
 */

import { InvalidArgumentError } from '../errors.js';
import type { RangePair } from '../types.js';

/** 2^64 − 1 — the inclusive upper bound of an unsigned 64-bit integer. */
const U64_MAX = 0xffff_ffff_ffff_ffffn;

/**
 * Pack `ranges` into the flat `BigUint64Array` expected by the WASM
 * boundary: `[off0, count0, off1, count1, ...]`.  Rejects negative /
 * fractional values; promotes `number` to `bigint` losslessly within
 * safe-integer range and lets `bigint` inputs flow through.
 */
export function flattenRangePairs(ranges: readonly RangePair[]): BigUint64Array {
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
export function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
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

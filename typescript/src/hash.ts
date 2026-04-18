// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `computeHash` — standalone hash computation over arbitrary bytes.
 *
 * Mirrors `tgm_compute_hash` / `tensogram::compute_hash`.  Used with
 * {@link encodePreEncoded} to stamp a precomputed hash, and by
 * application code that wants to check a payload before handing it off.
 *
 * Returns the hex digest as a lowercase `string`, matching the shape
 * stored in the wire-format `HashDescriptor.value` field.
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';

/** Supported hash algorithm names.  `"xxh3"` is the only one today. */
export type HashAlgorithm = 'xxh3';

/**
 * Compute a hash of the supplied bytes.
 *
 * @param data - Input bytes.  Must be a `Uint8Array`.
 * @param algo - Algorithm name; default `"xxh3"`.
 * @returns    - Lower-case hex string (16 chars for `xxh3`).
 * @throws {InvalidArgumentError} when `data` is not a `Uint8Array`.
 * @throws {MetadataError} when `algo` is not a recognised algorithm name.
 */
export function computeHash(data: Uint8Array, algo: HashAlgorithm = 'xxh3'): string {
  if (!(data instanceof Uint8Array)) {
    throw new InvalidArgumentError(`data must be a Uint8Array, got ${typeof data}`);
  }
  const wbg = getWbg();
  return rethrowTyped(() => wbg.compute_hash(data, algo));
}

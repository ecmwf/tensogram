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

/**
 * Read the per-object inline hash slots from a wire-format message.
 *
 * Mirrors the Rust core `tensogram::data_object_inline_hashes` (and the
 * Python `object_inline_hashes` / C FFI `tgm_object_hash_value`): walks
 * the frame chain of a **single** message and collects the inline hash
 * slot of every data-object frame, in emission order.
 *
 * In v3 the per-object digest lives in the frame footer, not the CBOR
 * descriptor (`plans/WIRE_FORMAT.md` §2.4), so this is the working
 * accessor for the integrity digests that `DataObjectDescriptor.hash`
 * cannot reach.
 *
 * @param buf - Wire-format message bytes (a single message, not a file).
 * @returns   - One entry per data object, in order: a 16-character
 *              lowercase xxh3-64 hex string (identical in form to the
 *              output of {@link computeHash}) when the frame carries a
 *              non-zero digest, or `null` when the slot is empty (the
 *              message was encoded with hashing disabled, or that frame
 *              opted out).
 * @throws {InvalidArgumentError} when `buf` is not a `Uint8Array`.
 * @throws {FramingError} when `buf` is not a structurally valid message
 *   (truncated, bad frame header, length overflow).
 */
export function dataObjectInlineHashes(buf: Uint8Array): (string | null)[] {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`buf must be a Uint8Array, got ${typeof buf}`);
  }
  const wbg = getWbg();
  return rethrowTyped(() => wbg.data_object_inline_hashes(buf) as (string | null)[]);
}

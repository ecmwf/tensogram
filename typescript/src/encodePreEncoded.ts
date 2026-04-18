// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `encodePreEncoded` — wrap already-encoded payloads into a Tensogram
 * message without running the encoding pipeline.
 *
 * Mirrors Rust `encode_pre_encoded`, Python `encode_pre_encoded`, and
 * FFI `tgm_encode_pre_encoded`.  The library still validates the
 * descriptor structure and any szip block offsets it finds; the hash
 * (when enabled) is always recomputed from the caller's bytes so
 * descriptor-supplied hashes are overwritten.
 *
 * Use case: callers that have already run encoding/compression via
 * another path (e.g. GPU pipeline, another Tensogram implementation,
 * cached pre-packed tiles) and want to wrap the bytes in a wire-format
 * message.
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';
import {
  assertValidDescriptor,
  assertValidMetadata,
} from './internal/validation.js';
import type {
  EncodePreEncodedOptions,
  GlobalMetadata,
  PreEncodedInput,
} from './types.js';

/**
 * Encode global metadata + pre-encoded object bytes into a single
 * Tensogram wire-format message.
 *
 * @param metadata - Global metadata (`version: 2` required).
 * @param objects  - `(descriptor, bytes)` pairs.  Each `data` is the
 *                   output of a prior encoding pipeline matching the
 *                   descriptor's `encoding`/`filter`/`compression`.
 * @param options  - `hash: "xxh3" | false`.  Default `"xxh3"`.
 * @returns        - `Uint8Array` wire-format message.
 * @throws {MetadataError}     if metadata or any descriptor is malformed
 * @throws {InvalidArgumentError} on bad argument types
 */
export function encodePreEncoded(
  metadata: GlobalMetadata,
  objects: readonly PreEncodedInput[],
  options?: EncodePreEncodedOptions,
): Uint8Array {
  assertValidMetadata(metadata);
  assertValidPreEncodedObjects(objects);

  const wbg = getWbg();
  const objArray = objects.map((o) => ({ descriptor: o.descriptor, data: o.data }));
  const hash = options?.hash !== false;
  return rethrowTyped(() => wbg.encode_pre_encoded(metadata, objArray, hash));
}

function assertValidPreEncodedObjects(objects: readonly PreEncodedInput[]): void {
  if (!Array.isArray(objects)) {
    throw new InvalidArgumentError(`objects must be an array, got ${typeof objects}`);
  }
  for (let i = 0; i < objects.length; i++) {
    const o = objects[i];
    if (o === null || typeof o !== 'object') {
      throw new InvalidArgumentError(
        `objects[${i}] must be a { descriptor, data } pair`,
      );
    }
    assertValidDescriptor(o.descriptor, i);
    if (!(o.data instanceof Uint8Array)) {
      throw new InvalidArgumentError(
        `objects[${i}].data must be a Uint8Array of pre-encoded bytes`,
      );
    }
  }
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Decode wrapper.
 *
 * Produces a `DecodedMessage` with dtype-aware `data()` / `dataView()`
 * methods on each object. Manages `free()` through an explicit
 * `close()` call and a best-effort `FinalizationRegistry` fallback.
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';
import { wrapWbgDecodedMessage, type WbgDecodedMessage } from './internal/wbgWrap.js';
import type {
  DecodedMessage,
  DecodeOptions,
  GlobalMetadata,
  MessagePosition,
} from './types.js';

/**
 * Decode all objects from a complete Tensogram message.
 *
 * Decode is a pure deserialisation by default.  Pass
 * `opts.verifyHash = true` to fold per-frame xxh3 verification
 * into the decode pass — see {@link DecodeOptions.verifyHash} for
 * the failure modes.
 *
 * @throws {FramingError}   if the buffer is not a valid message
 * @throws {MetadataError}  if CBOR metadata is malformed
 * @throws {CompressionError} on decompression failure
 * @throws {MissingHashError} on `verifyHash=true` + flag clear
 * @throws {HashMismatchError} on `verifyHash=true` + slot disagrees
 */
export function decode(buf: Uint8Array, opts?: DecodeOptions): DecodedMessage {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  const handle = rethrowTyped(() =>
    wbg.decode(
      buf,
      opts?.restoreNonFinite ?? true,
      opts?.verifyHash ?? false,
    ) as unknown as WbgDecodedMessage,
  );
  return buildDecodedMessage(handle);
}

/**
 * Decode only the global metadata; does not touch payload bytes.
 */
export function decodeMetadata(buf: Uint8Array): GlobalMetadata {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  return rethrowTyped(() => wbg.decode_metadata(buf) as GlobalMetadata);
}

/**
 * Decode a single object by zero-based index. O(1) seek to the object
 * frame using the message's binary index.
 */
export function decodeObject(
  buf: Uint8Array,
  index: number,
  opts?: DecodeOptions,
): DecodedMessage {
  assertUint8Array(buf, 'buf');
  if (!Number.isInteger(index) || index < 0) {
    throw new InvalidArgumentError(`index must be a non-negative integer, got ${index}`);
  }
  const wbg = getWbg();
  const handle = rethrowTyped(() =>
    wbg.decode_object(
      buf,
      index,
      opts?.restoreNonFinite ?? true,
      opts?.verifyHash ?? false,
    ) as unknown as WbgDecodedMessage,
  );
  return buildDecodedMessage(handle);
}

/**
 * Scan a buffer for concatenated Tensogram messages.
 *
 * Returns positions for each valid message. Garbage between messages is
 * silently skipped. The scanner tolerates corruption — a single bad
 * message does not abort the scan.
 */
export function scan(buf: Uint8Array): MessagePosition[] {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  // wbg.scan returns a JS array of [offset, length] pairs.
  const raw = rethrowTyped(() => wbg.scan(buf) as Array<[number, number]>);
  return raw.map(([offset, length]) => ({ offset, length }));
}

// ── internals ──────────────────────────────────────────────────────────────

function buildDecodedMessage(handle: WbgDecodedMessage): DecodedMessage {
  return wrapWbgDecodedMessage(handle);
}

function assertUint8Array(buf: unknown, name: string): asserts buf is Uint8Array {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`${name} must be a Uint8Array, got ${typeof buf}`);
  }
}

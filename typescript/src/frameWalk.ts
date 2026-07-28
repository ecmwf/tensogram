// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Structural introspection of a message: walk its frames and read its
 * envelope, without decoding payloads or CBOR.
 *
 * Mirror the Rust core `tensogram::frames` / `tensogram::message_header`
 * (`rust/tensogram/src/frame_walk.rs`):
 *
 * - {@link messageHeader} reads the 24-byte preamble as typed values —
 *   wire version, total length, and eight structural predicates.  Tells
 *   you whether a message is random-access (metadata / index / hashes
 *   in the *header*) or streaming (in the *footer*) without touching a
 *   single frame.
 * - {@link frames} walks the message's frames (types 1–9).  The
 *   preamble and postamble are **not** frames and are never returned.
 *
 * ## One message per call
 *
 * Both functions take the bytes of a **single** message, starting at
 * its `TENSOGRM` preamble magic.  A `.tgm` file is a concatenation of
 * messages, so callers holding a whole file must locate the boundaries
 * with {@link scan} and slice first:
 *
 * ```typescript
 * for (const { offset, length } of scan(fileBytes)) {
 *   const message = fileBytes.subarray(offset, offset + length);
 *   for (const frame of frames(message)) {
 *     // frame.offset is relative to `message`, not to `fileBytes`
 *     console.log(frame.frameType, offset + frame.offset, frame.length);
 *   }
 * }
 * ```
 */

import { getWbg } from './init.js';
import { InvalidArgumentError, rethrowTyped } from './errors.js';
import { safeNumberFromBigint } from './internal/layout.js';
import type { Frame, FrameTypeName, MessageHeader } from './types.js';

/** Shape of one entry in the wasm `frames()` array. */
interface WbgFrame {
  frame_type: FrameTypeName;
  frame_type_code: number;
  version: number;
  flags: number;
  offset: number;
  length: number;
  payload: Uint8Array;
  has_hash: boolean;
}

/** Shape of the wasm `message_header()` result. */
interface WbgMessageHeader {
  version: number;
  total_length: number | bigint;
  has_header_metadata: boolean;
  has_footer_metadata: boolean;
  has_header_index: boolean;
  has_footer_index: boolean;
  has_header_hashes: boolean;
  has_footer_hashes: boolean;
  has_preceder_metadata: boolean;
  has_hashes_present: boolean;
}

/**
 * Walk the frames of a **single** Tensogram message, in wire order.
 *
 * Returns an eagerly-materialised array rather than a generator: a
 * message holds only a handful of frames (the header frames plus one
 * per data object), and the underlying WASM call already materialises
 * the whole walk.  A lazily-pulled iterator would promise a laziness
 * this binding cannot deliver.  The consequence is that a malformed
 * frame chain **throws** instead of yielding a short array — corruption
 * is never silently rendered as "fewer frames".
 *
 * Each {@link Frame.payload} is a copy on the JS heap, not a view into
 * WASM linear memory, so it stays valid across later WASM calls and can
 * be mutated freely (see {@link Frame.payload} for the full contract).
 *
 * @param buf - Wire-format bytes of ONE message, starting at the
 *   `TENSOGRM` preamble magic.  Use {@link scan} on a multi-message
 *   buffer and slice before calling; frame offsets are relative to
 *   `buf`.
 * @returns One {@link Frame} per frame, in wire order.  The preamble
 *   and postamble are not frames and are not included.
 * @throws {InvalidArgumentError} if `buf` is not a `Uint8Array`.
 * @throws {FramingError} if `buf` does not start with a valid preamble,
 *   or if the frame chain is malformed (truncated frame header, a
 *   declared frame length that does not fit the message).
 */
export function frames(buf: Uint8Array): Frame[] {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  const raw = rethrowTyped(() => wbg.frames(buf) as unknown as WbgFrame[]);
  return raw.map((f) => ({
    frameType: f.frame_type,
    frameTypeCode: f.frame_type_code,
    version: f.version,
    flags: f.flags,
    offset: f.offset,
    length: f.length,
    payload: f.payload,
    hasHash: f.has_hash,
  }));
}

/**
 * Read a **single** message's envelope without walking its frames.
 *
 * The eight `has*` predicates are exact for a buffered message (one
 * produced by {@link encode}): the encoder knows the whole message up
 * front, so each flag is a precise statement about the frames present.
 * A {@link StreamingEncoder} writes the preamble before any object, so
 * there the flags are advisory — only `frame present ⇒ flag set` holds
 * (the encoder sets `PRECEDER_METADATA` optimistically, and
 * {@link MessageHeader.totalLength} may stay `0` when the sink was not
 * seekable at `finish()` time).
 *
 * @param buf - Wire-format bytes of ONE message, starting at the
 *   `TENSOGRM` preamble magic.  Use {@link scan} on a multi-message
 *   buffer and slice before calling.
 * @throws {InvalidArgumentError} if `buf` is not a `Uint8Array`, or if
 *   the message advertises a length above `Number.MAX_SAFE_INTEGER`.
 * @throws {FramingError} if `buf` does not start with a valid preamble.
 */
export function messageHeader(buf: Uint8Array): MessageHeader {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  const h = rethrowTyped(() => wbg.message_header(buf) as unknown as WbgMessageHeader);
  return {
    version: h.version,
    totalLength: safeNumberFromBigint(h.total_length, 'messageHeader.totalLength'),
    hasHeaderMetadata: h.has_header_metadata,
    hasFooterMetadata: h.has_footer_metadata,
    hasHeaderIndex: h.has_header_index,
    hasFooterIndex: h.has_footer_index,
    hasHeaderHashes: h.has_header_hashes,
    hasFooterHashes: h.has_footer_hashes,
    hasPrecederMetadata: h.has_preceder_metadata,
    hasHashesPresent: h.has_hashes_present,
  };
}

function assertUint8Array(buf: unknown, name: string): asserts buf is Uint8Array {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`${name} must be a Uint8Array, got ${typeof buf}`);
  }
}

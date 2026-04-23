// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Wire-format layout types and helpers.
 *
 * `@internal` — not re-exported from `index.ts`.
 *
 * `MessageLayout` mirrors the per-message state cached by Rust's
 * `remote.rs`: the preamble flags, the (optionally populated)
 * `GlobalMetadata`, and the (optionally populated) index frame that
 * maps object indices to their `(offset, length)` inside the message.
 * Populated lazily — an opened lazy-HTTP backend starts with the
 * preamble only and fetches metadata/index on demand via
 * [`ensureLayoutForMessage`] (implemented in `file.ts`).
 */

import { InvalidArgumentError } from '../errors.js';
import type { DataObjectDescriptor, GlobalMetadata } from '../types.js';

/** Flags from the message preamble. */
export interface PreambleInfo {
  version: number;
  flags: number;
  totalLength: number;
  hasHeaderMetadata: boolean;
  hasHeaderIndex: boolean;
  hasFooterMetadata: boolean;
  hasFooterIndex: boolean;
  hasPrecederMetadata: boolean;
  hashesPresent: boolean;
}

/** Fields from the message postamble. */
export interface PostambleInfo {
  firstFooterOffset: number;
  totalLength: number;
  endMagicOk: boolean;
}

/** The decoded index frame, telling us where each data-object frame lives. */
export interface FrameIndex {
  offsets: number[];
  lengths: number[];
}

/** Per-message cached state held by the lazy-HTTP backend. */
export interface MessageLayout {
  offset: number;
  length: number;
  preamble: PreambleInfo;
  metadata?: GlobalMetadata;
  index?: FrameIndex;
  descriptors?: DataObjectDescriptor[];
}

/**
 * Normalise a `bigint` or `number` returned from WASM into a JS
 * `number` usable as a file position.  Throws
 * [`InvalidArgumentError`] for values above `Number.MAX_SAFE_INTEGER`.
 *
 * WASM uses u64 for every file position, and
 * `serde_wasm_bindgen::Serializer::json_compatible()` demotes values
 * below `2^53 - 1` to JS `number` but leaves larger values as
 * `BigInt`.  TensogramFile file positions are `number`-based
 * throughout (JS cursor arithmetic cannot safely exceed 2^53 - 1), so
 * we reject the out-of-range case with a clear error rather than
 * silently truncate.
 */
export function safeNumberFromBigint(v: number | bigint, context: string): number {
  if (typeof v === 'number') {
    if (!Number.isFinite(v) || !Number.isInteger(v) || v < 0) {
      throw new InvalidArgumentError(`${context}: not a non-negative integer (${v})`);
    }
    if (v > Number.MAX_SAFE_INTEGER) {
      throw new InvalidArgumentError(
        `${context}: ${v} exceeds MAX_SAFE_INTEGER (${Number.MAX_SAFE_INTEGER})`,
      );
    }
    return v;
  }
  if (typeof v === 'bigint') {
    if (v < 0n) {
      throw new InvalidArgumentError(`${context}: negative bigint (${v})`);
    }
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new InvalidArgumentError(
        `${context}: bigint ${v} exceeds MAX_SAFE_INTEGER (${Number.MAX_SAFE_INTEGER}); ` +
          'file sizes above 9 PB must be processed via the Rust or Python bindings',
      );
    }
    return Number(v);
  }
  throw new InvalidArgumentError(`${context}: expected number or bigint, got ${typeof v}`);
}

/** WASM-side PreambleInfo shape (fields arrive as number or bigint). */
interface WbgPreambleInfo {
  version: number;
  flags: number;
  total_length: number | bigint;
  has_header_metadata: boolean;
  has_header_index: boolean;
  has_footer_metadata: boolean;
  has_footer_index: boolean;
  has_preceder_metadata: boolean;
  hashes_present: boolean;
}

/** Convert WASM PreambleInfo into its normalised TS form. */
export function normalisePreambleInfo(raw: unknown): PreambleInfo {
  if (!raw || typeof raw !== 'object') {
    throw new InvalidArgumentError('PreambleInfo: expected an object from WASM');
  }
  const r = raw as WbgPreambleInfo;
  return {
    version: r.version,
    flags: r.flags,
    totalLength: safeNumberFromBigint(r.total_length, 'preamble.total_length'),
    hasHeaderMetadata: r.has_header_metadata,
    hasHeaderIndex: r.has_header_index,
    hasFooterMetadata: r.has_footer_metadata,
    hasFooterIndex: r.has_footer_index,
    hasPrecederMetadata: r.has_preceder_metadata,
    hashesPresent: r.hashes_present,
  };
}

/** Fields from a data-object frame footer. */
export interface FrameFooterInfo {
  cborOffset: number;
  endMagicOk: boolean;
}

/** WASM-side PostambleInfo shape. */
interface WbgPostambleInfo {
  first_footer_offset: number | bigint;
  total_length: number | bigint;
  end_magic_ok: boolean;
}

/** WASM-side DataObjectFooter shape. */
interface WbgFrameFooterInfo {
  cbor_offset: number | bigint;
  hash_hex: string;
  end_magic_ok: boolean;
}

/** Convert WASM data-object frame footer into its normalised TS form. */
export function normaliseFrameFooter(raw: unknown): FrameFooterInfo {
  if (!raw || typeof raw !== 'object') {
    throw new InvalidArgumentError('FrameFooter: expected an object from WASM');
  }
  const r = raw as WbgFrameFooterInfo;
  return {
    cborOffset: safeNumberFromBigint(r.cbor_offset, 'frame.cbor_offset'),
    endMagicOk: r.end_magic_ok,
  };
}

/** Convert WASM PostambleInfo into its normalised TS form. */
export function normalisePostambleInfo(raw: unknown): PostambleInfo {
  if (!raw || typeof raw !== 'object') {
    throw new InvalidArgumentError('PostambleInfo: expected an object from WASM');
  }
  const r = raw as WbgPostambleInfo;
  return {
    firstFooterOffset: safeNumberFromBigint(r.first_footer_offset, 'postamble.first_footer_offset'),
    totalLength: safeNumberFromBigint(r.total_length, 'postamble.total_length'),
    endMagicOk: r.end_magic_ok,
  };
}

/** WASM-side FrameIndex shape — offsets/lengths arrive as BigUint64Array. */
interface WbgFrameIndex {
  offsets: BigUint64Array | Array<number | bigint>;
  lengths: BigUint64Array | Array<number | bigint>;
}

/** Convert WASM FrameIndex into its normalised TS form. */
export function normaliseFrameIndex(raw: unknown): FrameIndex {
  if (!raw || typeof raw !== 'object') {
    throw new InvalidArgumentError('FrameIndex: expected an object from WASM');
  }
  const r = raw as WbgFrameIndex;
  const offsets: number[] = [];
  const lengths: number[] = [];
  for (let i = 0; i < r.offsets.length; i++) {
    offsets.push(safeNumberFromBigint(r.offsets[i], `index.offsets[${i}]`));
  }
  for (let i = 0; i < r.lengths.length; i++) {
    lengths.push(safeNumberFromBigint(r.lengths[i], `index.lengths[${i}]`));
  }
  if (offsets.length !== lengths.length) {
    throw new InvalidArgumentError(
      `FrameIndex: offsets.length (${offsets.length}) != lengths.length (${lengths.length})`,
    );
  }
  return { offsets, lengths };
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Typed error hierarchy mirroring `tensogram_core::error::TensogramError`.
 *
 * The underlying Rust core raises eight variants (Framing, Metadata,
 * Encoding, Compression, Object, Io, Remote, HashMismatch). `tensogram-wasm`
 * converts these into `JsError` instances with a stringly-typed message
 * of the form `"<variant> error: human-readable description"` (or, for
 * hash mismatches, `"hash mismatch: expected <hex>, got <hex>"`).
 *
 * This module parses those messages at the TS boundary and re-throws
 * them as concrete TS classes so consumers can use `instanceof` checks.
 * Two additional TS-only classes — `InvalidArgumentError` and
 * `StreamingLimitError` — cover failures raised inside this wrapper
 * before a WASM call is made.
 */

/** Common base class. All thrown errors extend this. */
export abstract class TensogramError extends Error {
  /**
   * The original message from the WASM boundary, including the variant
   * prefix (e.g. `"framing error: ..."`). Useful for debugging; the
   * user-facing, prefix-free form lives in `message`.
   *
   * When the error is constructed from the TS layer (no WASM prefix to
   * strip), this equals `message`.
   */
  readonly rawMessage: string;

  /**
   * @param message     user-facing message (prefix-free)
   * @param rawMessage  original WASM-boundary string (prefix included).
   *                    Defaults to `message` — TS-layer errors with no
   *                    prefix pass a single argument.
   */
  constructor(message: string, rawMessage: string = message) {
    super(message);
    this.rawMessage = rawMessage;
    // Fix prototype chain for `instanceof` across CJS/ESM boundaries.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** Invalid wire framing: magic bytes, preamble, frame ordering. */
export class FramingError extends TensogramError {
  readonly name = 'FramingError';
}

/** Metadata validation / CBOR parsing failure. */
export class MetadataError extends TensogramError {
  readonly name = 'MetadataError';
}

/** Encoding pipeline failure (e.g. NaN in simple_packing). */
export class EncodingError extends TensogramError {
  readonly name = 'EncodingError';
}

/** Compression or decompression codec failure. */
export class CompressionError extends TensogramError {
  readonly name = 'CompressionError';
}

/** Per-object error: index out of range, shape overflow, etc. */
export class ObjectError extends TensogramError {
  readonly name = 'ObjectError';
}

/** File / stream I/O failure (file not found, permission denied, read/write error). */
export class IoError extends TensogramError {
  readonly name = 'IoError';
}

/** Remote object-store error (S3 / GCS / Azure / HTTP). */
export class RemoteError extends TensogramError {
  readonly name = 'RemoteError';
}

/**
 * Base class for decode-time integrity verification failures.
 *
 * Subclasses surface specific failure modes when the user opts in
 * to verification via `DecodeOptions.verifyHash = true`:
 *
 *   - `HashMismatchError` — the per-frame `HASH_PRESENT` flag is
 *     set and the inline-hash slot disagrees with the recomputed
 *     digest.
 *   - `MissingHashError` — the per-frame `HASH_PRESENT` flag is
 *     clear (no digest was recorded for this frame).
 *
 * Catch this base when you want either kind handled uniformly;
 * use `instanceof` to discriminate.
 *
 * @see {@link plans/DESIGN.md} §"Integrity Hashing"
 * @see {@link plans/WIRE_FORMAT.md} §2.5
 */
export abstract class IntegrityError extends TensogramError {
  /**
   * Zero-based index of the data-object frame whose verification
   * failed.  Always populated by the decode-time path; the
   * offline validator may produce `undefined` for non-object
   * frames (header / footer / index / hash frames).
   */
  readonly objectIndex: number | undefined;

  constructor(message: string, rawMessage: string = message, objectIndex?: number) {
    super(message, rawMessage);
    this.objectIndex = objectIndex;
  }
}

/**
 * Decode-time hash verification was requested and the frame's
 * `HASH_PRESENT` flag is set, but the inline-hash slot disagrees
 * with the recomputed digest of the frame body.  Distinct from
 * {@link MissingHashError}: the digest WAS recorded but doesn't
 * match — typically a transport / storage corruption, or a
 * deliberately tampered frame.
 *
 * `expected` / `actual` carry the 16-character lowercase hex
 * digests as recorded vs. recomputed; comparing them helps tell
 * an off-by-a-byte tamper from full payload corruption.
 *
 * @see {@link plans/WIRE_FORMAT.md} §2.5
 */
export class HashMismatchError extends IntegrityError {
  readonly name = 'HashMismatchError';
  /** Hex-encoded digest stored in the frame's inline slot. */
  readonly expected: string | undefined;
  /** Hex-encoded digest recomputed from the frame body. */
  readonly actual: string | undefined;

  constructor(
    message: string,
    rawMessage: string = message,
    expected?: string,
    actual?: string,
    objectIndex?: number,
  ) {
    super(message, rawMessage, objectIndex);
    this.expected = expected;
    this.actual = actual;
  }
}

/**
 * Decode-time hash verification was requested but the per-frame
 * `HASH_PRESENT` flag is clear — no digest was recorded for that
 * frame.  Distinct from {@link HashMismatchError}: the digest
 * didn't disagree; there was no digest to compare against.
 *
 * Resolutions:
 *   - decode without `verifyHash`, or
 *   - re-encode the source with hashing enabled.
 */
export class MissingHashError extends IntegrityError {
  readonly name = 'MissingHashError';
}

/** Non-WASM-origin error thrown by the TS layer itself (e.g. NULL input, wrong type, pre-init accessor call). */
export class InvalidArgumentError extends TensogramError {
  readonly name = 'InvalidArgumentError';
}

/** Streaming-decoder buffer-limit exceeded. */
export class StreamingLimitError extends TensogramError {
  readonly name = 'StreamingLimitError';
}

/**
 * Maps a raw `Error`/`JsError` message into a concrete `TensogramError`
 * subclass.
 *
 * Rust → WASM errors are produced by `JsError::new(&e.to_string())`;
 * `TensogramError`'s `Display` impl emits `"<variant> error: ..."`
 * (e.g. `"framing error: ..."`) and the special case
 * `"hash mismatch: expected <hex>, got <hex>"`. This function maps
 * those prefixes to the right TS class.
 *
 * Any error that doesn't match a known prefix is wrapped in a
 * `FramingError` as a safe default (every variant has a known prefix
 * today — the catch-all only triggers if the Rust side ever
 * introduces a new variant without updating this mapping).
 *
 * @internal — not re-exported from the package barrel.
 */
export function mapTensogramError(err: unknown): TensogramError {
  const raw = err instanceof Error ? err.message : String(err);

  // Decode-time integrity failures (PLAN_DECODE_HASH_VERIFICATION).
  //
  // The WASM `convert::js_err` helper attaches structured own-
  // properties to the thrown JS Error for these variants:
  //   * `name`        — `"HashMismatchError"` / `"MissingHashError"`
  //   * `objectIndex` — number; absent when unknown for HashMismatch
  //   * `expected`    — 16-char xxh3-64 hex (HashMismatch only)
  //   * `actual`      — 16-char xxh3-64 hex (HashMismatch only)
  //
  // We prefer the structured payload when present and fall back
  // to message-string parsing for any future-variant or
  // non-WASM-origin error that didn't go through `js_err`.
  if (typeof err === 'object' && err !== null) {
    const errAny = err as { name?: unknown; objectIndex?: unknown;
                            expected?: unknown; actual?: unknown };
    if (errAny.name === 'HashMismatchError') {
      const objectIndex =
        typeof errAny.objectIndex === 'number' ? errAny.objectIndex : undefined;
      const expected = typeof errAny.expected === 'string' ? errAny.expected : undefined;
      const actual = typeof errAny.actual === 'string' ? errAny.actual : undefined;
      const stripped = raw.replace(/^hash mismatch:?\s*/, '') || raw;
      return new HashMismatchError(stripped, raw, expected, actual, objectIndex);
    }
    if (errAny.name === 'MissingHashError') {
      const objectIndex =
        typeof errAny.objectIndex === 'number' ? errAny.objectIndex : undefined;
      return new MissingHashError(raw, raw, objectIndex);
    }
  }

  // Fallback: parse the Display string when the structured payload
  // is absent.  Exercised by errors that don't originate from
  // `js_err` — currently none in-tree but future bindings or
  // FFI wrappers may produce them.
  //
  // Rust `HashMismatch` Display:
  //   - "hash mismatch on object N: expected {hex}, got {hex}"
  //     when the offending object index is known.
  //   - "hash mismatch on frame: expected {hex}, got {hex}"
  //     when the underlying error was raised by a context
  //     without a surrounding object (e.g. validator on a
  //     non-data-object frame).
  //
  // Rust `MissingHash` Display:
  //   "hash verification requested but object {N} has no inline hash recorded ..."
  if (raw.startsWith('hash mismatch')) {
    const expectedMatch = /expected[=\s]+([0-9a-fA-F]+)/.exec(raw);
    const actualMatch = /(?:got|actual)[=\s]+([0-9a-fA-F]+)/.exec(raw);
    const indexMatch = /object\s+(\d+)/.exec(raw);
    const objectIndex = indexMatch ? Number.parseInt(indexMatch[1], 10) : undefined;
    const stripped = raw.replace(/^hash mismatch:?\s*/, '') || raw;
    return new HashMismatchError(
      stripped,
      raw,
      expectedMatch?.[1],
      actualMatch?.[1],
      objectIndex,
    );
  }
  if (raw.startsWith('hash verification requested but object')) {
    const indexMatch = /object\s+(\d+)/.exec(raw);
    const objectIndex = indexMatch ? Number.parseInt(indexMatch[1], 10) : undefined;
    return new MissingHashError(raw, raw, objectIndex);
  }

  // Streaming decoder buffer-limit errors are raw JsError strings, not
  // TensogramError::Display. Route them explicitly.
  if (raw.startsWith('streaming buffer would grow') || raw.includes('buffer size overflow')) {
    return new StreamingLimitError(raw);
  }

  // Prefix-driven routing for the standard TensogramError variants.
  // Each tuple is [prefix, class]. The first matching entry wins.
  const table: Array<[string, new (message: string, rawMessage?: string) => TensogramError]> = [
    ['framing error: ', FramingError],
    ['metadata error: ', MetadataError],
    ['encoding error: ', EncodingError],
    ['compression error: ', CompressionError],
    ['object error: ', ObjectError],
    ['io error: ', IoError],
    ['remote error: ', RemoteError],
  ];

  for (const [prefix, ctor] of table) {
    if (raw.startsWith(prefix)) {
      return new ctor(raw.slice(prefix.length), raw);
    }
  }

  // Fallbacks: many wasm-bindgen-generated errors arrive with just the
  // human-readable text (e.g. `"object index 5 out of range (have 2)"`
  // from tensogram-wasm's own JsError calls). Route those by keyword.
  if (/\bindex\b.*\bout of range\b/i.test(raw)) {
    return new ObjectError(raw);
  }
  // Metadata-shaped failures — these keywords appear in Rust error
  // messages like "shape product overflow", "unknown dtype: ...",
  // "invalid byte_order: ...", "missing required descriptor field".
  if (
    /shape|dtype|byte_order|descriptor/i.test(raw)
    && /\berror|invalid|unknown|missing|overflow\b/i.test(raw)
  ) {
    return new MetadataError(raw);
  }

  // Safe default.
  return new FramingError(raw);
}

/**
 * Utility: run a thunk and re-throw any error through `mapTensogramError`.
 * Used by the public wrappers so consumers always see typed errors.
 *
 * @internal — not re-exported from the package barrel.
 */
export function rethrowTyped<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    if (err instanceof TensogramError) throw err;
    throw mapTensogramError(err);
  }
}

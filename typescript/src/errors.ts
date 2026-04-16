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
 * The underlying Rust core raises seven variants (Framing, Metadata,
 * Encoding, Compression, Object, Io, HashMismatch). `tensogram-wasm`
 * converts these into `JsError` instances with a stringly-typed message
 * of the form `"VariantName: human-readable description"`.
 *
 * This module parses those messages at the TS boundary and re-throws
 * them as concrete TS classes so consumers can use `instanceof` checks.
 */

/** Common base class. All thrown errors extend this. */
export abstract class TensogramError extends Error {
  /**
   * The original message from the WASM boundary, including the variant
   * prefix. Useful for debugging.
   */
  readonly rawMessage: string;

  constructor(rawMessage: string, strippedMessage: string) {
    super(strippedMessage);
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

/** File I/O error (only raised from Node file helpers). */
export class IoError extends TensogramError {
  readonly name = 'IoError';
}

/** Remote object-store error (S3 / GCS / Azure / HTTP). */
export class RemoteError extends TensogramError {
  readonly name = 'RemoteError';
}

/** Payload integrity hash mismatch. */
export class HashMismatchError extends TensogramError {
  readonly name = 'HashMismatchError';
  /** Hex-encoded expected digest, when available. */
  readonly expected: string | undefined;
  /** Hex-encoded actual digest, when available. */
  readonly actual: string | undefined;

  constructor(rawMessage: string, strippedMessage: string, expected?: string, actual?: string) {
    super(rawMessage, strippedMessage);
    this.expected = expected;
    this.actual = actual;
  }
}

/** Non-WASM-origin error thrown by the TS layer itself. */
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
 * `TensogramError`'s `Display` impl emits `"framing error: ..."`,
 * `"hash mismatch: expected=..., actual=..."`, etc. This function
 * maps those prefixes to the right TS class.
 *
 * Any error that doesn't match a known prefix is wrapped in a
 * `FramingError` as a safe default (every variant has a known prefix
 * today — the catch-all only triggers if the Rust side ever
 * introduces a new variant without updating this mapping).
 */
export function mapTensogramError(err: unknown): TensogramError {
  const raw = err instanceof Error ? err.message : String(err);

  // Hash mismatches have a structured payload we can pull fields from.
  // Rust formats this as: "hash mismatch: expected {hex}, got {hex}"
  if (raw.startsWith('hash mismatch')) {
    // Accept both "expected X" and "expected=X" for robustness across
    // future error-message tweaks.
    const expectedMatch = /expected[=\s]+([0-9a-fA-F]+)/.exec(raw);
    const actualMatch = /(?:got|actual)[=\s]+([0-9a-fA-F]+)/.exec(raw);
    return new HashMismatchError(
      raw,
      raw,
      expectedMatch?.[1],
      actualMatch?.[1],
    );
  }

  // Streaming decoder buffer-limit errors are raw JsError strings, not
  // TensogramError::Display. Route them explicitly.
  if (raw.startsWith('streaming buffer would grow') || raw.includes('buffer size overflow')) {
    return new StreamingLimitError(raw, raw);
  }

  const table: Array<[string, new (raw: string, stripped: string) => TensogramError]> = [
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
      return new ctor(raw, raw.slice(prefix.length));
    }
  }

  // Fallbacks: many wasm-bindgen-generated errors arrive with just the
  // human-readable text (e.g. `"object index 5 out of range (have 2)"`
  // from tensogram-wasm's own JsError calls). Route those by keyword.
  if (/\bindex\b.*\bout of range\b/i.test(raw)) {
    return new ObjectError(raw, raw);
  }
  if (/shape|dtype|byte_order|descriptor/i.test(raw) && /\berror|invalid|unknown|missing\b/i.test(raw)) {
    return new MetadataError(raw, raw);
  }

  // Safe default.
  return new FramingError(raw, raw);
}

/**
 * Utility: run a thunk and re-throw any error through `mapTensogramError`.
 * Used by the public wrappers so consumers always see typed errors.
 */
export function rethrowTyped<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    if (err instanceof TensogramError) throw err;
    throw mapTensogramError(err);
  }
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Public type definitions for `@ecmwf/tensogram`.
 *
 * These mirror the Rust types in `tensogram-core::types` and the wire
 * format defined in `plans/WIRE_FORMAT.md`. They are hand-written rather
 * than auto-generated from Rust because `ciborium::Value` — which holds
 * free-form metadata — does not map cleanly to any of the existing Rust→TS
 * code-generation tools.
 */

/**
 * Tensor element type.
 *
 * `bitmask` is a sentinel for sub-byte-packed boolean arrays
 * (`ceil(num_elements / 8)` bytes). `float16` / `bfloat16` / `complex64` /
 * `complex128` are represented in JS as `Uint16Array` or interleaved
 * `Float32Array` / `Float64Array` because JS has no native half-precision
 * or complex `TypedArray`.
 */
export type Dtype =
  | 'float16'
  | 'bfloat16'
  | 'float32'
  | 'float64'
  | 'complex64'
  | 'complex128'
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'uint64'
  | 'bitmask';

/** Endianness of payload bytes on the wire. */
export type ByteOrder = 'big' | 'little';

/** Stage 1 of the encoding pipeline. */
export type Encoding = 'none' | 'simple_packing' | (string & {});

/** Stage 2 of the encoding pipeline. */
export type Filter = 'none' | 'shuffle' | (string & {});

/** Stage 3 of the encoding pipeline. */
export type Compression =
  | 'none'
  | 'szip'
  | 'zstd'
  | 'lz4'
  | 'blosc2'
  | 'zfp'
  | 'sz3'
  | (string & {});

/**
 * Recursive CBOR value type.
 *
 * Matches `ciborium::Value` serialised via `Serializer::json_compatible()`.
 * Safe-range `u64` values come across as `number`; values that exceed
 * `Number.MAX_SAFE_INTEGER` come across as `bigint`.
 */
export type CborValue =
  | string
  | number
  | bigint
  | boolean
  | null
  | readonly CborValue[]
  | { readonly [key: string]: CborValue };

/** Per-object payload integrity hash (currently only `xxh3`). */
export interface HashDescriptor {
  /** Hash algorithm identifier. `"xxh3"` today. */
  type: string;
  /** Hex-encoded digest. */
  value: string;
}

/**
 * Data object descriptor (in-wire CBOR under a data object frame).
 *
 * The `encoding`-, `filter`-, and `compression`-specific parameters
 * (e.g. `reference_value`, `bits_per_value`, `zstd_level`,
 * `szip_block_offsets`) are carried as extra top-level keys, so the
 * interface uses an index signature rather than enumerating them.
 */
export interface DataObjectDescriptor {
  /** Object type, `"ntensor"` today. */
  type: string;
  /** Number of dimensions. `0` for a scalar. */
  ndim: number;
  /** Size of each dimension. `[]` for a scalar. */
  shape: readonly number[];
  /** Element stride per dimension. `[]` for a scalar. */
  strides: readonly number[];
  /** Element type. */
  dtype: Dtype;
  /** Endianness of the stored payload bytes. */
  byte_order: ByteOrder;
  /** Encoding stage (see `plans/DESIGN.md`). */
  encoding: Encoding;
  /** Filter stage. */
  filter: Filter;
  /** Compression stage. */
  compression: Compression;
  /** Optional per-object payload integrity hash. */
  hash?: HashDescriptor;
  /**
   * Encoding pipeline parameters, flattened alongside the fixed fields.
   * E.g. `reference_value`, `bits_per_value`, `zstd_level`,
   * `szip_block_offsets`.
   */
  readonly [key: string]: CborValue | readonly CborValue[] | HashDescriptor | string | number | readonly number[] | undefined;
}

/**
 * Per-object metadata entry.
 *
 * Each entry holds ALL structured metadata for that object independently.
 * The library does not interpret keys — e.g. `mars`, `cf`, `grib` are
 * conventions; any key-value pairs are valid.
 *
 * The encoder auto-populates `_reserved_.tensor` (with `ndim`, `shape`,
 * `strides`, `dtype`) in each entry; client code must not write
 * `_reserved_` itself.
 */
export type BaseEntry = {
  readonly [key: string]: CborValue;
};

/**
 * Global metadata carried in the header or footer metadata frame.
 *
 * See `plans/WIRE_FORMAT.md` and `docs/src/format/cbor-metadata.md` for
 * the CBOR schema.
 */
export interface GlobalMetadata {
  /** Wire format version. Currently `2`. */
  version: number;
  /** Per-object metadata, one entry per data object. */
  base?: BaseEntry[];
  /** Library-managed provenance. Do not write. */
  _reserved_?: { readonly [key: string]: CborValue };
  /** Client-writable message-level annotations. */
  _extra_?: { readonly [key: string]: CborValue };
}

/**
 * A single decoded object, with its descriptor and a dtype-aware view
 * onto the payload.
 */
export interface DecodedObject {
  /** The object's descriptor. */
  readonly descriptor: DataObjectDescriptor;
  /**
   * Copy of the payload on the JS heap, typed according to
   * `descriptor.dtype`.
   *
   * Safe across WASM memory growth — the copy outlives the
   * underlying `DecodedMessage`.
   */
  data(): TypedArray;
  /**
   * Zero-copy view into WASM linear memory.
   *
   * **Warning:** the view is invalidated if WASM memory grows, which
   * can happen on the next `encode()` / `decode()` call. Read or
   * copy the bytes before any further WASM call.
   */
  dataView(): TypedArray;
  /** Raw payload byte length (before dtype interpretation). */
  readonly byteLength: number;
}

/** Result of a full `decode()`. */
export interface DecodedMessage {
  readonly metadata: GlobalMetadata;
  readonly objects: readonly DecodedObject[];
  /** Release the underlying WASM memory. Idempotent. */
  close(): void;
}

/**
 * A single decoded object frame from a streaming decoder.
 *
 * Similar to {@link DecodedObject} but additionally carries the
 * `base_entry` — the matching `base[i]` metadata entry from the
 * containing message — so callers can correlate a frame with its
 * per-object MARS / CF / other metadata without re-decoding the
 * message.
 */
export interface DecodedFrame {
  /** The frame's descriptor. */
  readonly descriptor: DataObjectDescriptor;
  /** Per-object metadata from the containing message's `base[i]`, or `null`. */
  readonly baseEntry: BaseEntry | null;
  /** Raw payload byte length. */
  readonly byteLength: number;
  /**
   * Copy of the payload on the JS heap, typed according to
   * `descriptor.dtype`. Safe across WASM memory growth.
   */
  data(): TypedArray;
  /**
   * Zero-copy view into WASM linear memory. Invalidated if WASM memory
   * grows — read or copy immediately.
   */
  dataView(): TypedArray;
  /** Release the underlying WASM memory. Idempotent. */
  close(): void;
}

/** A `(descriptor, data)` pair suitable for `encode()`. */
export interface EncodeInput {
  descriptor: DataObjectDescriptor;
  data: ArrayBufferView;
}

/** Any concrete JavaScript `TypedArray`. */
export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array;

/** A message offset + length pair returned by `scan()`. */
export interface MessagePosition {
  offset: number;
  length: number;
}

/** Options for `encode()`. */
export interface EncodeOptions {
  /**
   * Hash algorithm applied to each object payload. Default `"xxh3"`.
   * Pass `false` to disable hashing entirely.
   */
  hash?: 'xxh3' | false;
}

/** Options for `decode()` / `decodeObject()`. */
export interface DecodeOptions {
  /**
   * If `true`, the decoder verifies every object's payload hash and
   * throws `HashMismatchError` on mismatch. Default `false`.
   */
  verifyHash?: boolean;
}

/** Information supplied to {@link DecodeStreamOptions.onError}. */
export interface StreamDecodeError {
  /** Human-readable message from the most recent corrupt message. */
  message: string;
  /**
   * Running count of skipped messages since the decoder started.
   * Monotonically non-decreasing.
   */
  skippedCount: number;
}

/** Options for {@link decodeStream}. */
export interface DecodeStreamOptions {
  /**
   * Aborts the stream iteration. When fired, the underlying reader
   * is cancelled and the decoder is freed.
   */
  signal?: AbortSignal;
  /**
   * Maximum size of the internal buffer in bytes. When exceeded, the
   * next `feed()` call throws. Default: 256 MiB (matching the WASM side).
   */
  maxBufferBytes?: number;
  /**
   * Invoked whenever the underlying decoder skips a corrupt message.
   * The iterator does NOT throw on corrupt messages — it advances and
   * keeps going. Use this callback to surface the errors to application code.
   */
  onError?: (err: StreamDecodeError) => void;
}

/** Options for {@link TensogramFile.fromUrl}. */
export interface FromUrlOptions {
  /**
   * Custom fetch implementation. Defaults to `globalThis.fetch`.
   * Handy for tests and for browsers with a polyfilled fetch.
   */
  fetch?: typeof globalThis.fetch;
  /** Extra request headers (e.g. auth tokens). */
  headers?: HeadersInit;
  /** AbortSignal passed through to the underlying fetch. */
  signal?: AbortSignal;
}

/** Options for {@link TensogramFile.open}. */
export interface OpenFileOptions {
  /**
   * AbortSignal to cancel the initial file read. Once the file is
   * loaded, further operations are synchronous.
   */
  signal?: AbortSignal;
}

/** Where the bytes backing a {@link TensogramFile} came from. */
export type FileSource = 'local' | 'remote' | 'buffer';

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Public type definitions for `@ecmwf.int/tensogram`.
 *
 * These mirror the Rust types in `tensogram::types` and the wire
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
 * (e.g. `sp_reference_value`, `sp_bits_per_value`, `zstd_level`,
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
   * E.g. `sp_reference_value`, `sp_bits_per_value`, `zstd_level`,
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
 * The CBOR metadata frame is **free-form** — the only library-
 * interpreted top-level sections are `base`, `_reserved_`, and
 * `_extra_`.  Any other key (including a stray legacy `version`)
 * flows into `_extra_` on decode.  The wire-format version lives
 * in the preamble (see `plans/WIRE_FORMAT.md` §3); import
 * {@link WIRE_VERSION} from this package for the compile-time value.
 *
 * On decoded messages, `version` is populated from the preamble and
 * is therefore always `3` in the current library.  It is optional on
 * encode input — if omitted, the preamble supplies it automatically.
 */
export type GlobalMetadata = {
  /**
   * Wire-format version of the message this metadata came from
   * (sourced from the preamble on decode).  Optional on encode input:
   * if provided, it is treated as a free-form annotation and flows
   * into `_extra_`; the real wire version is always
   * {@link WIRE_VERSION}.
   */
  version?: number;
  /** Per-object metadata, one entry per data object. */
  base?: BaseEntry[];
  /** Library-managed provenance. Do not write. */
  _reserved_?: { readonly [key: string]: CborValue };
  /** Client-writable message-level annotations. */
  _extra_?: { readonly [key: string]: CborValue };
} & {
  /**
   * Any other caller-supplied top-level key.  The CBOR metadata
   * frame is free-form (see `plans/WIRE_FORMAT.md` §6.1): anything
   * besides `base`, `_reserved_`, `_extra_`, and the synthetic
   * `version` read-back accessor flows into `_extra_` on decode.
   * Declared here so strict TypeScript configurations accept
   * free-form metadata like `{ foo: 'bar', product: 'efi' }` on
   * encode input.
   */
  readonly [key: string]: CborValue | BaseEntry[] | undefined;
};

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
  /** Global metadata parsed from the message's header metadata frame. */
  readonly metadata: GlobalMetadata;
  /** One entry per data-object frame in the message, in wire order. */
  readonly objects: readonly DecodedObject[];
  /** Release the underlying WASM memory. Idempotent. */
  close(): void;
}

/**
 * Raw per-kind NaN / Inf bitmasks for one object, as returned by
 * {@link decodeWithMasks}.
 *
 * Each mask is a `Uint8Array` of length `n_elements` (the descriptor's
 * shape product), **one byte per tensor element**: `1` marks a position
 * that held the corresponding non-finite value at encode time, `0` marks
 * a finite position.  A kind the frame did not carry is `null`.
 *
 * The three kinds never overlap at the same index — the encoder assigns
 * each element to at most one kind (the complex priority rule, see
 * `plans/WIRE_FORMAT.md` §6.5.2).  Mask arrays are JS-heap copies and are
 * safe to retain after the owning {@link DecodedMaskedMessage} is closed.
 *
 * Mirrors the Rust core `DecodedMaskSet { nan, pos_inf, neg_inf }`
 * (each `Option<Vec<bool>>`).
 */
export interface DecodedMaskSet {
  /** Positions that held NaN at encode time, or `null`. */
  readonly nan: Uint8Array | null;
  /** Positions that held `+Inf`, or `null`. */
  readonly posInf: Uint8Array | null;
  /** Positions that held `-Inf`, or `null`. */
  readonly negInf: Uint8Array | null;
}

/**
 * A single object decoded by {@link decodeWithMasks}: a normal
 * {@link DecodedObject} whose `data()` / `dataView()` expose the
 * `0.0`-substituted payload, plus the raw {@link DecodedMaskSet} for the
 * three non-finite kinds.
 */
export interface DecodedMaskedObject extends DecodedObject {
  /** Raw decompressed NaN / +Inf / -Inf masks for this object. */
  readonly masks: DecodedMaskSet;
}

/**
 * Result of {@link decodeWithMasks}.
 *
 * Like {@link DecodedMessage} but every object additionally carries a
 * {@link DecodedMaskSet}.  The payloads are always `0.0`-substituted
 * (restoration is intentionally not applied — that is the whole point of
 * this advanced API); use the masks to reconstruct or reinterpret the
 * missing positions.
 */
export interface DecodedMaskedMessage {
  /** Global metadata parsed from the message's header metadata frame. */
  readonly metadata: GlobalMetadata;
  /** One entry per data-object frame in the message, in wire order. */
  readonly objects: readonly DecodedMaskedObject[];
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
  /** Shape, dtype, and encoding-pipeline configuration for this object. */
  descriptor: DataObjectDescriptor;
  /**
   * Raw payload bytes in native byte order. Any `ArrayBufferView` is
   * accepted (`TypedArray`, `DataView`, ...); the wrapper normalises
   * to a `Uint8Array` view before the WASM call.
   */
  data: ArrayBufferView;
}

/**
 * Any concrete JavaScript `TypedArray` plus the three Scope-C view
 * classes used for dtypes JS has no native array for:
 *
 * - `Float16Polyfill` (or a native `Float16Array` when the host ships
 *    one) for `float16`.
 * - `Bfloat16Array` for `bfloat16`.
 * - `ComplexArray` for `complex64` / `complex128`.
 *
 * Built-in TypedArrays still cover every other dtype.
 */
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
  | BigUint64Array
  | HalfArrayLike
  | BfloatArrayLike
  | ComplexArrayLike;

/**
 * Lowest-common-denominator structural shape of the `float16` view —
 * every member listed below is present on both the native TC39
 * `Float16Array` and the `Float16Polyfill` from `./float16.ts`.
 * Callers that need polyfill-only accessors (e.g. `.bits`,
 * `.toFloat32Array()`) should check `instanceof Float16Polyfill`
 * explicitly.
 *
 * Declared here rather than imported from `./float16.ts` so
 * `types.ts` stays dependency-free and the module graph stays acyclic.
 */
export interface HalfArrayLike {
  readonly BYTES_PER_ELEMENT: 2;
  readonly length: number;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly buffer: ArrayBufferLike;
  at(index: number): number | undefined;
  [Symbol.iterator](): IterableIterator<number>;
}

/** Structural shape of the `bfloat16` view.  See {@link HalfArrayLike}. */
export type BfloatArrayLike = HalfArrayLike;

/** Structural shape of the `complex64` / `complex128` view. */
export interface ComplexArrayLike {
  readonly dtype: 'complex64' | 'complex128';
  readonly length: number;
  readonly data: Float32Array | Float64Array;
  real(i: number): number;
  imag(i: number): number;
  get(i: number): { re: number; im: number };
  [Symbol.iterator](): IterableIterator<{ re: number; im: number }>;
}

/** A message offset + length pair returned by `scan()`. */
export interface MessagePosition {
  /** Byte offset where the message's preamble begins. */
  offset: number;
  /** Total byte length of the message (preamble through postamble). */
  length: number;
}

/**
 * Canonical wire-format name of a frame type, as spelled in
 * `plans/WIRE_FORMAT.md` §6 and in the Rust `FrameType` enum.
 *
 * Type 4 is reserved (it held the obsolete v2 tensor frame) and never
 * appears — a v3 reader that meets one raises a {@link FramingError}.
 */
export type FrameTypeName =
  | 'HeaderMetadata'
  | 'HeaderIndex'
  | 'HeaderHash'
  | 'FooterHash'
  | 'FooterIndex'
  | 'FooterMetadata'
  | 'PrecederMetadata'
  | 'NTensorFrame';

/**
 * One frame of a message, as returned by {@link frames}.
 *
 * Mirrors the Rust core `tensogram::FrameInfo`.
 */
export interface Frame {
  /** Canonical frame-type name.  See {@link FrameTypeName}. */
  frameType: FrameTypeName;
  /** The frame type's wire number (1–3, 5–9). */
  frameTypeCode: number;
  /** Frame-type-specific version field from the frame header. */
  version: number;
  /**
   * Raw 16-bit frame flags.  Bit 1 is `HASH_PRESENT`, surfaced as
   * {@link Frame.hasHash}.
   */
  flags: number;
  /**
   * Byte offset of the frame header, **relative to the start of the
   * message** passed to {@link frames} — not to a multi-message file.
   */
  offset: number;
  /**
   * Whole-frame span in bytes: the frame header through its `ENDF`
   * magic, excluding any inter-frame alignment padding that follows.
   */
  length: number;
  /**
   * The frame's content: everything between the 16-byte frame header
   * and the type-specific footer (20 bytes for `NTensorFrame`, 12 for
   * every other type).  For an `NTensorFrame` this is the encoded
   * payload plus any mask blobs plus the trailing CBOR descriptor; for
   * every other frame type it is the CBOR body.
   *
   * Always a **copy on the JS heap**, never a view into WASM linear
   * memory: safe to retain across later WASM calls (which may grow that
   * memory) and safe to mutate without touching the caller's buffer.
   * Same ownership contract as the mask arrays from
   * {@link decodeWithMasks}.  Use {@link Frame.offset} /
   * {@link Frame.length} when you want the whole frame instead.
   */
  payload: Uint8Array;
  /**
   * `true` when this frame's `HASH_PRESENT` flag is set, i.e. its hash
   * slot holds a meaningful digest.
   */
  hasHash: boolean;
}

/**
 * A message's envelope (its 24-byte preamble) decoded into typed
 * values, as returned by {@link messageHeader}.
 *
 * Mirrors the Rust core `tensogram::MessageHeader`.
 */
export interface MessageHeader {
  /** Wire-format version — always {@link WIRE_VERSION} in v3. */
  version: number;
  /**
   * Total message length in bytes, preamble through postamble.  `0` for
   * a streaming message whose length was never back-filled.
   */
  totalLength: number;
  /** A `HeaderMetadata` frame is present (random-access mode). */
  hasHeaderMetadata: boolean;
  /** A `FooterMetadata` frame is present (streaming mode). */
  hasFooterMetadata: boolean;
  /** A `HeaderIndex` frame is present. */
  hasHeaderIndex: boolean;
  /** A `FooterIndex` frame is present. */
  hasFooterIndex: boolean;
  /** A `HeaderHash` frame is present. */
  hasHeaderHashes: boolean;
  /** A `FooterHash` frame is present. */
  hasFooterHashes: boolean;
  /** At least one `PrecederMetadata` frame appears in the body. */
  hasPrecederMetadata: boolean;
  /**
   * Advisory: every frame in this message has its per-frame
   * `HASH_PRESENT` bit set.  {@link Frame.hasHash} remains
   * authoritative for any single frame.
   */
  hasHashesPresent: boolean;
}

/**
 * Options for {@link scanWithOptions}.
 *
 * Mirrors the Rust core `framing::ScanOptions`.
 */
export interface ScanOptions {
  /**
   * Enable the meet-in-the-middle walker pair (default `true`).  When
   * `false`, the scanner walks forward from offset 0 only — still
   * correct, just without the hop-count halving on multi-message
   * buffers.
   */
  bidirectional?: boolean;
  /**
   * Upper bound (in bytes) on the `total_length` a postamble may
   * advertise.  Anything larger is treated as corruption and the
   * backward walker yields to the forward walker.  Default `4 GiB`
   * (`4 * 1024 ** 3`).  Lower this for producers that guarantee
   * smaller messages to tighten corruption detection.
   */
  maxMessageSize?: number;
}

/**
 * Mask compression methods recognised by the NaN / Inf companion
 * bitmask frame (wire type 9 `NTensorFrame`, see
 * `plans/WIRE_FORMAT.md` §6.5.1).
 */
export type MaskMethod =
  | 'none'
  | 'rle'
  | 'roaring'
  | 'lz4'
  | 'zstd'
  | 'blosc2';

/**
 * Where the aggregate hash frame is written inside a message.
 *
 * The aggregate frame is a CBOR list of every per-object inline hash
 * slot — a redundant copy that lets tools read all digests at once
 * without walking the body.  The per-frame inline slots always remain
 * authoritative (see {@link dataObjectInlineHashes}).
 *
 * Mirrors the Rust core `AggregateHashPolicy`:
 *
 * - `"auto"` — the encoder picks; buffered {@link encode} writes a
 *   `HeaderHash` frame.  This is the default.
 * - `"none"` — no aggregate frame at all.
 * - `"header"` — a `HeaderHash` frame only.
 * - `"footer"` — a `FooterHash` frame only.
 * - `"both"` — a `HeaderHash` **and** a `FooterHash` frame, carrying
 *   identical hash lists.
 *
 * Ignored when `hash: false`: with per-frame hashing disabled there is
 * nothing to aggregate, so no frame is written whichever policy is
 * requested.  Use {@link frames} to observe the resulting placement.
 */
export type AggregateHashPolicy = 'auto' | 'none' | 'header' | 'footer' | 'both';

/**
 * Which compression implementation the encode pipeline should use.
 *
 * Mirrors the Rust core `CompressionBackend` (`"auto"` consults the
 * `TENSOGRAM_COMPRESSION_BACKEND` environment variable and the platform
 * default; `"ffi"` selects the C libraries — libaec, libzstd; `"pure"`
 * selects the pure-Rust codecs).
 *
 * **Platform no-op in these bindings.**  The WASM bundle is compiled
 * with the pure-Rust codecs only (`lz4`, `szip-pure`, `zstd-pure`);
 * there is no C library to link against in a browser or in Node's WASM
 * sandbox.  Every accepted value therefore produces byte-identical
 * output here.  The knob exists for source symmetry with the Rust,
 * Python, and C bindings — code written against one binding reads the
 * same against another — and is validated (an unrecognised name
 * throws) but never changes behaviour.  The same accepted asymmetry
 * applies to the core's `threads` knob, which the single-threaded WASM
 * build does not expose at all.
 */
export type CompressionBackend = 'auto' | 'ffi' | 'pure';

/** Options for `encode()`. */
export interface EncodeOptions {
  /**
   * Hash algorithm applied to each object payload. Default `"xxh3"`.
   * Pass `false` to disable hashing entirely.
   */
  hash?: 'xxh3' | false;
  /**
   * When `true`, NaN values in float / complex payloads are
   * substituted with `0.0` and recorded in a bitmask companion
   * section of the data-object frame.  When `false` (the default),
   * any NaN in the input is a hard encode error.  See
   * `docs/src/guide/nan-inf-handling.md`.
   */
  allowNan?: boolean;
  /**
   * When `true`, `+Inf` AND `-Inf` are substituted with `0.0` and
   * recorded in per-sign bitmasks.  When `false` (the default), any
   * `±Inf` in the input is a hard encode error.
   */
  allowInf?: boolean;
  /** Compression method for the NaN mask.  Default `"roaring"`. */
  nanMaskMethod?: MaskMethod;
  /** Compression method for the `+Inf` mask.  Default `"roaring"`. */
  posInfMaskMethod?: MaskMethod;
  /** Compression method for the `-Inf` mask.  Default `"roaring"`. */
  negInfMaskMethod?: MaskMethod;
  /**
   * Uncompressed byte-count threshold below which mask blobs are
   * written with method `"none"` regardless of the requested method.
   * Default `128`.  Set to `0` to disable the auto-fallback.
   */
  smallMaskThresholdBytes?: number;
  /**
   * Where to emit the aggregate hash frame.  Default `"auto"`, which
   * this buffered encoder resolves to a `HeaderHash` frame.  See
   * {@link AggregateHashPolicy}.
   */
  aggregateHash?: AggregateHashPolicy;
  /**
   * Which compression implementation to use.  **No-op in the WASM
   * bindings** — this build carries pure-Rust codecs only.  Accepted
   * and validated for source symmetry with the other bindings; see
   * {@link CompressionBackend}.
   */
  compressionBackend?: CompressionBackend;
}

/** Options for `decode()` / `decodeObject()`. */
export interface DecodeOptions {
  /**
   * When `true` (the default), decoded payloads are converted to the
   * host's native byte order regardless of the wire byte order
   * declared in the object descriptor.  Set to `false` to receive the
   * bytes in the message's declared wire `byte_order` — useful for
   * zero-copy forwarding to another consumer that expects the wire
   * order.
   *
   * Note the {@link TypedArray} views returned by `.data()` /
   * `.dataView()` always interpret bytes in the **host's** native
   * order, so a non-native payload read back through a typed view
   * appears byte-swapped; use the raw byte length / an explicit
   * `DataView` when forwarding in wire order.
   *
   * Mirrors the Rust core `DecodeOptions.native_byte_order`.
   */
  nativeByteOrder?: boolean;
  /**
   * When `true` (the default), decode writes canonical NaN / ±Inf
   * bit patterns at positions recorded in the frame's mask companion.
   * Set to `false` to receive the `0.0`-substituted bytes as they
   * are on disk.  Only meaningful for frames produced with
   * `allowNan` / `allowInf` on encode.  See
   * `docs/src/guide/nan-inf-handling.md`.
   */
  restoreNonFinite?: boolean;
  /**
   * When `true`, every data-object frame the decoder touches is
   * verified against its inline xxh3-64 hash.  See
   * `plans/DESIGN.md` §"Integrity Hashing" for the contract.
   *
   * Default `false` — opt-in.  Decode is otherwise a pure
   * deserialisation (the per-frame `HASH_PRESENT` flag and the
   * slot value are both ignored).
   *
   * Throws:
   *   - {@link MissingHashError} when the per-frame
   *     `HASH_PRESENT` flag is clear (no digest recorded).
   *   - {@link HashMismatchError} when the slot disagrees with
   *     the recomputed digest.
   *
   * Both errors carry the offending `objectIndex` and subclass
   * {@link IntegrityError} for unified catch.
   *
   * **Not respected by `decodeRange` / `TensogramFile.decodeRange`:**
   * range decode reads only a slice of the encoded payload, so
   * verifying the whole-frame hash would defeat the optimisation.
   * When integrity matters on a partial-load workflow, use
   * `decodeObject` with `verifyHash=true`.
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
  /**
   * Maximum number of concurrent HTTP Range requests for fan-out
   * operations (`messageObjectBatch`, `prefetchLayouts`, descriptor
   * prefix fetches). Default is `6`, matching typical browser
   * per-host connection limits.  Setting `1` forces serial fetches;
   * any positive integer is accepted.
   */
  concurrency?: number;
  /**
   * Enable the bidirectional remote-scan walker on open.  Defaults to
   * `true`.
   *
   * When `true` (the default), `lazyScanMessages` runs a pipelined
   * bidirectional walk: each iteration fetches the next forward
   * preamble, the next backward postamble, AND the previous
   * iteration's candidate-preamble validation in one parallel
   * `Promise.allSettled` round, collapsing the per-round critical
   * path from 2 RTTs to 1 RTT.  On real-network workloads this
   * roughly halves wall-clock for full layout discovery.
   *
   * Set `false` to force a forward-only walk — useful when an
   * adversarial server might serve disagreeing forward and backward
   * reads, or when you specifically want serial Range fetches with
   * `concurrency: 1`.
   *
   * Mirrors the Rust `RemoteScanOptions { bidirectional: true }`
   * default and the Python `bidirectional=True` keyword default.
   *
   * `bidirectional: true` requires `concurrency >= 2` so the paired
   * round can fan out; the default `concurrency` of `6` is fine.
   * `concurrency: 1` is only legal alongside `bidirectional: false`;
   * any other combination rejects the {@link TensogramFile.fromUrl}
   * promise with `InvalidArgumentError` before any HTTP probe.
   *
   * Falls back to the eager full-body GET when the server does not
   * support Range requests, the same as forward-only mode.
   */
  bidirectional?: boolean;
  /**
   * Emit `console.debug` events on every scan-walker state
   * transition (`tensogram:scan:mode`, `tensogram:scan:fallback`,
   * `tensogram:scan:fwd-terminated`, `tensogram:scan:gap-closed`,
   * `tensogram:scan:hop`).  Mirrors the Rust `tracing` events at
   * `target = "tensogram::remote_scan"`.  Default `false`.
   */
  debug?: boolean;
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

// ── Scope-C additions ────────────────────────────────────────────────────────

/**
 * A single (offset, count) pair describing one sub-tensor slab to
 * decode via {@link decodeRange}.  Both values are element counts, not
 * bytes.  `bigint` is accepted for values above 2^53; `number` is fine
 * for everyday sizes.
 */
export type RangePair = readonly [number | bigint, number | bigint];

/** Options for {@link decodeRange}. */
export interface DecodeRangeOptions {
  /**
   * If `true`, the resulting `parts` array has exactly one entry — the
   * concatenation of every requested range.  Default `false` (one entry
   * per range, in request order).
   */
  join?: boolean;
}

/** Result of {@link decodeRange}. */
export interface DecodeRangeResult {
  /** The descriptor of the object being sliced. */
  readonly descriptor: DataObjectDescriptor;
  /**
   * One typed view per requested range (or a single view if `join: true`).
   * The array type reflects `descriptor.dtype`.
   */
  readonly parts: readonly TypedArray[];
}

/** Options for {@link encodePreEncoded}. */
export interface EncodePreEncodedOptions {
  /**
   * Hash algorithm.  Default `"xxh3"`.  Pass `false` to disable.  The
   * hash is always recomputed from the caller's pre-encoded bytes — any
   * `hash` field on the descriptor is ignored.
   */
  hash?: 'xxh3' | false;
}

/** A single (descriptor, pre-encoded bytes) pair for {@link encodePreEncoded}. */
export interface PreEncodedInput {
  descriptor: DataObjectDescriptor;
  /**
   * Pre-encoded bytes matching the descriptor's pipeline declaration.
   * For `encoding: "simple_packing"` this must be the packed-integer
   * output; for `compression: "szip"` it must be the szip-compressed
   * bytes and the descriptor's params must include `szip_block_offsets`.
   */
  data: Uint8Array;
}

/** Simple-packing params produced by {@link simplePackingComputeParams}.
 *
 * Keys use the ``sp_`` prefix to match the on-wire descriptor convention
 * (see ``plans/WIRE_FORMAT.md``) so callers can spread the result
 * straight into a descriptor literal:
 *
 * ```ts
 * const params = await simplePackingComputeParams(values, 16, 0);
 * const desc = { encoding: "simple_packing", shape: [N], dtype: "float64", ...params };
 * ```
 *
 * The encoder also auto-computes these values when the descriptor
 * carries only `sp_bits_per_value` (and optionally
 * `sp_decimal_scale_factor`) — calling this function explicitly is
 * only needed if the caller wants to cache or inspect the derived
 * params across multiple encodes (for example to pin
 * `sp_reference_value` across a time-series).
 */
export interface SimplePackingParams {
  /** First value the packed integer `0` represents. */
  sp_reference_value: number;
  /** Power-of-2 scale (E).  Applied as `2^(-E)` during encode. */
  sp_binary_scale_factor: number;
  /** Power-of-10 scale (D).  Applied as `10^D` during encode. */
  sp_decimal_scale_factor: number;
  /** Width of each packed integer in bits (1–64; 0 for constant fields). */
  sp_bits_per_value: number;
}

/** Validation depth levels, matching the Rust `ValidationLevel` enum. */
export type ValidationLevel = 'structure' | 'metadata' | 'integrity' | 'fidelity';

/** Severity of a {@link ValidationIssue}. */
export type IssueSeverity = 'error' | 'warning';

/**
 * Stable machine-readable issue code.  The full list lives in
 * `rust/tensogram/src/validate/types.rs`; keeping this as `string` keeps
 * the TS surface forward-compatible as new codes are added.
 */
export type IssueCode = string;

/**
 * A single finding from {@link validate} or {@link validateFile}.  The
 * field names mirror the on-the-wire JSON shape emitted by the Rust
 * core's `ValidationReport` (snake_case) so no renaming happens at
 * the WASM boundary.
 */
export interface ValidationIssue {
  /** Stable machine-readable identifier — e.g. `"truncated_message"`. */
  code: IssueCode;
  /** Which depth of validation surfaced the issue. */
  level: ValidationLevel;
  /** Severity: `"error"` fails the message, `"warning"` is informational. */
  severity: IssueSeverity;
  /** Index of the offending object, when applicable. */
  object_index?: number;
  /** Byte offset within the message buffer, when applicable. */
  byte_offset?: number;
  /** Human-readable description suitable for logs or UI. */
  description: string;
}

/** Result of {@link validate}. */
export interface ValidationReport {
  readonly issues: readonly ValidationIssue[];
  readonly object_count: number;
  readonly hash_verified: boolean;
}

/** A file-level issue (not tied to a specific message). */
export interface FileIssue {
  byte_offset: number;
  length: number;
  description: string;
}

/** Result of {@link validateFile}. */
export interface FileValidationReport {
  readonly file_issues: readonly FileIssue[];
  readonly messages: readonly ValidationReport[];
}

/** Human-facing validation mode.  Maps to the CLI's flag set. */
export type ValidateMode = 'quick' | 'default' | 'checksum' | 'full';

/** Options for {@link validate} / {@link validateFile}. */
export interface ValidateOptions {
  /** Validation depth.  Default `"default"`. */
  mode?: ValidateMode;
  /** Enable RFC 8949 canonical-CBOR-ordering checks.  Default `false`. */
  canonical?: boolean;
}

/** Options for {@link TensogramFile#append}. */
export interface AppendOptions {
  /** Hash algorithm.  Default `"xxh3"`.  Pass `false` to disable. */
  hash?: 'xxh3' | false;
  /** See {@link EncodeOptions.allowNan}.  Default `false`. */
  allowNan?: boolean;
  /** See {@link EncodeOptions.allowInf}.  Default `false`. */
  allowInf?: boolean;
  /** See {@link EncodeOptions.nanMaskMethod}. */
  nanMaskMethod?: MaskMethod;
  /** See {@link EncodeOptions.posInfMaskMethod}. */
  posInfMaskMethod?: MaskMethod;
  /** See {@link EncodeOptions.negInfMaskMethod}. */
  negInfMaskMethod?: MaskMethod;
  /** See {@link EncodeOptions.smallMaskThresholdBytes}.  Default `128`. */
  smallMaskThresholdBytes?: number;
  /** See {@link EncodeOptions.aggregateHash}.  Default `"auto"`. */
  aggregateHash?: AggregateHashPolicy;
  /** See {@link EncodeOptions.compressionBackend}.  No-op on WASM. */
  compressionBackend?: CompressionBackend;
}

/** Options for {@link StreamingEncoder}. */
export interface StreamingEncoderOptions {
  /** Hash algorithm.  Default `"xxh3"`.  Pass `false` to disable. */
  hash?: 'xxh3' | false;
  /** See {@link EncodeOptions.allowNan}.  Default `false`. */
  allowNan?: boolean;
  /** See {@link EncodeOptions.allowInf}.  Default `false`. */
  allowInf?: boolean;
  /** See {@link EncodeOptions.nanMaskMethod}. */
  nanMaskMethod?: MaskMethod;
  /** See {@link EncodeOptions.posInfMaskMethod}. */
  posInfMaskMethod?: MaskMethod;
  /** See {@link EncodeOptions.negInfMaskMethod}. */
  negInfMaskMethod?: MaskMethod;
  /** See {@link EncodeOptions.smallMaskThresholdBytes}.  Default `128`. */
  smallMaskThresholdBytes?: number;
  /**
   * Synchronous streaming sink.
   *
   * When set, each chunk of wire-format bytes the encoder produces is
   * forwarded to this callback as it is produced — no full-message
   * buffering is performed.  The callback is invoked during
   * construction (preamble + header metadata frame), during each
   * {@link StreamingEncoder.writeObject} / `writeObjectPreEncoded`
   * (one data-object frame's bytes, potentially split across multiple
   * invocations), and during {@link StreamingEncoder.finish} (footer
   * frames + postamble).
   *
   * In this mode `finish()` returns an empty `Uint8Array` — every
   * byte has already been delivered to the callback.  Concatenating
   * every `chunk` the callback sees (in order) yields a message
   * byte-for-byte identical to the one buffered mode would return.
   *
   * **Synchronous contract.**  The callback must complete its work
   * synchronously.  `Promise` return values are silently discarded
   * because the Rust/WASM writer contract is synchronous.  Use the
   * buffered mode with a single `fetch` call if you need async work.
   *
   * **Chunk ownership.**  Each `chunk` is a fresh JS-owned
   * `Uint8Array` whose buffer is invalidated when the WASM module's
   * linear memory grows.  Copy (`chunk.slice()`) or consume it before
   * the next `writeObject` call if you need to hold on to it.
   *
   * **Errors.**  If the callback throws, the exception surfaces as an
   * `IoError` from the next `writeObject` / `finish` that triggers a
   * flush.  The encoder state is undefined after an error — call
   * {@link StreamingEncoder.close} and start over.
   */
  onBytes?: (chunk: Uint8Array) => void;
}

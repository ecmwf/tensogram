// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `@ecmwf.int/tensogram` — TypeScript bindings for Tensogram, ECMWF's binary
 * message format for N-dimensional scientific tensors.
 *
 * Wraps `rust/tensogram-wasm` (built via `wasm-pack`) with a typed API:
 *
 * - {@link init} — one-time WASM initialisation
 * - {@link encode} — encode a `GlobalMetadata` + objects into wire bytes
 * - {@link decode}, {@link decodeMetadata}, {@link decodeObject}, {@link scan} —
 *   whole-buffer decoding with dtype-aware payload views
 * - {@link decodeStream} — progressive decode over a `ReadableStream<Uint8Array>`
 * - {@link TensogramFile} — random-access file / URL / in-memory reader
 * - {@link getMetaKey}, {@link computeCommon}, {@link cborValuesEqual} —
 *   metadata helpers
 * - {@link typedArrayFor}, {@link payloadByteSize}, {@link shapeElementCount},
 *   {@link DTYPE_BYTE_WIDTH}, {@link SUPPORTED_DTYPES}, {@link isDtype} —
 *   dtype introspection and dispatch
 * - {@link TensogramError} and subclasses — typed error hierarchy
 *
 * See `docs/src/guide/typescript-api.md` for the user guide.
 */

export { init } from './init.js';
export type { InitOptions } from './init.js';

export { encode } from './encode.js';
export { decode, decodeMetadata, decodeObject, scan } from './decode.js';
export { decodeStream } from './streaming.js';
export { TensogramFile } from './file.js';
export { getMetaKey, computeCommon, cborValuesEqual } from './metadata.js';
export { DTYPE_BYTE_WIDTH, payloadByteSize, shapeElementCount, typedArrayFor, isDtype, SUPPORTED_DTYPES } from './dtype.js';

// ── Optional auth helpers ─────────────────────────────────────────────────
export { signAwsV4Request } from './auth/signAwsV4.js';
export type { SigV4Credentials, SigV4Input, SigV4Result } from './auth/signAwsV4.js';
export { createAwsSigV4Fetch } from './auth/awsSigV4Fetch.js';
export type { AwsSigV4FetchOptions } from './auth/awsSigV4Fetch.js';

// ── Scope C.1 — API parity ────────────────────────────────────────────────
export { decodeRange } from './range.js';
export { computeHash } from './hash.js';
export type { HashAlgorithm } from './hash.js';
export { simplePackingComputeParams } from './simplePacking.js';
export { validate, validateBuffer, validateFile } from './validate.js';
export { encodePreEncoded } from './encodePreEncoded.js';
export { StreamingEncoder } from './streamingEncoder.js';

// ── Scope C.2 — first-class half-precision / complex dtypes ───────────────
//
// Raw bit-twiddling helpers (`halfBitsToFloat`, `floatToHalfBits`,
// `bfloat16BitsToFloat`, `floatToBfloat16Bits`) and the internal
// `isComplexDtype` guard are deliberately NOT re-exported here — they
// are implementation details of the view classes.  Callers that need
// bits should reach them through a view's `.bits` accessor; callers
// that need to route on dtype should pattern-match on the descriptor's
// `dtype` string directly.
export {
  Float16Polyfill,
  float16FromBytes,
  getFloat16ArrayCtor,
  hasNativeFloat16Array,
} from './float16.js';
export type { Float16ArrayLike, Float16Ctor } from './float16.js';

export { Bfloat16Array, bfloat16FromBytes } from './bfloat16.js';

export { ComplexArray, complexFromBytes } from './complex.js';
export type { ComplexDtype, ComplexPair, ComplexStorage } from './complex.js';

export {
  TensogramError,
  FramingError,
  MetadataError,
  EncodingError,
  CompressionError,
  ObjectError,
  IoError,
  RemoteError,
  HashMismatchError,
  InvalidArgumentError,
  StreamingLimitError,
} from './errors.js';

export type {
  AppendOptions,
  BaseEntry,
  ByteOrder,
  CborValue,
  Compression,
  DataObjectDescriptor,
  DecodedFrame,
  DecodedMessage,
  DecodedObject,
  DecodeOptions,
  DecodeRangeOptions,
  DecodeRangeResult,
  Dtype,
  EncodeInput,
  EncodeOptions,
  EncodePreEncodedOptions,
  Encoding,
  FileIssue,
  FileSource,
  FileValidationReport,
  Filter,
  FromUrlOptions,
  GlobalMetadata,
  HashDescriptor,
  IssueCode,
  IssueSeverity,
  MaskMethod,
  MessagePosition,
  OpenFileOptions,
  PreEncodedInput,
  RangePair,
  SimplePackingParams,
  StreamDecodeError,
  StreamingEncoderOptions,
  TypedArray,
  ValidateMode,
  ValidateOptions,
  ValidationIssue,
  ValidationLevel,
  ValidationReport,
} from './types.js';

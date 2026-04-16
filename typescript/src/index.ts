// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `@ecmwf/tensogram` — TypeScript bindings for Tensogram, ECMWF's binary
 * message format for N-dimensional scientific tensors.
 *
 * Wraps `rust/tensogram-wasm` (built via `wasm-pack`) with a typed API:
 *
 * - {@link init} — one-time WASM initialisation
 * - {@link encode} — encode a `GlobalMetadata` + objects into wire bytes
 * - {@link decode} — decode wire bytes to a `DecodedMessage` with typed payload views
 * - {@link decodeMetadata}, {@link decodeObject}, {@link scan}
 * - {@link getMetaKey}, {@link computeCommon} — metadata helpers
 * - {@link TensogramError} and subclasses — typed error hierarchy
 *
 * See `docs/src/guide/typescript-api.md` for the user guide and
 * `plans/TYPESCRIPT_WRAPPER.md` for the design doc.
 */

export { init } from './init.js';
export type { InitOptions } from './init.js';

export { encode } from './encode.js';
export { decode, decodeMetadata, decodeObject, scan } from './decode.js';
export { decodeStream } from './streaming.js';
export { TensogramFile } from './file.js';
export { getMetaKey, computeCommon, cborValuesEqual } from './metadata.js';
export { DTYPE_BYTE_WIDTH, payloadByteSize, shapeElementCount, typedArrayFor, isDtype, SUPPORTED_DTYPES } from './dtype.js';

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
  BaseEntry,
  ByteOrder,
  CborValue,
  Compression,
  DataObjectDescriptor,
  DecodedFrame,
  DecodedMessage,
  DecodedObject,
  DecodeOptions,
  DecodeStreamOptions,
  Dtype,
  EncodeInput,
  EncodeOptions,
  Encoding,
  FileSource,
  Filter,
  FromUrlOptions,
  GlobalMetadata,
  HashDescriptor,
  MessagePosition,
  OpenFileOptions,
  StreamDecodeError,
  TypedArray,
} from './types.js';

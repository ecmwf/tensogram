// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Lazy per-object iterators over a single Tensogram message.
 *
 * Mirror the Rust core `tensogram::iter::objects` /
 * `tensogram::iter::objects_metadata` (and the Python `objects` /
 * `objects_metadata` generators): walk the objects of ONE message
 * lazily instead of eagerly materialising the whole
 * {@link decode}`().objects` array.
 *
 * - {@link objects} decodes each object's payload on demand — object
 *   `i + 1` is only decoded when the consumer pulls it.
 * - {@link objectsMetadata} yields the per-object descriptors WITHOUT
 *   decoding any payload (the decompression / filter / endian pipeline
 *   never runs), backed by the cheap wasm `decode_descriptors` parse.
 *
 * Both yield **self-contained** values: the payload bytes are copied to
 * the JS heap and the underlying WASM handle is freed before the value
 * is yielded, so there is no handle to `close()` and nothing is
 * invalidated by a later WASM call.  (`dataView()` therefore views the
 * JS-heap copy rather than WASM linear memory — still zero-copy over
 * that copy.)
 */

import { getWbg } from './init.js';
import { InvalidArgumentError, rethrowTyped } from './errors.js';
import { typedArrayFor } from './dtype.js';
import type {
  DataObjectDescriptor,
  DecodedObject,
  DecodeOptions,
  TypedArray,
} from './types.js';
import type { WbgDecodedMessage } from './internal/wbgWrap.js';

/** Shape of the wasm `decode_descriptors` result. */
interface WbgDescriptors {
  metadata: unknown;
  descriptors: DataObjectDescriptor[];
}

/**
 * Lazily iterate over the decoded objects of a single message.
 *
 * Each object is decoded through the full pipeline (encoding + filter +
 * decompression) only when the consumer requests it, mirroring the Rust
 * core `iter::objects`.  The yielded {@link DecodedObject} is
 * self-contained — its payload has already been copied to the JS heap,
 * so it needs no `.close()` and stays valid across later WASM calls.
 *
 * @param buf  - Wire-format bytes of a single message.
 * @param opts - Decode options; forwarded to each per-object decode.
 *               See {@link DecodeOptions}.
 * @throws {InvalidArgumentError} if `buf` is not a `Uint8Array`.
 * @throws {FramingError}   if the buffer is not a valid message.
 * @throws {MetadataError}  if CBOR metadata is malformed.
 * @throws {CompressionError} on decompression failure.
 */
export function* objects(
  buf: Uint8Array,
  opts?: DecodeOptions,
): Generator<DecodedObject, void, undefined> {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  const restoreNonFinite = opts?.restoreNonFinite ?? true;
  const verifyHash = opts?.verifyHash ?? false;
  const nativeByteOrder = opts?.nativeByteOrder ?? true;

  const count = objectCount(buf);
  for (let i = 0; i < count; i++) {
    // `decode_object` is an O(1) seek to frame `i` via the binary
    // index, then a single-object decode.  We copy the payload out and
    // free the handle immediately so the yielded object is standalone.
    const handle = rethrowTyped(
      () =>
        wbg.decode_object(
          buf,
          i,
          restoreNonFinite,
          verifyHash,
          nativeByteOrder,
        ) as unknown as WbgDecodedMessage,
    );
    try {
      const descriptor = rethrowTyped(
        () => handle.object_descriptor(0) as DataObjectDescriptor,
      );
      const view = rethrowTyped(() => handle.object_data_u8(0));
      // `object_data_u8` is a zero-copy view into WASM memory; copy it
      // onto the JS heap before we free the handle below.
      const bytes = new Uint8Array(view);
      yield selfContainedObject(descriptor, bytes);
    } finally {
      handle.free();
    }
  }
}

/**
 * Lazily iterate over the per-object descriptors of a single message
 * WITHOUT decoding any payload data.
 *
 * Mirrors the Rust core `iter::objects_metadata`: a single cheap parse
 * (`decode_descriptors` — no decompression, no filter reversal) yields
 * every {@link DataObjectDescriptor} in wire order.  Ideal for building
 * an index (shapes, dtypes, encodings) before deciding what to decode.
 *
 * @param buf - Wire-format bytes of a single message.
 * @throws {InvalidArgumentError} if `buf` is not a `Uint8Array`.
 * @throws {FramingError}  if the buffer is not a valid message.
 * @throws {MetadataError} if CBOR metadata is malformed.
 */
export function* objectsMetadata(
  buf: Uint8Array,
): Generator<DataObjectDescriptor, void, undefined> {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  const result = rethrowTyped(() => wbg.decode_descriptors(buf) as unknown as WbgDescriptors);
  for (const descriptor of result.descriptors) {
    yield descriptor;
  }
}

// ── internals ──────────────────────────────────────────────────────────────

/**
 * Cheap object count via `decode_descriptors` (metadata-only parse — no
 * payload decode).
 */
function objectCount(buf: Uint8Array): number {
  const wbg = getWbg();
  const result = rethrowTyped(() => wbg.decode_descriptors(buf) as unknown as WbgDescriptors);
  return result.descriptors.length;
}

/**
 * Build a standalone {@link DecodedObject} over JS-heap `bytes`.  Both
 * `data()` and `dataView()` wrap the same copied buffer, so the object
 * carries no WASM handle and needs no `.close()`.
 */
function selfContainedObject(
  descriptor: DataObjectDescriptor,
  bytes: Uint8Array,
): DecodedObject {
  return {
    descriptor,
    byteLength: bytes.byteLength,
    data(): TypedArray {
      return typedArrayFor(descriptor.dtype, bytes, /* copy */ true);
    },
    dataView(): TypedArray {
      return typedArrayFor(descriptor.dtype, bytes, /* copy */ false);
    },
  };
}

function assertUint8Array(buf: unknown, name: string): asserts buf is Uint8Array {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`${name} must be a Uint8Array, got ${typeof buf}`);
  }
}

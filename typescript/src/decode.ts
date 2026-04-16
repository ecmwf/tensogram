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
import { typedArrayFor } from './dtype.js';
import type {
  DataObjectDescriptor,
  DecodedMessage,
  DecodedObject,
  DecodeOptions,
  GlobalMetadata,
  MessagePosition,
  TypedArray,
} from './types.js';

// wasm-bindgen generates `DecodedMessage` as an opaque class; we treat
// it structurally here and only hold it as an opaque handle.
interface WbgDecodedMessage {
  free(): void;
  object_count(): number;
  object_descriptor(index: number): unknown;
  object_data_u8(index: number): Uint8Array;
  object_byte_length(index: number): number;
}

// Only invoked if the wrapper is GC'd without an explicit close().
const finalizationRegistry = new FinalizationRegistry<WbgDecodedMessage>((handle) => {
  try {
    handle.free();
  } catch {
    // best-effort cleanup
  }
});

/**
 * Decode all objects from a complete Tensogram message.
 *
 * @throws {FramingError}   if the buffer is not a valid message
 * @throws {MetadataError}  if CBOR metadata is malformed
 * @throws {CompressionError} on decompression failure
 * @throws {HashMismatchError} if `opts.verifyHash` is true and a hash doesn't match
 */
export function decode(buf: Uint8Array, opts?: DecodeOptions): DecodedMessage {
  assertUint8Array(buf, 'buf');
  const wbg = getWbg();
  const handle = rethrowTyped(() =>
    wbg.decode(buf, opts?.verifyHash ?? false) as unknown as WbgDecodedMessage,
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
    throw new InvalidArgumentError(
      `index must be a non-negative integer, got ${index}`,
      `index must be a non-negative integer, got ${index}`,
    );
  }
  const wbg = getWbg();
  const handle = rethrowTyped(() =>
    wbg.decode_object(buf, index, opts?.verifyHash ?? false) as unknown as WbgDecodedMessage,
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
  // Hoist the metadata once; .data()/.dataView() pull payload bytes
  // lazily per call, so objects[] is cheap to construct.
  // `metadata()` lives on the wasm-bindgen class; not in our interface,
  // fetch via the full typed handle.
  const wbgHandle = handle as unknown as WbgDecodedMessage & { metadata(): GlobalMetadata };
  const metadata = rethrowTyped(() => wbgHandle.metadata()) as GlobalMetadata;

  const count = handle.object_count();
  const objects: DecodedObject[] = [];
  let closed = false;

  for (let i = 0; i < count; i++) {
    const descriptor = rethrowTyped(
      () => handle.object_descriptor(i) as DataObjectDescriptor,
    );
    const byteLength = handle.object_byte_length(i);
    const idx = i;

    const obj: DecodedObject = {
      descriptor,
      byteLength,
      data(): TypedArray {
        assertOpen(closed);
        const bytes = rethrowTyped(() => handle.object_data_u8(idx));
        return typedArrayFor(descriptor.dtype, bytes, /* copy */ true);
      },
      dataView(): TypedArray {
        assertOpen(closed);
        const bytes = rethrowTyped(() => handle.object_data_u8(idx));
        return typedArrayFor(descriptor.dtype, bytes, /* copy */ false);
      },
    };

    objects.push(obj);
  }

  const msg: DecodedMessage = {
    metadata,
    objects,
    close(): void {
      if (closed) return;
      closed = true;
      finalizationRegistry.unregister(msg);
      try {
        handle.free();
      } catch {
        // best-effort; free is idempotent on our side
      }
    },
  };

  finalizationRegistry.register(msg, handle, msg);
  return msg;
}

function assertOpen(closed: boolean): void {
  if (closed) {
    throw new InvalidArgumentError(
      'decoded message has been closed — payload access is no longer valid',
      'decoded message has been closed — payload access is no longer valid',
    );
  }
}

function assertUint8Array(buf: unknown, name: string): asserts buf is Uint8Array {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(
      `${name} must be a Uint8Array, got ${typeof buf}`,
      `${name} must be a Uint8Array, got ${typeof buf}`,
    );
  }
}

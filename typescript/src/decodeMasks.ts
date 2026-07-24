// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `decodeWithMasks` — advanced decode that also returns the raw
 * NaN / +Inf / -Inf bitmasks alongside the substituted payloads.
 *
 * Mirrors the Rust core `tensogram::decode_with_masks` and the Python
 * `decode_with_masks`.  Unlike {@link decode}, the payloads are always
 * `0.0`-substituted — restoration is intentionally NOT applied — and
 * each object carries a {@link DecodedMaskSet}.  Use this when you want
 * to apply your own missing-value representation (e.g. `NaN`, a sentinel
 * fill value, or a boolean mask array) rather than the canonical
 * NaN / ±Inf bit patterns {@link decode} restores.
 *
 * See `docs/src/guide/nan-inf-handling.md` for the companion guide.
 */

import { getWbg } from './init.js';
import { InvalidArgumentError, rethrowTyped } from './errors.js';
import { typedArrayFor } from './dtype.js';
import type {
  DataObjectDescriptor,
  DecodedMaskedMessage,
  DecodedMaskedObject,
  DecodedMaskSet,
  DecodeOptions,
  GlobalMetadata,
  TypedArray,
} from './types.js';

/** Structural shape of the wasm-bindgen `DecodedMaskedMessage` class. */
interface WbgDecodedMaskedMessage {
  free(): void;
  metadata(): unknown;
  object_count(): number;
  object_descriptor(index: number): unknown;
  object_data_u8(index: number): Uint8Array;
  object_byte_length(index: number): number;
  object_mask_nan(index: number): Uint8Array | null;
  object_mask_pos_inf(index: number): Uint8Array | null;
  object_mask_neg_inf(index: number): Uint8Array | null;
}

const finalizationRegistry = new FinalizationRegistry<WbgDecodedMaskedMessage>((handle) => {
  try {
    handle.free();
  } catch {
    // best-effort cleanup
  }
});

/**
 * Decode all objects AND their raw NaN / Inf masks from a complete
 * Tensogram message.
 *
 * The returned payloads (`obj.data()` / `obj.dataView()`) always hold
 * `0.0` at non-finite positions; the companion masks live on
 * `obj.masks` (see {@link DecodedMaskSet} for the byte-per-element
 * representation).  Objects with no mask companion simply have a
 * {@link DecodedMaskSet} of three `null`s.
 *
 * Only {@link DecodeOptions.nativeByteOrder} is honoured here.
 * `restoreNonFinite` is meaningless (this API always returns the
 * substituted payload by design) and `verifyHash` is not applied —
 * decode is a pure deserialisation in v3.
 *
 * The returned {@link DecodedMaskedMessage} owns WASM memory; call
 * `.close()` when done.  A `FinalizationRegistry` frees it best-effort
 * if you forget, but explicit `.close()` is strongly recommended.
 *
 * @throws {FramingError}    if the buffer is not a valid message
 * @throws {MetadataError}   if CBOR metadata is malformed
 * @throws {CompressionError} on decompression failure
 * @throws {InvalidArgumentError} if `buf` is not a `Uint8Array`
 */
export function decodeWithMasks(buf: Uint8Array, opts?: DecodeOptions): DecodedMaskedMessage {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`buf must be a Uint8Array, got ${typeof buf}`);
  }
  const wbg = getWbg();
  const handle = rethrowTyped(
    () =>
      wbg.decode_with_masks(
        buf,
        opts?.nativeByteOrder ?? true,
      ) as unknown as WbgDecodedMaskedMessage,
  );
  return wrapMaskedMessage(handle);
}

// ── internals ──────────────────────────────────────────────────────────────

function wrapMaskedMessage(handle: WbgDecodedMaskedMessage): DecodedMaskedMessage {
  const metadata = rethrowTyped(() => handle.metadata()) as GlobalMetadata;
  const count = handle.object_count();
  const objects: DecodedMaskedObject[] = [];
  let closed = false;

  for (let i = 0; i < count; i++) {
    const descriptor = rethrowTyped(
      () => handle.object_descriptor(i) as DataObjectDescriptor,
    );
    const byteLength = handle.object_byte_length(i);

    // Masks are eagerly materialised as JS-heap `Uint8Array` copies (the
    // WASM accessors return copies via `Uint8Array::from`), so they are
    // safe to retain past `close()` and across WASM memory growth.  A
    // missing kind comes back as `null`.
    const masks: DecodedMaskSet = {
      nan: rethrowTyped(() => handle.object_mask_nan(i)),
      posInf: rethrowTyped(() => handle.object_mask_pos_inf(i)),
      negInf: rethrowTyped(() => handle.object_mask_neg_inf(i)),
    };

    const obj: DecodedMaskedObject = {
      descriptor,
      byteLength,
      masks,
      data(): TypedArray {
        assertOpen(closed);
        const bytes = rethrowTyped(() => handle.object_data_u8(i));
        return typedArrayFor(descriptor.dtype, bytes, /* copy */ true);
      },
      dataView(): TypedArray {
        assertOpen(closed);
        const bytes = rethrowTyped(() => handle.object_data_u8(i));
        return typedArrayFor(descriptor.dtype, bytes, /* copy */ false);
      },
    };

    objects.push(obj);
  }

  const msg: DecodedMaskedMessage = {
    metadata,
    objects,
    close(): void {
      if (closed) return;
      closed = true;
      finalizationRegistry.unregister(msg);
      try {
        handle.free();
      } catch {
        // free is idempotent on our side; swallow
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
    );
  }
}

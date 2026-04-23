// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Shared adapter that wraps a wasm-bindgen `DecodedMessage` handle in
 * the public `DecodedMessage` interface.
 *
 * `@internal` — not re-exported from `index.ts`.
 *
 * Used by both the buffer-level `decode` / `decodeObject` (in
 * `decode.ts`) and the lazy file backend's `messageObject` (in
 * `file.ts`) so both call sites get the same lifecycle (`close()` +
 * `FinalizationRegistry` fallback) and the same dtype-aware
 * `data()` / `dataView()` accessors.
 */

import { typedArrayFor } from '../dtype.js';
import { InvalidArgumentError, rethrowTyped } from '../errors.js';
import type {
  DataObjectDescriptor,
  DecodedMessage,
  DecodedObject,
  TypedArray,
} from '../types.js';

/** Structural shape of the wasm-bindgen `DecodedMessage` class. */
export interface WbgDecodedMessage {
  free(): void;
  metadata(): unknown;
  object_count(): number;
  object_descriptor(index: number): unknown;
  object_data_u8(index: number): Uint8Array;
  object_byte_length(index: number): number;
}

const finalizationRegistry = new FinalizationRegistry<WbgDecodedMessage>((handle) => {
  try {
    handle.free();
  } catch {
    // best-effort cleanup
  }
});

/** Wrap a wbg handle in the public DecodedMessage shape. */
export function wrapWbgDecodedMessage(handle: WbgDecodedMessage): DecodedMessage {
  const metadata = rethrowTyped(() => handle.metadata()) as DecodedMessage['metadata'];
  const count = handle.object_count();
  const objects: DecodedObject[] = [];
  let closed = false;

  for (let i = 0; i < count; i++) {
    const descriptor = rethrowTyped(
      () => handle.object_descriptor(i) as DataObjectDescriptor,
    );
    const byteLength = handle.object_byte_length(i);

    const obj: DecodedObject = {
      descriptor,
      byteLength,
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

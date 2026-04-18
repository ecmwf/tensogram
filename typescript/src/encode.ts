// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Encode wrapper.
 *
 * Translates the ergonomic TS surface into the shape the wasm-bindgen
 * `encode()` export expects:
 *   `(metadata: any, objects: Array<{descriptor, data}>, hash?: boolean)`
 */

import { rethrowTyped, InvalidArgumentError } from './errors.js';
import { getWbg } from './init.js';
import {
  assertValidDescriptor,
  assertValidMetadata,
  toUint8View,
} from './internal/validation.js';
import type { EncodeInput, EncodeOptions, GlobalMetadata } from './types.js';

/**
 * Encode global metadata + a list of `(descriptor, data)` pairs into a
 * single Tensogram message.
 *
 * @param metadata - Global metadata (`version: 2` required)
 * @param objects  - Data objects; each `data` is any `ArrayBufferView`
 *                   (`TypedArray`, `DataView`, ...) in native byte order
 * @param options  - Optional hash selection and strict-finite flags
 * @returns Wire-format bytes as a `Uint8Array`
 * @throws {MetadataError} if metadata is malformed
 * @throws {EncodingError} if a pipeline stage rejects the input (e.g. NaN in simple_packing)
 *                         or if the strict-finite check catches a NaN/Inf
 * @throws {CompressionError} if a compression codec fails
 */
export function encode(
  metadata: GlobalMetadata,
  objects: readonly EncodeInput[],
  options?: EncodeOptions,
): Uint8Array {
  assertValidMetadata(metadata);
  assertValidObjects(objects);

  const wbg = getWbg();
  const objArray = objects.map((o) => ({
    descriptor: o.descriptor,
    data: toUint8View(o.data),
  }));

  // Default to hashing. Opt out explicitly with `{ hash: false }`.
  const hash = options?.hash !== false;
  // Strict-finite flags default to false, matching the Rust and Python
  // APIs. See EncodeOptions doc for full semantics.
  const rejectNan = options?.rejectNan === true;
  const rejectInf = options?.rejectInf === true;
  return rethrowTyped(() =>
    wbg.encode(metadata, objArray, hash, rejectNan, rejectInf),
  );
}

function assertValidObjects(objects: readonly EncodeInput[]): void {
  if (!Array.isArray(objects)) {
    throw new InvalidArgumentError(`objects must be an array, got ${typeof objects}`);
  }
  for (let i = 0; i < objects.length; i++) {
    const o = objects[i];
    if (o === null || typeof o !== 'object') {
      throw new InvalidArgumentError(`objects[${i}] must be a { descriptor, data } pair`);
    }
    assertValidDescriptor(o.descriptor, i);
    if (!ArrayBuffer.isView(o.data)) {
      throw new InvalidArgumentError(
        `objects[${i}].data must be an ArrayBufferView (TypedArray, DataView, ...)`,
      );
    }
  }
}

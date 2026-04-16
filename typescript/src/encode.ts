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
import { SUPPORTED_DTYPES } from './dtype.js';
import type {
  DataObjectDescriptor,
  EncodeInput,
  EncodeOptions,
  GlobalMetadata,
} from './types.js';

/**
 * Encode global metadata + a list of `(descriptor, data)` pairs into a
 * single Tensogram message.
 *
 * @param metadata - Global metadata (`version: 2` required)
 * @param objects  - Data objects; each `data` is any `ArrayBufferView`
 *                   (`TypedArray`, `DataView`, ...) in native byte order
 * @param options  - Optional hash selection
 * @returns Wire-format bytes as a `Uint8Array`
 * @throws {MetadataError} if metadata is malformed
 * @throws {EncodingError} if a pipeline stage rejects the input (e.g. NaN in simple_packing)
 * @throws {CompressionError} if a compression codec fails
 */
export function encode(
  metadata: GlobalMetadata,
  objects: readonly EncodeInput[],
  options?: EncodeOptions,
): Uint8Array {
  validateMetadata(metadata);
  validateObjects(objects);

  const wbg = getWbg();
  const objArray = objects.map((o) => ({
    descriptor: o.descriptor,
    data: toUint8ArrayView(o.data),
  }));

  const hash: boolean = options?.hash === false ? false : true;
  return rethrowTyped(() => wbg.encode(metadata, objArray, hash));
}

function validateMetadata(meta: GlobalMetadata): void {
  if (meta === null || typeof meta !== 'object') {
    throw new InvalidArgumentError(
      `metadata must be a plain object, got ${typeof meta}`,
      `metadata must be a plain object, got ${typeof meta}`,
    );
  }
  if (typeof meta.version !== 'number' || !Number.isInteger(meta.version) || meta.version < 0) {
    throw new InvalidArgumentError(
      `metadata.version must be a non-negative integer, got ${String(meta.version)}`,
      `metadata.version must be a non-negative integer, got ${String(meta.version)}`,
    );
  }
  if ('_reserved_' in meta && meta._reserved_ !== undefined) {
    throw new InvalidArgumentError(
      "metadata._reserved_ is managed by the library; client code must not write it",
      "metadata._reserved_ is managed by the library; client code must not write it",
    );
  }
  if (meta.base !== undefined) {
    if (!Array.isArray(meta.base)) {
      throw new InvalidArgumentError(
        `metadata.base must be an array, got ${typeof meta.base}`,
        `metadata.base must be an array, got ${typeof meta.base}`,
      );
    }
    for (let i = 0; i < meta.base.length; i++) {
      const entry = meta.base[i];
      if (entry === null || typeof entry !== 'object') {
        throw new InvalidArgumentError(
          `metadata.base[${i}] must be a plain object`,
          `metadata.base[${i}] must be a plain object`,
        );
      }
      if ('_reserved_' in entry) {
        throw new InvalidArgumentError(
          `metadata.base[${i}]._reserved_ is managed by the library; client code must not write it`,
          `metadata.base[${i}]._reserved_ is managed by the library; client code must not write it`,
        );
      }
    }
  }
}

function validateObjects(objects: readonly EncodeInput[]): void {
  if (!Array.isArray(objects)) {
    throw new InvalidArgumentError(
      `objects must be an array, got ${typeof objects}`,
      `objects must be an array, got ${typeof objects}`,
    );
  }
  for (let i = 0; i < objects.length; i++) {
    const o = objects[i];
    if (o === null || typeof o !== 'object') {
      throw new InvalidArgumentError(
        `objects[${i}] must be a { descriptor, data } pair`,
        `objects[${i}] must be a { descriptor, data } pair`,
      );
    }
    validateDescriptor(o.descriptor, i);
    if (!ArrayBuffer.isView(o.data)) {
      throw new InvalidArgumentError(
        `objects[${i}].data must be an ArrayBufferView (TypedArray, DataView, ...)`,
        `objects[${i}].data must be an ArrayBufferView (TypedArray, DataView, ...)`,
      );
    }
  }
}

function validateDescriptor(desc: DataObjectDescriptor, i: number): void {
  if (desc === null || typeof desc !== 'object') {
    throw new InvalidArgumentError(
      `objects[${i}].descriptor must be a plain object`,
      `objects[${i}].descriptor must be a plain object`,
    );
  }
  if (typeof desc.type !== 'string') {
    throw new InvalidArgumentError(
      `objects[${i}].descriptor.type must be a string`,
      `objects[${i}].descriptor.type must be a string`,
    );
  }
  if (!Array.isArray(desc.shape)) {
    throw new InvalidArgumentError(
      `objects[${i}].descriptor.shape must be an array`,
      `objects[${i}].descriptor.shape must be an array`,
    );
  }
  if (!SUPPORTED_DTYPES.has(desc.dtype)) {
    throw new InvalidArgumentError(
      `objects[${i}].descriptor.dtype=${String(desc.dtype)} is not a recognised dtype`,
      `objects[${i}].descriptor.dtype=${String(desc.dtype)} is not a recognised dtype`,
    );
  }
  if (desc.byte_order !== 'big' && desc.byte_order !== 'little') {
    throw new InvalidArgumentError(
      `objects[${i}].descriptor.byte_order must be "big" or "little"`,
      `objects[${i}].descriptor.byte_order must be "big" or "little"`,
    );
  }
}

/**
 * Normalise any `ArrayBufferView` into a `Uint8Array` over exactly the
 * bytes it covers. The WASM `encode` function accepts any TypedArray
 * and picks up the byte range from `byteOffset` / `byteLength`, but a
 * `DataView` is not a TypedArray on the JS side — so we wrap it.
 */
function toUint8ArrayView(view: ArrayBufferView): Uint8Array {
  if (view instanceof Uint8Array) return view;
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

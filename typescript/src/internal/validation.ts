// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Shared runtime-validation helpers for the write paths.
 *
 * Every public TS entrypoint that builds or appends a wire-format
 * message (`encode`, `encodePreEncoded`, `StreamingEncoder`,
 * `TensogramFile#append`) funnels through these helpers so error
 * messages are identical across the surface and the policy for
 * rejecting malformed `metadata` / `descriptor` only lives in one
 * place.
 *
 * `@internal` — not re-exported from `index.ts`.
 */

import { SUPPORTED_DTYPES } from '../dtype.js';
import { InvalidArgumentError } from '../errors.js';
import type { DataObjectDescriptor, GlobalMetadata } from '../types.js';

/**
 * Assert that `meta` is shaped like a `GlobalMetadata` the encoder can
 * process: a plain object, no client-written `_reserved_`, and every
 * `base[i]` entry is a plain object with no `_reserved_` key.
 *
 * The CBOR metadata frame is free-form (see
 * `plans/WIRE_FORMAT.md` §6.1) — the only library-enforced invariants
 * are the `_reserved_` protection and the shape of `base`.  Any
 * top-level `version` the caller supplies is carried through as a
 * free-form `_extra_` annotation; the wire-format version itself
 * lives in the preamble (see `plans/WIRE_FORMAT.md` §3) and is
 * accessible at `WIRE_VERSION`.
 */
export function assertValidMetadata(meta: GlobalMetadata): void {
  if (meta === null || typeof meta !== 'object') {
    throw new InvalidArgumentError(
      `metadata must be a plain object, got ${typeof meta}`,
    );
  }
  if ('_reserved_' in meta && meta._reserved_ !== undefined) {
    throw new InvalidArgumentError(
      'metadata._reserved_ is managed by the library; client code must not write it',
    );
  }
  if (meta.base !== undefined) {
    if (!Array.isArray(meta.base)) {
      throw new InvalidArgumentError(
        `metadata.base must be an array, got ${typeof meta.base}`,
      );
    }
    for (let i = 0; i < meta.base.length; i++) {
      const entry = meta.base[i];
      if (entry === null || typeof entry !== 'object') {
        throw new InvalidArgumentError(`metadata.base[${i}] must be a plain object`);
      }
      if ('_reserved_' in entry) {
        throw new InvalidArgumentError(
          `metadata.base[${i}]._reserved_ is managed by the library; client code must not write it`,
        );
      }
    }
  }
}

/**
 * Assert that `desc` looks like a valid `DataObjectDescriptor`.  When
 * `index` is supplied, error messages prefix the field paths with
 * `objects[i].descriptor.` for extra context — matching the way the
 * call-site loops over object arrays.
 */
export function assertValidDescriptor(
  desc: DataObjectDescriptor,
  index?: number,
): void {
  const prefix = index === undefined ? 'descriptor' : `objects[${index}].descriptor`;
  if (desc === null || typeof desc !== 'object') {
    throw new InvalidArgumentError(`${prefix} must be a plain object`);
  }
  if (typeof desc.type !== 'string') {
    throw new InvalidArgumentError(`${prefix}.type must be a string`);
  }
  if (!Array.isArray(desc.shape)) {
    throw new InvalidArgumentError(`${prefix}.shape must be an array`);
  }
  if (!SUPPORTED_DTYPES.has(desc.dtype)) {
    throw new InvalidArgumentError(
      `${prefix}.dtype=${String(desc.dtype)} is not a recognised dtype`,
    );
  }
  if (desc.byte_order !== 'big' && desc.byte_order !== 'little') {
    throw new InvalidArgumentError(`${prefix}.byte_order must be "big" or "little"`);
  }
}

/**
 * Normalise any `ArrayBufferView` into a `Uint8Array` covering exactly
 * the bytes the view refers to.  The WASM encode entry points accept
 * any TypedArray and read `byteOffset` + `byteLength`, but a
 * `DataView` is not a TypedArray on the JS side — so we wrap it into
 * a `Uint8Array` pointing at the same `ArrayBuffer` range.
 */
export function toUint8View(view: ArrayBufferView): Uint8Array {
  if (view instanceof Uint8Array) return view;
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

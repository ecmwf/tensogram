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
 * Translates the ergonomic TS surface into the positional argument list
 * the wasm-bindgen `encode()` export expects: metadata, the
 * `{descriptor, data}` array, then the option scalars (hash, the
 * NaN / Inf mask knobs, aggregate-hash placement).
 *
 * Option names that the WASM side does not parse itself —
 * `aggregateHash` and the platform-no-op `compressionBackend` — are
 * validated here so a typo surfaces as an {@link InvalidArgumentError}
 * naming the offending option rather than as a generic framing error
 * from across the boundary.
 */

import { rethrowTyped, InvalidArgumentError } from './errors.js';
import { getWbg } from './init.js';
import {
  assertValidDescriptor,
  assertValidMetadata,
  toUint8View,
} from './internal/validation.js';
import type {
  AggregateHashPolicy,
  CompressionBackend,
  EncodeInput,
  EncodeOptions,
  GlobalMetadata,
} from './types.js';

/**
 * Accepted {@link AggregateHashPolicy} names, for runtime validation of
 * plain-JS callers who bypass the compile-time union.
 *
 * Declared as a `Record` keyed by the union so the compiler rejects
 * this table the moment a policy is added to — or removed from — the
 * type: the runtime mirror cannot drift from `AggregateHashPolicy`.
 */
const AGGREGATE_HASH_POLICIES: Record<AggregateHashPolicy, true> = {
  auto: true,
  none: true,
  header: true,
  footer: true,
  both: true,
};

/** Accepted {@link CompressionBackend} names.  See above for the shape. */
const COMPRESSION_BACKENDS: Record<CompressionBackend, true> = {
  auto: true,
  ffi: true,
  pure: true,
};

/**
 * Encode global metadata + a list of `(descriptor, data)` pairs into a
 * single Tensogram message.
 *
 * @param metadata - Global metadata (free-form CBOR; only `base`, `_reserved_`, and `_extra_` are library-interpreted)
 * @param objects  - Data objects; each `data` is any `ArrayBufferView`
 *                   (`TypedArray`, `DataView`, ...) in native byte order
 * @param options  - Optional hash selection, strict-finite flags, and
 *                   aggregate-hash placement.  See {@link EncodeOptions}.
 * @returns Wire-format bytes as a `Uint8Array`
 * @throws {MetadataError} if metadata is malformed
 * @throws {EncodingError} if a pipeline stage rejects the input (e.g. NaN in simple_packing)
 *                         or if the strict-finite check catches a NaN/Inf
 * @throws {CompressionError} if a compression codec fails
 * @throws {InvalidArgumentError} if `aggregateHash` or `compressionBackend`
 *                         names an unknown value
 */
export function encode(
  metadata: GlobalMetadata,
  objects: readonly EncodeInput[],
  options?: EncodeOptions,
): Uint8Array {
  assertValidMetadata(metadata);
  assertValidObjects(objects);
  assertKnownName(
    options?.aggregateHash,
    AGGREGATE_HASH_POLICIES,
    'aggregateHash',
  );
  // Validated but deliberately NOT forwarded: the WASM bundle carries
  // pure-Rust codecs only, so no value can change the bytes produced.
  // See `EncodeOptions.compressionBackend` for the platform-no-op
  // rationale — we still reject typos rather than swallow them.
  assertKnownName(
    options?.compressionBackend,
    COMPRESSION_BACKENDS,
    'compressionBackend',
  );

  const wbg = getWbg();
  const objArray = objects.map((o) => ({
    descriptor: o.descriptor,
    data: toUint8View(o.data),
  }));

  // Default to hashing. Opt out explicitly with `{ hash: false }`.
  const hash = options?.hash !== false;
  return rethrowTyped(() =>
    wbg.encode(
      metadata,
      objArray,
      hash,
      options?.allowNan ?? false,
      options?.allowInf ?? false,
      options?.nanMaskMethod,
      options?.posInfMaskMethod,
      options?.negInfMaskMethod,
      options?.smallMaskThresholdBytes,
      options?.aggregateHash,
    ),
  );
}

/**
 * Reject an option value that is not in its accepted-name table.  An
 * `undefined` value means "not supplied" and always passes — the
 * encoder applies its own default.
 */
function assertKnownName(
  value: string | undefined,
  accepted: Record<string, true>,
  option: string,
): void {
  if (value === undefined) return;
  if (!Object.hasOwn(accepted, value)) {
    throw new InvalidArgumentError(
      `unknown ${option} '${value}', expected one of: ${Object.keys(accepted).join(', ')}`,
    );
  }
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

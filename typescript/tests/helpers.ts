// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { beforeAll } from 'vitest';
import { init } from '../src/index.js';
import type { DataObjectDescriptor, Dtype, GlobalMetadata } from '../src/index.js';

/** Ensure WASM is initialised exactly once per worker. */
export function initOnce(): void {
  beforeAll(async () => {
    await init();
  });
}

/**
 * Build a minimal descriptor in little-endian native layout, with row-major
 * strides derived from `shape`.
 */
export function makeDescriptor(shape: readonly number[], dtype: Dtype): DataObjectDescriptor {
  const strides: number[] = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype,
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

/** Minimal v2 metadata. */
export function defaultMeta(): GlobalMetadata {
  return { version: 2 };
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `computeHash` — standalone hash computation tests.  The core
 * correctness contract is "same bytes → same digest, length is the
 * advertised xxh3 16-char hex".  Cross-checked against the hash
 * stamped by `encode()` on uncompressed payloads so we know the TS
 * wrapper uses the same algorithm.
 */

import { describe, expect, it } from 'vitest';
import {
  computeHash,
  decode,
  encode,
  InvalidArgumentError,
  MetadataError,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.1 — computeHash', () => {
  initOnce();

  it('returns a 16-char lower-case hex string for xxh3', () => {
    const hex = computeHash(new Uint8Array([1, 2, 3, 4]));
    expect(hex).toMatch(/^[0-9a-f]{16}$/);
  });

  it('is deterministic on identical input', () => {
    const data = new Uint8Array([42, 43, 44, 45, 46, 47, 48, 49]);
    expect(computeHash(data)).toBe(computeHash(data));
  });

  it('produces different digests for different input', () => {
    const a = computeHash(new Uint8Array([1, 2, 3]));
    const b = computeHash(new Uint8Array([1, 2, 4]));
    expect(a).not.toBe(b);
  });

  it('accepts explicit algo="xxh3"', () => {
    const hex = computeHash(new Uint8Array([5]), 'xxh3');
    expect(hex).toMatch(/^[0-9a-f]{16}$/);
  });

  it('handles empty buffer', () => {
    const hex = computeHash(new Uint8Array(0));
    expect(hex).toMatch(/^[0-9a-f]{16}$/);
  });

  it('rejects non-Uint8Array input', () => {
    // @ts-expect-error intentional bad input
    expect(() => computeHash([1, 2, 3])).toThrow(InvalidArgumentError);
    // @ts-expect-error intentional bad input
    expect(() => computeHash('hello')).toThrow(InvalidArgumentError);
  });

  it('rejects an unknown hash algorithm as a typed error', () => {
    expect(() =>
      // @ts-expect-error intentional: HashAlgorithm is strictly narrowed
      computeHash(new Uint8Array([1]), 'sha256'),
    ).toThrow(MetadataError);
  });

  it('matches the hash stamped on an encoded payload (no pipeline)', async () => {
    // For encoding=filter=compression=none, the encoded payload is the
    // raw native-endian bytes.  TS-side hash of those bytes must equal
    // the descriptor's recorded hash.
    const values = new Float32Array([1.25, 2.5, 3.75, 4.0]);
    const raw = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data: values },
    ]);
    const decoded = decode(msg);
    try {
      const descHash = decoded.objects[0].descriptor.hash;
      expect(descHash?.type).toBe('xxh3');
      const standalone = computeHash(raw);
      expect(standalone).toBe(descHash?.value);
    } finally {
      decoded.close();
    }
  });
});

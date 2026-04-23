// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Tests for `internal/layout.ts` — normalisation boundary between
 * WASM `bigint`/`number` values and TypeScript `number`-based file
 * positions.
 *
 * Addresses Oracle plan-review concern #5 (numeric contract pinning).
 */

import { describe, expect, it } from 'vitest';
import { InvalidArgumentError } from '../src/index.js';
import {
  normaliseFrameIndex,
  normalisePostambleInfo,
  normalisePreambleInfo,
  safeNumberFromBigint,
} from '../src/internal/layout.js';

describe('safeNumberFromBigint', () => {
  it('accepts finite non-negative integers as number', () => {
    expect(safeNumberFromBigint(0, 'x')).toBe(0);
    expect(safeNumberFromBigint(1, 'x')).toBe(1);
    expect(safeNumberFromBigint(Number.MAX_SAFE_INTEGER, 'x')).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('accepts non-negative bigints that fit in a safe integer', () => {
    expect(safeNumberFromBigint(0n, 'x')).toBe(0);
    expect(safeNumberFromBigint(42n, 'x')).toBe(42);
    expect(safeNumberFromBigint(BigInt(Number.MAX_SAFE_INTEGER), 'x')).toBe(
      Number.MAX_SAFE_INTEGER,
    );
  });

  it('rejects values above MAX_SAFE_INTEGER with InvalidArgumentError', () => {
    // 2^53 is the first unsafe integer.
    const over = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
    expect(() => safeNumberFromBigint(over, 'offset')).toThrow(InvalidArgumentError);
    expect(() => safeNumberFromBigint(over, 'offset')).toThrow(/MAX_SAFE_INTEGER/);
  });

  it('rejects 2^64 - 1 (u64 max) — the canonical wire-format overflow', () => {
    const u64Max = (1n << 64n) - 1n;
    expect(() => safeNumberFromBigint(u64Max, 'offset')).toThrow(InvalidArgumentError);
  });

  it('rejects negative values', () => {
    expect(() => safeNumberFromBigint(-1, 'x')).toThrow(InvalidArgumentError);
    expect(() => safeNumberFromBigint(-1n, 'x')).toThrow(InvalidArgumentError);
  });

  it('rejects NaN / Infinity / fractional numbers', () => {
    expect(() => safeNumberFromBigint(Number.NaN, 'x')).toThrow(InvalidArgumentError);
    expect(() => safeNumberFromBigint(Infinity, 'x')).toThrow(InvalidArgumentError);
    expect(() => safeNumberFromBigint(1.5, 'x')).toThrow(InvalidArgumentError);
  });

  it('rejects non-number / non-bigint inputs', () => {
    // @ts-expect-error intentional: string is not accepted at runtime.
    expect(() => safeNumberFromBigint('42', 'x')).toThrow(InvalidArgumentError);
    // @ts-expect-error intentional: undefined is not accepted at runtime.
    expect(() => safeNumberFromBigint(undefined, 'x')).toThrow(InvalidArgumentError);
  });

  it('quotes the context name in the error message', () => {
    try {
      safeNumberFromBigint(-5, 'my.field');
      expect.fail('should have thrown');
    } catch (err) {
      expect(String(err)).toContain('my.field');
    }
  });
});

describe('normalisePreambleInfo', () => {
  it('converts WASM object to TS PreambleInfo shape', () => {
    const raw = {
      version: 3,
      flags: 0b1011,
      total_length: 12345n,
      has_header_metadata: true,
      has_header_index: true,
      has_footer_metadata: false,
      has_footer_index: false,
      has_preceder_metadata: false,
      hashes_present: true,
    };
    const info = normalisePreambleInfo(raw);
    expect(info.totalLength).toBe(12345);
    expect(info.hasHeaderMetadata).toBe(true);
    expect(info.hasFooterIndex).toBe(false);
  });

  it('rejects bigint total_length above MAX_SAFE_INTEGER', () => {
    const raw = {
      version: 3,
      flags: 0,
      total_length: BigInt(Number.MAX_SAFE_INTEGER) + 1n,
      has_header_metadata: false,
      has_header_index: false,
      has_footer_metadata: false,
      has_footer_index: false,
      has_preceder_metadata: false,
      hashes_present: false,
    };
    expect(() => normalisePreambleInfo(raw)).toThrow(InvalidArgumentError);
  });

  it('rejects null input', () => {
    expect(() => normalisePreambleInfo(null)).toThrow(InvalidArgumentError);
  });
});

describe('normalisePostambleInfo', () => {
  it('converts WASM object to TS PostambleInfo shape', () => {
    const raw = {
      first_footer_offset: 100n,
      total_length: 500n,
      end_magic_ok: true,
    };
    const info = normalisePostambleInfo(raw);
    expect(info.firstFooterOffset).toBe(100);
    expect(info.totalLength).toBe(500);
    expect(info.endMagicOk).toBe(true);
  });
});

describe('normaliseFrameIndex', () => {
  it('converts BigUint64Array offsets/lengths to number arrays', () => {
    const offsets = new BigUint64Array([100n, 200n, 300n]);
    const lengths = new BigUint64Array([50n, 60n, 70n]);
    const idx = normaliseFrameIndex({ offsets, lengths });
    expect(idx.offsets).toEqual([100, 200, 300]);
    expect(idx.lengths).toEqual([50, 60, 70]);
  });

  it('accepts plain arrays of number or bigint', () => {
    const idx = normaliseFrameIndex({
      offsets: [100, 200n],
      lengths: [50n, 60],
    });
    expect(idx.offsets).toEqual([100, 200]);
    expect(idx.lengths).toEqual([50, 60]);
  });

  it('rejects mismatched array lengths', () => {
    expect(() =>
      normaliseFrameIndex({
        offsets: new BigUint64Array([1n]),
        lengths: new BigUint64Array([1n, 2n]),
      }),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects any offset above MAX_SAFE_INTEGER', () => {
    const over = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
    expect(() =>
      normaliseFrameIndex({
        offsets: new BigUint64Array([1n, over]),
        lengths: new BigUint64Array([1n, 2n]),
      }),
    ).toThrow(InvalidArgumentError);
  });
});

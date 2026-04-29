// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

// Decode-time hash verification matrix on the TypeScript surface.
// Mirrors the per-binding test files; see
// `PLAN_DECODE_HASH_VERIFICATION.md` §5.2.

import { describe, expect, it } from 'vitest';
import {
  decode,
  decodeObject,
  encode,
  HashMismatchError,
  IntegrityError,
  MissingHashError,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

initOnce();

function encodeSimpleHashed(values: number[]): Uint8Array {
  return encode(
    defaultMeta(),
    [{ descriptor: makeDescriptor([values.length], 'float32'),
       data: new Float32Array(values) }],
    { hash: 'xxh3' },
  );
}

function encodeSimpleUnhashed(values: number[]): Uint8Array {
  return encode(
    defaultMeta(),
    [{ descriptor: makeDescriptor([values.length], 'float32'),
       data: new Float32Array(values) }],
    { hash: false },
  );
}

/** Walk the bytes to find the first NTensorFrame and return
 *  `[frame_start, total_length]`.  Mirrors the helper in the
 *  Python and Rust matrix tests. */
function locateFirstObjectFrame(buf: Uint8Array): [number, number] {
  // Skip the 24-byte preamble.
  let pos = 24;
  while (pos + 16 <= buf.length) {
    if (buf[pos] !== 0x46 || buf[pos + 1] !== 0x52) {
      pos += 1;
      continue;
    }
    const frameType = (buf[pos + 2] << 8) | buf[pos + 3];
    let totalLength = 0;
    for (let i = 0; i < 8; i++) {
      totalLength = totalLength * 256 + buf[pos + 8 + i];
    }
    if (frameType === 9) {
      return [pos, totalLength];
    }
    pos = (pos + totalLength + 7) & ~7;
  }
  throw new Error('no NTensorFrame found');
}

// ── Cells A & B — verify on/off, hashed message ───────────────────────

describe('verifyHash — Cells A & B (hashed message)', () => {
  it('decode without verifyHash succeeds', () => {
    const bytes = encodeSimpleHashed([1, 2, 3, 4]);
    const msg = decode(bytes);
    expect(msg.objects.length).toBe(1);
    msg.close();
  });

  it('decode with verifyHash=true succeeds', () => {
    const bytes = encodeSimpleHashed([1, 2, 3, 4]);
    const msg = decode(bytes, { verifyHash: true });
    expect(msg.objects.length).toBe(1);
    msg.close();
  });

  it('decodeObject with verifyHash=true succeeds', () => {
    const bytes = encodeSimpleHashed([10, 20]);
    const msg = decodeObject(bytes, 0, { verifyHash: true });
    expect(msg.objects.length).toBe(1);
    msg.close();
  });
});

// ── Cell C — unhashed message + verifyHash=true → MissingHashError ────

describe('verifyHash — Cell C (unhashed message)', () => {
  it('decode throws MissingHashError', () => {
    const bytes = encodeSimpleUnhashed([5]);
    let caught: unknown;
    try {
      decode(bytes, { verifyHash: true });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(MissingHashError);
    expect(caught).toBeInstanceOf(IntegrityError);
    expect((caught as MissingHashError).objectIndex).toBe(0);
  });

  it('decodeObject throws MissingHashError', () => {
    const bytes = encodeSimpleUnhashed([5]);
    let caught: unknown;
    try {
      decodeObject(bytes, 0, { verifyHash: true });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(MissingHashError);
    expect((caught as MissingHashError).objectIndex).toBe(0);
  });

  it('verifyHash defaults to false (unhashed message decodes cleanly)', () => {
    const bytes = encodeSimpleUnhashed([5, 6]);
    const msg = decode(bytes);
    expect(msg.objects.length).toBe(1);
    msg.close();
  });
});

// ── Cell D — tampered hash slot → HashMismatchError ───────────────────

describe('verifyHash — Cell D (tampered hash slot)', () => {
  it('decode reports HashMismatchError with objectIndex', () => {
    const bytes = encodeSimpleHashed([1, 2, 3]);
    const [frameStart, totalLength] = locateFirstObjectFrame(bytes);
    // Inline hash slot lives at frame_end - 12.
    const slotByte = frameStart + totalLength - 12;
    const tampered = new Uint8Array(bytes);
    tampered[slotByte] ^= 0xff;

    let caught: unknown;
    try {
      decode(tampered, { verifyHash: true });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(HashMismatchError);
    expect(caught).toBeInstanceOf(IntegrityError);
    const err = caught as HashMismatchError;
    expect(err.objectIndex).toBe(0);
    expect(err.expected).toBeTruthy();
    expect(err.actual).toBeTruthy();
    expect(err.expected).not.toBe(err.actual);
  });

  it('decodeObject reports HashMismatchError', () => {
    const bytes = encodeSimpleHashed([1, 2, 3]);
    const [frameStart, totalLength] = locateFirstObjectFrame(bytes);
    const slotByte = frameStart + totalLength - 12;
    const tampered = new Uint8Array(bytes);
    tampered[slotByte] ^= 0xff;

    expect(() => decodeObject(tampered, 0, { verifyHash: true }))
      .toThrow(HashMismatchError);
  });
});

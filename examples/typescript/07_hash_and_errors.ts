// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 07 — Typed error hierarchy (TypeScript)
 *
 * Demonstrates the typed error hierarchy.  `decode` on a garbage
 * buffer throws `FramingError`; out-of-range object indices throw
 * `ObjectError`.  All concrete errors extend `TensogramError` so a
 * single `catch (e) { if (e instanceof TensogramError) ... }`
 * handles every library-raised error.
 *
 * **v3 note.** Frame-level integrity verification moved from the
 * decoder to the validate API (plans/WIRE_FORMAT.md §11).
 * `decode(buf, { verifyHash: true })` is a no-op; corruption
 * surfaces through `tensogram validate --checksum` (or the TS
 * `validate` wrapper when the slot-level accessor lands).
 */

import {
  decode,
  encode,
  FramingError,
  HashMismatchError,
  init,
  ObjectError,
  TensogramError,
  decodeObject,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

function descFor(shape: number[]): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype: 'float32',
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

async function main(): Promise<void> {
  await init();

  const data = new Float32Array(1000);
  for (let i = 0; i < data.length; i++) data[i] = i;

  // Encode with the default xxh3 hash.
  const msg = encode({ version: 3 }, [{ descriptor: descFor([1000]), data }]);

  // 1. Clean decode with verifyHash: true — succeeds.
  const clean = decode(msg, { verifyHash: true });
  console.log(`clean decode: OK (${clean.objects.length} object)`);
  clean.close();

  // 2. v3: decode is no longer the integrity-verification surface.
  //     A byte flip in the payload may decode silently (the inline
  //     hash slot mismatch is not checked at decode time).  Use the
  //     validate API to detect the mismatch.  Structural tamper
  //     still surfaces as FramingError.
  const tampered = new Uint8Array(msg);
  tampered[500] ^= 0xff;
  try {
    const d = decode(tampered, { verifyHash: true });
    console.log('tamper not detected at decode (expected in v3 — use validate for integrity)');
    d.close();
  } catch (err) {
    if (err instanceof FramingError) {
      console.log(`tamper landed on structural byte: ${err.message}`);
    } else if (err instanceof HashMismatchError) {
      // Won't hit in v3, but catch defensively.
      console.log(`HashMismatchError: expected=${err.expected} actual=${err.actual}`);
    } else {
      throw err;
    }
  }

  // 3. Garbage input → FramingError.
  try {
    decode(new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]));
    console.error('expected a FramingError');
    process.exit(1);
  } catch (err) {
    if (!(err instanceof FramingError)) throw err;
    console.log(`framing error:    ${err.message}`);
  }

  // 4. Index out of range → ObjectError.
  try {
    decodeObject(msg, 99);
    console.error('expected an ObjectError');
    process.exit(1);
  } catch (err) {
    if (!(err instanceof ObjectError)) throw err;
    console.log(`object error:     ${err.message}`);
  }

  // 5. All concrete errors are TensogramError, enabling broad catches.
  const errors = [
    new HashMismatchError('hash mismatch', 'hash mismatch', 'aa', 'bb'),
    new FramingError('framing error: foo', 'foo'),
    new ObjectError('object error: bar', 'bar'),
  ];
  for (const e of errors) {
    console.log(`  ${e.name}: instanceof TensogramError = ${e instanceof TensogramError}`);
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

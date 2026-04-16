// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 04 — Hash verification and typed errors (TypeScript)
 *
 * Demonstrates the typed error hierarchy. `decode(buf, { verifyHash: true })`
 * throws `HashMismatchError` when the payload has been tampered with.
 * `decode` on a garbage buffer throws `FramingError`, and so on.
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
} from '@ecmwf/tensogram';

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
  const msg = encode({ version: 2 }, [{ descriptor: descFor([1000]), data }]);

  // 1. Clean decode with hash verification — succeeds.
  const clean = decode(msg, { verifyHash: true });
  console.log(`clean verify: OK (${clean.objects.length} object)`);
  clean.close();

  // 2. Tamper with a byte in the payload and verify again.
  const tampered = new Uint8Array(msg);
  tampered[500] ^= 0xff;
  try {
    decode(tampered, { verifyHash: true });
    console.error('expected a HashMismatchError');
    process.exit(1);
  } catch (err) {
    if (!(err instanceof HashMismatchError)) throw err;
    console.log(`tamper detected:  expected=${err.expected}  actual=${err.actual}`);
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

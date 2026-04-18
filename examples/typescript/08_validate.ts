// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 08 — validate a single buffer and a multi-message file.
 *
 * `validate(buf)` never throws on bad input — it returns a structured
 * report of every structural / metadata / hash / fidelity issue.  We
 * exercise both the happy path and a truncated buffer that reports
 * errors.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 08_validate.ts
 */

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  encode,
  init,
  validate,
  validateFile,
  type DataObjectDescriptor,
} from '@ecmwf/tensogram';

function describe(shape: number[], dtype: DataObjectDescriptor['dtype']): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
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

async function main(): Promise<void> {
  await init();

  // 1. Valid buffer — expect zero error-severity issues and hash_verified=true.
  const good = encode({ version: 2 }, [
    { descriptor: describe([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) },
  ]);
  const goodReport = validate(good);
  console.log('valid buffer:');
  console.log(`  object_count=${goodReport.object_count}  hash_verified=${goodReport.hash_verified}`);
  console.log(`  issues=${goodReport.issues.length}`);

  // 2. Truncated buffer — validate reports issues, does not throw.
  const truncated = good.subarray(0, Math.floor(good.byteLength / 2));
  const badReport = validate(truncated);
  console.log('\ntruncated buffer:');
  for (const issue of badReport.issues.slice(0, 3)) {
    console.log(
      `  [${issue.severity}] ${issue.code} (${issue.level}): ${issue.description}`,
    );
  }

  // 3. File-level validation with validateFile (Node only).
  const tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-example-'));
  try {
    const path = join(tmp, 'multi.tgm');
    // Concatenate two valid messages into a single file.
    const m1 = encode({ version: 2 }, [
      { descriptor: describe([2], 'float32'), data: new Float32Array([10, 20]) },
    ]);
    const m2 = encode({ version: 2 }, [
      { descriptor: describe([3], 'float64'), data: new Float64Array([0.1, 0.2, 0.3]) },
    ]);
    const combined = new Uint8Array(m1.byteLength + m2.byteLength);
    combined.set(m1, 0);
    combined.set(m2, m1.byteLength);
    writeFileSync(path, combined);

    const fileReport = await validateFile(path);
    console.log('\nvalidateFile:');
    console.log(`  messages=${fileReport.messages.length}  file_issues=${fileReport.file_issues.length}`);
    fileReport.messages.forEach((m, i) => {
      console.log(`  message[${i}]  objects=${m.object_count}  hash_verified=${m.hash_verified}`);
    });
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 19 — TensogramFile.create + append-with-masks (TypeScript)
 *
 * Two capabilities that mirror the Rust core and complete the write
 * surface:
 *
 * 1. {@link TensogramFile.create} — a Node-only empty-file factory
 *    (like `tensogram::file::create`).  It creates (or truncates) a
 *    `.tgm` file and creates any missing parent directories, so
 *    subsequent {@link TensogramFile.append} calls extend it on disk.
 *
 * 2. append-with-masks — {@link TensogramFile.append} forwards the FULL
 *    {@link AppendOptions} surface (`allowNan` / `allowInf` /
 *    `*MaskMethod` / `smallMaskThresholdBytes`) to the encoder, so a
 *    payload carrying NaN / ±Inf can be appended (they are recorded in
 *    a companion bitmask and restored on decode).  Without `allowNan`
 *    the same append is rejected.
 */

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  init,
  TensogramFile,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

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

  // A nested path whose parent directories do not exist yet — create()
  // makes them, just like the Rust core's `create_dir_all`.
  const dir = mkdtempSync(join(tmpdir(), 'tgm-ts-create-'));
  const path = join(dir, 'nested', 'series.tgm');

  try {
    console.log('─── 1. TensogramFile.create (empty-file factory) ────────');
    const file = await TensogramFile.create(path);
    try {
      console.log(`  path:         ${path}`);
      console.log(`  source:       ${file.source}`);
      console.log(`  messageCount: ${file.messageCount}  (empty on create)`);

      // A plain message with finite values.
      await file.append({ base: [{ note: 'finite field' }] }, [
        { descriptor: describe([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) },
      ]);
      console.log('\n─── 2. append a finite message ──────────────────────────');
      console.log(`  messageCount: ${file.messageCount}`);

      // A message carrying NaN / ±Inf.  This REQUIRES the mask options
      // to be forwarded to the encoder — otherwise the append is
      // rejected.  We also pick an explicit mask compression method.
      console.log('\n─── 3. append a NaN/Inf message (allowNan + allowInf) ───');
      const withHoles = new Float64Array([1.5, NaN, 3.5, Infinity, -Infinity, 6.5]);
      await file.append(
        { base: [{ note: 'field with gaps' }] },
        [{ descriptor: describe([withHoles.length], 'float64'), data: withHoles }],
        {
          allowNan: true,
          allowInf: true,
          nanMaskMethod: 'roaring',
          posInfMaskMethod: 'rle',
          negInfMaskMethod: 'rle',
          smallMaskThresholdBytes: 0,
        },
      );
      console.log(`  messageCount: ${file.messageCount}`);

      // Show that WITHOUT the mask options the same payload is rejected.
      let rejected = false;
      try {
        await file.append({}, [
          { descriptor: describe([2], 'float64'), data: new Float64Array([NaN, 1]) },
        ]);
      } catch (err) {
        rejected = true;
        console.log(`  plain append of NaN rejected: ${(err as Error).constructor.name}`);
      }
      if (!rejected) throw new Error('expected the plain NaN append to be rejected');
    } finally {
      file.close();
    }

    console.log('\n─── 4. reopen and read the masks back ───────────────────');
    const reopened = await TensogramFile.open(path);
    try {
      console.log(`  messageCount: ${reopened.messageCount}`);
      const msg = await reopened.message(1);
      const restored = msg.objects[0].data() as Float64Array;
      const shown = Array.from(restored).map((v) => {
        if (Number.isNaN(v)) return 'NaN';
        if (v === Infinity) return '+Inf';
        if (v === -Infinity) return '-Inf';
        return String(v);
      });
      console.log(`  message[1]:   [${shown.join(', ')}]  (NaN / ±Inf restored)`);
      msg.close();
    } finally {
      reopened.close();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

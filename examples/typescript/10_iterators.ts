// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 10 — Iterator APIs (TypeScript)
 *
 * Three ways to walk a multi-message `.tgm` without reading it all into
 * memory at once:
 *
 *   1. `for await (const msg of file)` — file-level async iteration.
 *   2. `scan(buf)` — buffer-level message-position scan, tolerant of
 *      corrupt regions between messages.
 *   3. `file.message(i)` / `file.messageMetadata(i)` — random-access
 *      by index (O(1) via the binary header).
 */

import {
  decode,
  encode,
  init,
  scan,
  TensogramFile,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '@ecmwf.int/tensogram';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function describe(shape: number[]): DataObjectDescriptor {
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

function makeMessage(param: string, step: number): Uint8Array {
  const metadata: GlobalMetadata = {
    version: 3,
    base: [{ mars: { param, step } }],
  };
  const data = new Float32Array(10).fill(step + 0.5);
  return encode(metadata, [{ descriptor: describe([10]), data }]);
}

async function main(): Promise<void> {
  await init();

  const tmp = mkdtempSync(join(tmpdir(), 'tgm-iter-'));
  const path = join(tmp, 'forecast.tgm');
  try {
    // ── 1. Write a file of four messages ────────────────────────────────
    const messages = [
      makeMessage('2t', 0),
      makeMessage('10u', 0),
      makeMessage('2t', 6),
      makeMessage('10u', 6),
    ];
    const concatenated = new Uint8Array(messages.reduce((n, m) => n + m.byteLength, 0));
    let offset = 0;
    for (const m of messages) {
      concatenated.set(m, offset);
      offset += m.byteLength;
    }
    writeFileSync(path, concatenated);
    console.log(`Wrote ${concatenated.byteLength} bytes, ${messages.length} messages`);

    // ── 2. Async iteration over the file ────────────────────────────────
    //
    // `TensogramFile` implements `AsyncIterable<DecodedMessage>`, so
    // `for await` is the idiomatic walk.  Each yielded handle owns
    // WASM-side memory and must be `.close()`'d.
    console.log('\n1. for await (const msg of file):');
    const file = await TensogramFile.open(path);
    try {
      let i = 0;
      for await (const msg of file) {
        try {
          const param = (msg.metadata.base?.[0] as { mars?: { param: string } } | undefined)
            ?.mars?.param ?? '?';
          console.log(`   [${i++}] param=${param}  objects=${msg.objects.length}`);
        } finally {
          msg.close();
        }
      }
    } finally {
      file.close();
    }

    // ── 3. Buffer-level scan ─────────────────────────────────────────────
    //
    // `scan` finds message boundaries in an arbitrary byte buffer — you
    // can scan bytes that didn't come from a file (HTTP body, socket,
    // pipe, clipboard, ...).  A corrupt stretch between messages is
    // skipped silently; `scan` only reports the positions it could
    // verify end-to-end.
    console.log('\n2. scan(buf) over an in-memory buffer:');
    const positions = scan(concatenated);
    for (let j = 0; j < positions.length; j++) {
      const { offset: o, length: len } = positions[j];
      const slice = concatenated.subarray(o, o + len);
      const msg = decode(slice);
      try {
        const param = (msg.metadata.base?.[0] as { mars?: { param: string } } | undefined)
          ?.mars?.param ?? '?';
        console.log(`   [${j}] offset=${o}  length=${len}  param=${param}`);
      } finally {
        msg.close();
      }
    }

    // ── 4. Random access by index ────────────────────────────────────────
    //
    // `file.message(i)` seeks directly to message i using the offset
    // map built during the initial scan.  Combine with
    // `file.messageMetadata(i)` for metadata-only inspection.
    console.log('\n3. Random access — file.message(i):');
    const file2 = await TensogramFile.open(path);
    try {
      const first = await file2.messageMetadata(0);
      const last = await file2.messageMetadata(file2.messageCount - 1);
      const paramOf = (m: GlobalMetadata) =>
        ((m.base?.[0] as { mars?: { param: string } } | undefined)?.mars?.param ?? '?');
      console.log(`   first: param=${paramOf(first)}`);
      console.log(`   last : param=${paramOf(last)}`);

      const mid = await file2.message(1);
      try {
        const shape = mid.objects[0].descriptor.shape;
        console.log(`   file.message(1).objects[0].shape = [${shape.join(', ')}]`);
      } finally {
        mid.close();
      }
    } finally {
      file2.close();
    }
  } finally {
    rmSync(tmp, { recursive: true });
  }

  console.log('\nIterator APIs OK.');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 06 — TensogramFile: Node filesystem + fetch (TypeScript)
 *
 * Demonstrates the three ways to open a `.tgm` file:
 * 1. {@link TensogramFile.open} — from a local path (Node)
 * 2. {@link TensogramFile.fromUrl} — over HTTP(S) via fetch (browser + Node)
 * 3. {@link TensogramFile.fromBytes} — from an in-memory buffer
 *
 * All three return the same shape, with O(1) random access to messages.
 */

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  encode,
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

function buildMessage(values: number[]): Uint8Array {
  return encode({ version: 3 }, [
    { descriptor: describe([values.length], 'float32'), data: new Float32Array(values) },
  ]);
}

async function sectionOpen(): Promise<void> {
  console.log('─── 1. TensogramFile.open (Node fs) ─────────────────────');
  const dir = mkdtempSync(join(tmpdir(), 'tgm-ts-example-'));
  try {
    const path = join(dir, 'demo.tgm');
    const m1 = buildMessage([1, 2, 3]);
    const m2 = buildMessage([10, 20]);
    const m3 = buildMessage([100]);
    const combined = new Uint8Array(m1.byteLength + m2.byteLength + m3.byteLength);
    combined.set(m1, 0);
    combined.set(m2, m1.byteLength);
    combined.set(m3, m1.byteLength + m2.byteLength);
    writeFileSync(path, combined);

    const file = await TensogramFile.open(path);
    try {
      console.log(`  path:         ${path}`);
      console.log(`  source:       ${file.source}`);
      console.log(`  byteLength:   ${file.byteLength}`);
      console.log(`  messages:     ${file.messageCount}`);

      const second = await file.message(1);
      console.log(`  message[1]:   ${Array.from(second.objects[0].data() as Float32Array).join(', ')}`);
      second.close();
    } finally {
      file.close();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function sectionFromUrl(): Promise<void> {
  console.log('\n─── 2. TensogramFile.fromUrl (fetch) ────────────────────');
  const bytes = buildMessage([0.1, 0.2, 0.3, 0.4]);

  // Use a fake fetch that returns the bytes directly. In real browser code
  // you would just call TensogramFile.fromUrl(url).
  const fakeFetch: typeof globalThis.fetch = async () => {
    const copy = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(copy).set(bytes);
    return new Response(copy, { status: 200 });
  };

  const file = await TensogramFile.fromUrl('https://example.com/demo.tgm', {
    fetch: fakeFetch,
    headers: { 'x-example': '1' },
  });
  try {
    console.log(`  source:       ${file.source}`);
    console.log(`  messageCount: ${file.messageCount}`);

    for await (const msg of file) {
      const arr = msg.objects[0].data() as Float32Array;
      console.log(`  async iter:   ${Array.from(arr).map((v) => v.toFixed(2)).join(', ')}`);
      msg.close();
    }
  } finally {
    file.close();
  }
}

async function sectionFromBytes(): Promise<void> {
  console.log('\n─── 3. TensogramFile.fromBytes (in-memory) ──────────────');
  const bytes = buildMessage([42, 43, 44]);
  const file = TensogramFile.fromBytes(bytes);
  try {
    console.log(`  source:       ${file.source}`);
    console.log(`  messageCount: ${file.messageCount}`);

    // `rawMessage` is async since Scope C — the lazy HTTP backend needs
    // to issue a Range request in the remote case; in-memory backends
    // resolve synchronously under the hood but the signature is unified.
    const raw = await file.rawMessage(0);
    console.log(`  rawMessage(0) length: ${raw.byteLength}`);
    console.log(`  magic:                ${new TextDecoder().decode(raw.subarray(0, 8))}`);
  } finally {
    file.close();
  }
}

async function main(): Promise<void> {
  await init();
  await sectionOpen();
  await sectionFromUrl();
  await sectionFromBytes();
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

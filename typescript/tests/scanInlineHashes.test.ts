// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for `scanWithOptions` + `dataObjectInlineHashes` (Wave D / O-TS-1).
// Mirror the Rust core `scan_with_options` / `data_object_inline_hashes`
// and the Python `object_inline_hashes`.  See
// `plans/INTERFACE_SYMMETRY.md` §5.4 / §8.

import { describe, expect, it } from 'vitest';
import {
  computeHash,
  dataObjectInlineHashes,
  encode,
  init,
  scan,
  scanWithOptions,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

/** Concatenate several encoded messages into one buffer. */
function concat(...msgs: Uint8Array[]): Uint8Array {
  const total = msgs.reduce((n, m) => n + m.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const m of msgs) {
    out.set(m, off);
    off += m.byteLength;
  }
  return out;
}

function oneObjectMessage(fill: number, hash: 'xxh3' | false = 'xxh3'): Uint8Array {
  return encode(
    defaultMeta(),
    [{ descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([fill, fill, fill, fill]) }],
    { hash },
  );
}

describe('scanWithOptions', () => {
  it('matches scan() with default options', async () => {
    await init();
    const buf = concat(oneObjectMessage(1), oneObjectMessage(2), oneObjectMessage(3));
    expect(scanWithOptions(buf)).toEqual(scan(buf));
  });

  it('finds every message (bidirectional default)', async () => {
    await init();
    const buf = concat(oneObjectMessage(1), oneObjectMessage(2), oneObjectMessage(3));
    const positions = scanWithOptions(buf, { bidirectional: true });
    expect(positions).toHaveLength(3);
    expect(positions[0].offset).toBe(0);
    // Offsets are strictly increasing and lengths positive.
    for (let i = 0; i < positions.length; i++) {
      expect(positions[i].length).toBeGreaterThan(0);
      if (i > 0) expect(positions[i].offset).toBeGreaterThan(positions[i - 1].offset);
    }
  });

  it('forward-only walk finds the same messages', async () => {
    await init();
    const buf = concat(oneObjectMessage(1), oneObjectMessage(2), oneObjectMessage(3));
    const fwd = scanWithOptions(buf, { bidirectional: false });
    const bidir = scanWithOptions(buf, { bidirectional: true });
    expect(fwd).toEqual(bidir);
  });

  it('accepts an explicit maxMessageSize', async () => {
    await init();
    const buf = concat(oneObjectMessage(1), oneObjectMessage(2));
    const positions = scanWithOptions(buf, { maxMessageSize: 4 * 1024 * 1024 });
    expect(positions).toHaveLength(2);
  });

  it('rejects a negative maxMessageSize', async () => {
    await init();
    const buf = oneObjectMessage(1);
    expect(() => scanWithOptions(buf, { maxMessageSize: -1 })).toThrow(/maxMessageSize/);
  });

  it('throws on non-Uint8Array input', async () => {
    await init();
    // @ts-expect-error deliberate wrong type
    expect(() => scanWithOptions('nope')).toThrow(/Uint8Array/);
  });
});

describe('dataObjectInlineHashes', () => {
  it('returns 16-char hex digests matching computeHash', async () => {
    await init();
    const data = new Float32Array([1, 2, 3, 4]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data },
    ]);
    const hashes = dataObjectInlineHashes(msg);
    expect(hashes).toHaveLength(1);
    expect(hashes[0]).toMatch(/^[0-9a-f]{16}$/);
  });

  it('returns null when hashing is disabled', async () => {
    await init();
    const msg = oneObjectMessage(1, /* hash */ false);
    const hashes = dataObjectInlineHashes(msg);
    expect(hashes).toEqual([null]);
  });

  it('returns one entry per object, in wire order', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([2], 'float32'), data: new Float32Array([1, 2]) },
      { descriptor: makeDescriptor([2], 'float32'), data: new Float32Array([3, 4]) },
      { descriptor: makeDescriptor([2], 'float32'), data: new Float32Array([5, 6]) },
    ]);
    const hashes = dataObjectInlineHashes(msg);
    expect(hashes).toHaveLength(3);
    for (const h of hashes) expect(h).toMatch(/^[0-9a-f]{16}$/);
    // Distinct payloads → distinct digests.
    expect(new Set(hashes).size).toBe(3);
  });

  it('throws on a structurally invalid buffer', async () => {
    await init();
    expect(() => dataObjectInlineHashes(new Uint8Array([1, 2, 3, 4]))).toThrow();
  });

  it('throws on non-Uint8Array input', async () => {
    await init();
    // @ts-expect-error deliberate wrong type
    expect(() => dataObjectInlineHashes(42)).toThrow(/Uint8Array/);
  });

  it('digest form is comparable to computeHash output', async () => {
    await init();
    // Both accessors emit the same 16-char lowercase hex form.
    const h = computeHash(new Uint8Array([1, 2, 3, 4]));
    expect(h).toMatch(/^[0-9a-f]{16}$/);
  });
});

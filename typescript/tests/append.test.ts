// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `TensogramFile#append` — Node local-file append tests.  Covers:
 *
 * - A fresh `.tgm` file created via `open(path)` + `append(...)`
 *   gains a new message; `messageCount` reflects it; every message
 *   (including the new one) decodes correctly.
 * - `fromBytes` and `fromUrl` both reject `append` with
 *   `InvalidArgumentError` — matches the Rust / Python / C++ contract.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  encode,
  InvalidArgumentError,
  TensogramFile,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.1 — TensogramFile#append', () => {
  initOnce();

  let tmp: string;
  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-append-'));
  });
  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it('appends a message to an empty file opened via open()', async () => {
    const path = join(tmp, 'grow.tgm');
    // Seed the file with one message so open() has something to index.
    writeFileSync(
      path,
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([1, 2]),
        },
      ]),
    );

    const file = await TensogramFile.open(path);
    try {
      expect(file.messageCount).toBe(1);
      await file.append(defaultMeta(), [
        {
          descriptor: makeDescriptor([3], 'float32'),
          data: new Float32Array([10, 20, 30]),
        },
      ]);
      expect(file.messageCount).toBe(2);
      const second = await file.message(1);
      expect(Array.from(second.objects[0].data() as Float32Array)).toEqual([10, 20, 30]);
      second.close();
    } finally {
      file.close();
    }
  });

  it('flushes to disk — a reopened handle sees the appended message', async () => {
    const path = join(tmp, 'persist.tgm');
    writeFileSync(
      path,
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([0]),
        },
      ]),
    );

    const writer = await TensogramFile.open(path);
    try {
      await writer.append(defaultMeta(), [
        {
          descriptor: makeDescriptor([2], 'float64'),
          data: new Float64Array([7.5, 8.5]),
        },
      ]);
    } finally {
      writer.close();
    }

    const reader = await TensogramFile.open(path);
    try {
      expect(reader.messageCount).toBe(2);
      const m1 = await reader.message(1);
      expect(Array.from(m1.objects[0].data() as Float64Array)).toEqual([7.5, 8.5]);
      m1.close();
    } finally {
      reader.close();
    }
  });

  it('respects options.hash to skip hashing', async () => {
    const path = join(tmp, 'nohash.tgm');
    writeFileSync(path, new Uint8Array(0));
    // The seed is empty — open() tolerates no messages.
    // NB: appendFile will create the file if it doesn't exist, but
    // open() would error on a fully-missing file, so seed with an
    // empty file.
    const file = await TensogramFile.open(path);
    try {
      await file.append(
        defaultMeta(),
        [
          {
            descriptor: makeDescriptor([1], 'uint8'),
            data: new Uint8Array([42]),
          },
        ],
        { hash: false },
      );
      expect(file.messageCount).toBe(1);
      const m = await file.message(0);
      // Descriptor should have no `hash` field.
      expect(m.objects[0].descriptor.hash).toBeUndefined();
      m.close();
    } finally {
      file.close();
    }
  });

  it('rejects append on fromBytes-backed file with InvalidArgumentError', async () => {
    const bytes = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1], 'uint8'),
        data: new Uint8Array([1]),
      },
    ]);
    const file = TensogramFile.fromBytes(bytes);
    try {
      await expect(
        file.append(defaultMeta(), [
          {
            descriptor: makeDescriptor([1], 'uint8'),
            data: new Uint8Array([2]),
          },
        ]),
      ).rejects.toThrow(InvalidArgumentError);
    } finally {
      file.close();
    }
  });

  it('rejects append on fromUrl-backed file', async () => {
    // Use a fake fetch so we don't need network access.
    const bytes = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1], 'uint8'),
        data: new Uint8Array([1]),
      },
    ]);
    const fakeFetch: typeof globalThis.fetch = async () => {
      // Probe path returns 200 without Accept-Ranges → eager fallback.
      const copy = new ArrayBuffer(bytes.byteLength);
      new Uint8Array(copy).set(bytes);
      return new Response(copy, { status: 200 });
    };
    const file = await TensogramFile.fromUrl('https://example.invalid/a.tgm', {
      fetch: fakeFetch,
    });
    try {
      await expect(
        file.append(defaultMeta(), [
          {
            descriptor: makeDescriptor([1], 'uint8'),
            data: new Uint8Array([2]),
          },
        ]),
      ).rejects.toThrow(InvalidArgumentError);
    } finally {
      file.close();
    }
  });

  it('rejects append after close()', async () => {
    const path = join(tmp, 'closed.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    file.close();
    await expect(
      file.append(defaultMeta(), [
        {
          descriptor: makeDescriptor([1], 'uint8'),
          data: new Uint8Array([1]),
        },
      ]),
    ).rejects.toThrow(InvalidArgumentError);
  });

  it('two consecutive appends both land on disk', async () => {
    const path = join(tmp, 'two.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    try {
      await file.append(defaultMeta(), [
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([1]),
        },
      ]);
      await file.append(defaultMeta(), [
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([2, 3]),
        },
      ]);
      expect(file.messageCount).toBe(2);
      const m0 = await file.message(0);
      const m1 = await file.message(1);
      expect(Array.from(m0.objects[0].data() as Float32Array)).toEqual([1]);
      expect(Array.from(m1.objects[0].data() as Float32Array)).toEqual([2, 3]);
      m0.close();
      m1.close();
    } finally {
      file.close();
    }
  });

  it('appended messages round-trip via a fresh open()', async () => {
    // Per-call UUID/timestamp in `_reserved_` means two independent
    // `encode()` calls for the same inputs are NOT byte-equal — so we
    // assert semantic equality (decoded values) across a reopen.
    const path = join(tmp, 'parity.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    const values = [
      new Float32Array([1, 2]),
      new Float32Array([3, 4, 5]),
    ];
    try {
      for (const v of values) {
        await file.append(defaultMeta(), [
          { descriptor: makeDescriptor([v.length], 'float32'), data: v },
        ]);
      }
    } finally {
      file.close();
    }

    const reopen = await TensogramFile.open(path);
    try {
      expect(reopen.messageCount).toBe(values.length);
      for (let i = 0; i < values.length; i++) {
        const msg = await reopen.message(i);
        try {
          expect(Array.from(msg.objects[0].data() as Float32Array)).toEqual(
            Array.from(values[i]),
          );
        } finally {
          msg.close();
        }
      }
    } finally {
      reopen.close();
    }
  });
});

/**
 * BUG-TS regression — `TensogramFile#append` previously forwarded ONLY
 * `options.hash` to `encode()`, silently dropping every other
 * `AppendOptions` field (`allowNan`, `allowInf`, `*MaskMethod`,
 * `smallMaskThresholdBytes`).  These tests prove the full option
 * surface now reaches `encode()`.
 *
 * See `plans/INTERFACE_SYMMETRY.md` §5.4 / §6 (BUG-TS).
 */
describe('BUG-TS — TensogramFile#append forwards all AppendOptions (masks)', () => {
  initOnce();

  let tmp: string;
  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-append-masks-'));
  });
  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it('a plain append (no mask options) rejects a NaN payload — the precondition', async () => {
    const path = join(tmp, 'nan-reject.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    try {
      await expect(
        file.append(defaultMeta(), [
          { descriptor: makeDescriptor([3], 'float64'), data: new Float64Array([1, NaN, 3]) },
        ]),
      ).rejects.toThrow(/NaN/);
      // encode() throws before any bytes hit disk.
      expect(file.messageCount).toBe(0);
    } finally {
      file.close();
    }
  });

  it('append forwards allowNan — round-trips a NaN a plain append rejects', async () => {
    const path = join(tmp, 'nan-allow.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    try {
      await file.append(
        defaultMeta(),
        [{ descriptor: makeDescriptor([5], 'float64'), data: new Float64Array([1, NaN, 3, NaN, 5]) }],
        { allowNan: true, smallMaskThresholdBytes: 0 },
      );
      expect(file.messageCount).toBe(1);
      const m = await file.message(0);
      const out = m.objects[0].data() as Float64Array;
      expect(out[0]).toBe(1);
      expect(Number.isNaN(out[1])).toBe(true);
      expect(out[2]).toBe(3);
      expect(Number.isNaN(out[3])).toBe(true);
      expect(out[4]).toBe(5);
      m.close();
    } finally {
      file.close();
    }
  });

  it('append forwards allowInf — restores +Inf and -Inf across a reopen', async () => {
    const path = join(tmp, 'inf-allow.tgm');
    writeFileSync(path, new Uint8Array(0));
    const writer = await TensogramFile.open(path);
    try {
      await writer.append(
        defaultMeta(),
        [{ descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([1, Infinity, -Infinity, 2]) }],
        { allowInf: true, smallMaskThresholdBytes: 0 },
      );
    } finally {
      writer.close();
    }

    const reader = await TensogramFile.open(path);
    try {
      const m = await reader.message(0);
      const out = m.objects[0].data() as Float32Array;
      expect(out[0]).toBe(1);
      expect(out[1]).toBe(Infinity);
      expect(out[2]).toBe(-Infinity);
      expect(out[3]).toBe(2);
      m.close();
    } finally {
      reader.close();
    }
  });

  it('append forwards nanMaskMethod — an unsupported method (zstd) surfaces the encode error', async () => {
    // This is the sharpest proof the option is forwarded: with the BUG
    // (only `hash` forwarded) `nanMaskMethod` was dropped and the
    // append silently succeeded with the default 'roaring' method.
    // The fix forwards it, so the WASM feature-gate rejects zstd —
    // meaning the option genuinely reached encode().
    const path = join(tmp, 'nan-zstd.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    try {
      await expect(
        file.append(
          defaultMeta(),
          [{ descriptor: makeDescriptor([3], 'float64'), data: new Float64Array([NaN, 1, 2]) }],
          { allowNan: true, nanMaskMethod: 'zstd', smallMaskThresholdBytes: 0 },
        ),
      ).rejects.toThrow(/zstd/i);
      expect(file.messageCount).toBe(0);
    } finally {
      file.close();
    }
  });

  it('append honours a supported nanMaskMethod (lz4) end-to-end', async () => {
    const path = join(tmp, 'nan-lz4.tgm');
    writeFileSync(path, new Uint8Array(0));
    const file = await TensogramFile.open(path);
    try {
      const values = new Float64Array(128);
      for (let i = 0; i < 128; i++) values[i] = i;
      values[10] = NaN;
      values[50] = NaN;
      values[100] = NaN;
      await file.append(
        defaultMeta(),
        [{ descriptor: makeDescriptor([128], 'float64'), data: values }],
        { allowNan: true, nanMaskMethod: 'lz4', smallMaskThresholdBytes: 0 },
      );
      expect(file.messageCount).toBe(1);
      const m = await file.message(0);
      const out = m.objects[0].data() as Float64Array;
      expect(Number.isNaN(out[10])).toBe(true);
      expect(Number.isNaN(out[50])).toBe(true);
      expect(Number.isNaN(out[100])).toBe(true);
      expect(out[11]).toBe(11);
      expect(out[99]).toBe(99);
      m.close();
    } finally {
      file.close();
    }
  });
});

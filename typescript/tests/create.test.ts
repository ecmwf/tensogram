// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `TensogramFile.create` — Node empty-file factory (mirrors the Rust
 * core `tensogram::file::create`).  Covers:
 *
 * - create → append → reopen → messageCount (the primary workflow)
 * - create truncates an existing file to empty
 * - create makes missing parent directories (mirrors `create_dir_all`)
 * - create accepts a `file://` URL
 * - create rejects a non-string / non-URL path
 *
 * See `plans/INTERFACE_SYMMETRY.md` §5.4 / §8.2 (O-TS-1).
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { encode, InvalidArgumentError, TensogramFile } from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('O-TS-1 — TensogramFile.create', () => {
  initOnce();

  let tmp: string;
  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-create-'));
  });
  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it('creates an empty file that append() extends (create → append → reopen → messageCount)', async () => {
    const path = join(tmp, 'series.tgm');
    const file = await TensogramFile.create(path);
    try {
      expect(file.source).toBe('local');
      expect(file.messageCount).toBe(0);
      expect(file.byteLength).toBe(0);
      expect(existsSync(path)).toBe(true);

      await file.append(defaultMeta(), [
        { descriptor: makeDescriptor([2], 'float32'), data: new Float32Array([1, 2]) },
      ]);
      await file.append(defaultMeta(), [
        { descriptor: makeDescriptor([3], 'float64'), data: new Float64Array([3, 4, 5]) },
      ]);
      expect(file.messageCount).toBe(2);
    } finally {
      file.close();
    }

    // Reopen from disk and confirm both messages persisted + decode.
    const reopened = await TensogramFile.open(path);
    try {
      expect(reopened.messageCount).toBe(2);
      const m0 = await reopened.message(0);
      const m1 = await reopened.message(1);
      expect(Array.from(m0.objects[0].data() as Float32Array)).toEqual([1, 2]);
      expect(Array.from(m1.objects[0].data() as Float64Array)).toEqual([3, 4, 5]);
      m0.close();
      m1.close();
    } finally {
      reopened.close();
    }
  });

  it('truncates an existing file to empty', async () => {
    const path = join(tmp, 'existing.tgm');
    // Seed with a real one-message file.
    writeFileSync(
      path,
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([1], 'uint8'), data: new Uint8Array([42]) },
      ]),
    );
    expect(readFileSync(path).byteLength).toBeGreaterThan(0);

    const file = await TensogramFile.create(path);
    try {
      expect(file.messageCount).toBe(0);
      expect(readFileSync(path).byteLength).toBe(0);
    } finally {
      file.close();
    }
  });

  it('creates missing parent directories (mirrors core create_dir_all)', async () => {
    const path = join(tmp, 'nested', 'deep', 'out.tgm');
    expect(existsSync(join(tmp, 'nested'))).toBe(false);

    const file = await TensogramFile.create(path);
    try {
      expect(existsSync(path)).toBe(true);
      await file.append(defaultMeta(), [
        { descriptor: makeDescriptor([1], 'float32'), data: new Float32Array([7]) },
      ]);
      expect(file.messageCount).toBe(1);
      const m = await file.message(0);
      expect(Array.from(m.objects[0].data() as Float32Array)).toEqual([7]);
      m.close();
    } finally {
      file.close();
    }
  });

  it('accepts a file:// URL path', async () => {
    const url = pathToFileURL(join(tmp, 'via-url.tgm'));
    const file = await TensogramFile.create(url);
    try {
      expect(file.messageCount).toBe(0);
      await file.append(defaultMeta(), [
        { descriptor: makeDescriptor([1], 'float32'), data: new Float32Array([1]) },
      ]);
      expect(file.messageCount).toBe(1);
    } finally {
      file.close();
    }
  });

  it('rejects a non-string / non-URL path with InvalidArgumentError', async () => {
    await expect(
      // @ts-expect-error intentional bad input
      TensogramFile.create(123),
    ).rejects.toBeInstanceOf(InvalidArgumentError);
  });
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it } from 'vitest';
import { init, InvalidArgumentError, encode, decode } from '../src/index.js';
import { _resetForTests, getWbg } from '../src/init.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

/**
 * Coverage-closer tests for the init module. The main suite already
 * exercises the default path (init() with no args); these tests target
 * the branches for `wasmInput`, `_resetForTests`, and pre-init accessor
 * errors.
 *
 * Resetting the cached WASM module between tests is intentionally done
 * in the last test so that the other suites don't observe a cleared
 * instance.
 */
describe('init — coverage closers', () => {
  afterEach(async () => {
    // Always leave the process with a usable WASM instance so that any
    // subsequent tests in this file (or later files, when vitest runs
    // them in the same worker) can call encode/decode.
    await init();
  });

  it('getWbg throws before init()', () => {
    _resetForTests();
    expect(() => getWbg()).toThrow(InvalidArgumentError);
  });

  it('getWbg throws with a clear message', () => {
    _resetForTests();
    try {
      getWbg();
      expect.fail('expected throw');
    } catch (err) {
      expect(err).toBeInstanceOf(InvalidArgumentError);
      expect((err as Error).message).toMatch(/init\(\) must be awaited/);
    }
  });

  it('_resetForTests clears the cached instance', async () => {
    await init();
    expect(() => getWbg()).not.toThrow();
    _resetForTests();
    expect(() => getWbg()).toThrow(InvalidArgumentError);
  });

  it('init accepts an explicit wasmInput of raw bytes (BufferSource)', async () => {
    _resetForTests();
    const wasmUrl = new URL(
      '../wasm/tensogram_wasm_bg.wasm',
      import.meta.url,
    );
    const wasmBytes = readFileSync(fileURLToPath(wasmUrl));
    const buf = wasmBytes.buffer.slice(
      wasmBytes.byteOffset,
      wasmBytes.byteOffset + wasmBytes.byteLength,
    );
    await init({ wasmInput: buf as ArrayBuffer });
    // Now we can encode/decode.
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([3], 'float32'), data: new Float32Array([1, 2, 3]) },
    ]);
    const decoded = decode(msg);
    expect(decoded.objects).toHaveLength(1);
    decoded.close();
  });

  it('init is idempotent after an explicit wasmInput init', async () => {
    // After the previous test set a custom input, calling init() again
    // must be a no-op. Calling init() multiple times with different inputs
    // is not supposed to reload — the documented semantics are "first win".
    await expect(init()).resolves.toBeUndefined();
    await expect(init()).resolves.toBeUndefined();
  });
});

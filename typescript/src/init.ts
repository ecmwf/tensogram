// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * WASM initialisation.
 *
 * `init()` is idempotent: subsequent calls reuse the same instance.
 * Must be awaited before any other exported function is called.
 *
 * In browsers the default behaviour matches `wasm-pack --target web` —
 * the `.wasm` file is fetched relative to `import.meta.url`. In Node
 * (≥ 20) we read the file from disk explicitly, which avoids depending
 * on Node's experimental `file://` `fetch` support.
 */

import wbgInit, * as wbg from '../wasm/tensogram_wasm.js';
import type * as wbgTypes from '../wasm/tensogram_wasm.js';

import { InvalidArgumentError } from './errors.js';

/** Options for {@link init}. */
export interface InitOptions {
  /**
   * Explicit location of the `.wasm` binary. Accepts a `URL`, a
   * `Response`, a `WebAssembly.Module`, or raw bytes.
   *
   * When omitted, the wrapper picks the default for the current
   * runtime (fetch in browsers, file read in Node).
   */
  wasmInput?: URL | Response | WebAssembly.Module | BufferSource;
}

type WbgModule = typeof wbgTypes & { default: typeof wbgInit };

let instance: WbgModule | undefined;
let initPromise: Promise<WbgModule> | undefined;

/**
 * Initialise the WebAssembly module. Idempotent across concurrent
 * callers — only a single instantiation happens per process.
 *
 * @returns a resolved promise once the WASM module is ready.
 */
export function init(opts?: InitOptions): Promise<void> {
  if (instance !== undefined) return Promise.resolve();
  if (initPromise !== undefined) return initPromise.then(() => undefined);

  initPromise = doInit(opts).then((m) => {
    instance = m;
    return m;
  });
  return initPromise.then(() => undefined);
}

/**
 * Internal accessor used by `encode`, `decode`, etc. Throws if
 * `init()` has not been awaited yet.
 */
export function getWbg(): WbgModule {
  if (instance === undefined) {
    throw new InvalidArgumentError(
      'tensogram: init() must be awaited before using any other API',
      'tensogram: init() must be awaited before using any other API',
    );
  }
  return instance;
}

/**
 * Test helper: reset the cached instance. Intended for unit tests
 * that want to exercise the init path repeatedly.
 *
 * @internal
 */
export function _resetForTests(): void {
  instance = undefined;
  initPromise = undefined;
}

async function doInit(opts?: InitOptions): Promise<WbgModule> {
  const wbgAny = wbg as unknown as Record<string, unknown>;

  if (opts?.wasmInput !== undefined) {
    await wbgInit({ module_or_path: opts.wasmInput });
    return { ...wbg, default: wbgInit } as WbgModule;
  }

  // Node: read the .wasm file off disk explicitly. Avoids depending on
  // Node's file:// fetch support (Node < 21 can't fetch file URLs).
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    const [{ readFile }, { fileURLToPath }] = await Promise.all([
      import('node:fs/promises'),
      import('node:url'),
    ]);
    // The wasm file lives next to the generated glue in `../wasm/`.
    const wasmUrl = new URL('../wasm/tensogram_wasm_bg.wasm', import.meta.url);
    const wasmPath = fileURLToPath(wasmUrl);
    const bytes = await readFile(wasmPath);
    // wasm-bindgen expects a real ArrayBuffer, not a Node Buffer view; copy
    // into a fresh ArrayBuffer so `instanceof ArrayBuffer` works.
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    await wbgInit({ module_or_path: buf as ArrayBuffer });
    return { ...wbg, default: wbgInit } as WbgModule;
  }

  // Browser (or any runtime with WASM streaming): fall through to
  // wasm-pack's built-in fetch-from-import.meta.url path.
  // The `_` marker suppresses a false-positive lint about the empty
  // call expression; it's the documented default init path.
  void wbgAny;
  await wbgInit();
  return { ...wbg, default: wbgInit } as WbgModule;
}

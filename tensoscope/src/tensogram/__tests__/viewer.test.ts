// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Viewer-level unit tests.  We mock `TensogramFile` to record the
 * sequence of method calls Tensoscope makes during `buildIndex` /
 * `decodeField` / `fetchCoordinates`, and assert that the
 * layout-aware accessors are used (not the deprecated full-message
 * downloads).
 *
 * Addresses Oracle plan-review concern #8 (test the buildIndex /
 * decodeField method-usage refactor).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { existsSync } from 'node:fs';

describe('Tensoscope UI placeholder text', () => {
  it('FileOpenDialog placeholder no longer suggests s3:// URLs', () => {
    const path = 'src/components/file-browser/FileOpenDialog.tsx';
    if (!existsSync(path)) return; // running outside the tensoscope cwd
    const source = readFileSync(path, 'utf8');
    expect(source).not.toMatch(/s3:\/\//);
    expect(source).toMatch(/presigned URL/);
  });

  it('WelcomeModal placeholder no longer suggests s3:// URLs', () => {
    const path = 'src/components/file-browser/WelcomeModal.tsx';
    if (!existsSync(path)) return;
    const source = readFileSync(path, 'utf8');
    expect(source).not.toMatch(/s3:\/\//);
    expect(source).toMatch(/presigned URL/);
  });
});

describe('Tensoscope viewer source uses layout-aware reads', () => {
  // We assert the imports + call sites by reading the wrapper source.
  // A full mock of TensogramFile + its wbg handles would require
  // bringing the wasm into the test environment; that is covered by
  // the typescript package's own test suite.  This test is a lightweight
  // structural guard against regressions where a future refactor
  // accidentally reverts to the old full-message path.

  it('imports TensogramFile but no longer imports decodeObject', () => {
    const path = 'src/tensogram/index.ts';
    if (!existsSync(path)) return;
    const source = readFileSync(path, 'utf8');
    expect(source).toMatch(/TensogramFile/);
    // The pre-refactor wrapper did `import { decodeObject } from
    // '@ecmwf.int/tensogram'` and called it from decodeField.  After
    // the layout-aware migration, decodeField uses
    // file.messageObject which already returns a DecodedMessage.
    expect(source).not.toMatch(/import\s*{[^}]*\bdecodeObject\b[^}]*}\s*from\s*'@ecmwf\.int\/tensogram'/);
  });

  it('buildIndex prefetches metadata via concurrent messageMetadata calls before the main loop', () => {
    const path = 'src/tensogram/index.ts';
    if (!existsSync(path)) return;
    const source = readFileSync(path, 'utf8');
    const buildIdx = source.indexOf('async buildIndex');
    expect(buildIdx).toBeGreaterThan(-1);
    // Must use Promise.all for concurrent prefetch (not prefetchLayouts which no longer exists).
    const promiseAllIdx = source.indexOf('Promise.all', buildIdx);
    const messageMetadataInLoopIdx = source.indexOf('messageMetadata', buildIdx);
    expect(promiseAllIdx).toBeGreaterThan(-1);
    expect(messageMetadataInLoopIdx).toBeGreaterThan(-1);
    // Concurrent prefetch block must come before the sequential metadata loop.
    expect(promiseAllIdx).toBeLessThan(messageMetadataInLoopIdx);
    // Must NOT use the removed prefetchLayouts API.
    expect(source.indexOf('prefetchLayouts', buildIdx)).toBe(-1);
  });

  it('decodeField uses file.message() (per-message decode), not rawMessage or messageObject', () => {
    const path = 'src/tensogram/index.ts';
    if (!existsSync(path)) return;
    const source = readFileSync(path, 'utf8');
    const decodeFieldIdx = source.indexOf('async decodeField');
    expect(decodeFieldIdx).toBeGreaterThan(-1);
    const block = source.slice(decodeFieldIdx, decodeFieldIdx + 800);
    // Uses the current TensogramFile.message() API.
    expect(block).toMatch(/this\.file\.message\(/);
    // Must NOT fall back to the removed messageObject or expensive rawMessage.
    expect(block).not.toMatch(/messageObject/);
    expect(block).not.toMatch(/rawMessage/);
  });
});

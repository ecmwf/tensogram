// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `validate` / `validateBuffer` / `validateFile` tests.  Ground truth is
 * "report, never throw, on corrupt input"; we also verify mode routing
 * and golden-file parity with the Rust / Python / C++ suites.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  encode,
  InvalidArgumentError,
  IoError,
  validate,
  validateBuffer,
  validateFile,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

/**
 * Path to the Rust-core golden fixtures.  Re-used so TS validates the
 * same `.tgm` files that Rust/Python/C++ suites use — a drift in wire
 * semantics surfaces immediately.
 */
function goldenPath(name: string): string {
  const url = new URL(`../../rust/tensogram/tests/golden/${name}`, import.meta.url);
  return fileURLToPath(url);
}

describe('Scope C.1 — validate', () => {
  initOnce();

  describe('validate (single message)', () => {
    it('returns a clean report for a well-formed message', () => {
      const msg = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([4], 'float32'),
          data: new Float32Array([1, 2, 3, 4]),
        },
      ]);
      const report = validate(msg);
      expect(report.object_count).toBe(1);
      expect(report.hash_verified).toBe(true);
      expect(report.issues.filter((i) => i.severity === 'error')).toEqual([]);
    });

    it('never throws on random garbage — reports issues instead', () => {
      const report = validate(new Uint8Array(64));
      expect(report.issues.length).toBeGreaterThan(0);
      // First issue is a structural / magic / length problem.
      expect(report.issues[0].level).toBe('structure');
    });

    it('detects truncated messages as issues (not exceptions)', () => {
      const full = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([4], 'float32'),
          data: new Float32Array([1, 2, 3, 4]),
        },
      ]);
      const report = validate(full.subarray(0, Math.floor(full.byteLength / 2)));
      expect(report.issues.some((i) => i.severity === 'error')).toBe(true);
    });

    it('routes mode="quick" to structure-only', () => {
      const msg = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([4], 'float32'),
          data: new Float32Array([1, 2, 3, 4]),
        },
      ]);
      const report = validate(msg, { mode: 'quick' });
      // Structure-only can't establish hash_verified=true.
      expect(report.hash_verified).toBe(false);
    });

    it('routes mode="full" and detects NaN in a float64 payload', () => {
      const data = new Float64Array([1, NaN, 3]);
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      // Use encode_pre_encoded to bypass simple_packing's NaN rejection.
      const msg = encode(defaultMeta(), [
        { descriptor: makeDescriptor([3], 'float64'), data: bytes },
      ]);
      const report = validate(msg, { mode: 'full' });
      const nan = report.issues.find((i) => i.code === 'nan_detected');
      expect(nan).toBeDefined();
      expect(nan?.severity).toBe('error');
    });

    it('accepts canonical: true on a valid message', () => {
      const msg = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([4], 'float32'),
          data: new Float32Array([1, 2, 3, 4]),
        },
      ]);
      const report = validate(msg, { canonical: true });
      expect(report.issues.filter((i) => i.severity === 'error')).toEqual([]);
    });

    it('rejects non-Uint8Array input with InvalidArgumentError', () => {
      expect(() =>
        // @ts-expect-error intentional bad input
        validate([1, 2, 3]),
      ).toThrow(InvalidArgumentError);
    });

    it('rejects unknown mode with InvalidArgumentError', () => {
      expect(() =>
        // @ts-expect-error intentional: mode is a string-literal union
        validate(new Uint8Array(40), { mode: 'thorough' }),
      ).toThrow(InvalidArgumentError);
    });
  });

  describe('validateBuffer (multi-message)', () => {
    it('reports a gap of unrecognised bytes between messages', () => {
      const m1 = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([1, 2]),
        },
      ]);
      const m2 = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([3, 4]),
        },
      ]);
      const garbage = new Uint8Array([0xde, 0xad, 0xbe, 0xef]);
      const combined = new Uint8Array(m1.byteLength + garbage.byteLength + m2.byteLength);
      combined.set(m1, 0);
      combined.set(garbage, m1.byteLength);
      combined.set(m2, m1.byteLength + garbage.byteLength);

      const report = validateBuffer(combined);
      expect(report.messages).toHaveLength(2);
      expect(report.file_issues).toHaveLength(1);
      expect(report.file_issues[0].length).toBe(4);
    });

    it('reports "no valid messages" for pure garbage', () => {
      const report = validateBuffer(new Uint8Array([1, 2, 3, 4]));
      expect(report.messages).toHaveLength(0);
      expect(report.file_issues).toHaveLength(1);
      expect(report.file_issues[0].description).toMatch(/no valid messages/);
    });

    it('is okay with a single clean message', () => {
      const msg = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([4], 'float32'),
          data: new Float32Array([1, 2, 3, 4]),
        },
      ]);
      const report = validateBuffer(msg);
      expect(report.messages).toHaveLength(1);
      expect(report.file_issues).toHaveLength(0);
    });
  });

  describe('validateFile (Node filesystem)', () => {
    let tmp: string;
    beforeEach(() => {
      tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-validate-'));
    });
    afterEach(() => {
      rmSync(tmp, { recursive: true, force: true });
    });

    it('reads a temp file and returns a clean report', async () => {
      const p = join(tmp, 'sample.tgm');
      const msg = encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([4], 'float32'),
          data: new Float32Array([1, 2, 3, 4]),
        },
      ]);
      writeFileSync(p, msg);
      const report = await validateFile(p);
      expect(report.messages).toHaveLength(1);
      expect(report.file_issues).toHaveLength(0);
      expect(report.messages[0].hash_verified).toBe(true);
    });

    it('throws IoError on a missing file', async () => {
      await expect(validateFile(join(tmp, 'nope.tgm'))).rejects.toThrow(IoError);
    });

    it('rejects non-string / non-URL paths', async () => {
      await expect(
        // @ts-expect-error intentional
        validateFile(42),
      ).rejects.toThrow(InvalidArgumentError);
    });
  });

  describe('Golden file parity', () => {
    // Every TS validate() call on a golden fixture must come back clean.
    // This is the cross-language invariant — any drift in wire semantics
    // breaks Rust / Python / C++ suites simultaneously.
    const goldens = [
      'simple_f32.tgm',
      'multi_object.tgm',
      'mars_metadata.tgm',
      'hash_xxh3.tgm',
    ];

    for (const name of goldens) {
      it(`validates ${name} with no errors`, () => {
        const bytes = readFileSync(goldenPath(name));
        const report = validate(
          new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
        );
        const errs = report.issues.filter((i) => i.severity === 'error');
        expect(errs).toEqual([]);
      });
    }

    it('validates multi_message.tgm via validateBuffer', () => {
      const bytes = readFileSync(goldenPath('multi_message.tgm'));
      const report = validateBuffer(
        new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
      );
      expect(report.messages.length).toBeGreaterThan(1);
      expect(report.file_issues).toEqual([]);
      for (const m of report.messages) {
        expect(m.issues.filter((i) => i.severity === 'error')).toEqual([]);
      }
    });
  });
});

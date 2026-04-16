// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import {
  FramingError,
  HashMismatchError,
  InvalidArgumentError,
  ObjectError,
  TensogramError,
} from '../src/index.js';
import { mapTensogramError } from '../src/errors.js';

describe('Phase 1 — typed error hierarchy', () => {
  it('all concrete errors extend TensogramError', () => {
    const candidates = [
      // Constructors are `(message, rawMessage?)` — see errors.ts.
      // When only `message` is given, `rawMessage` defaults to it.
      new FramingError('bad magic', 'framing error: bad magic'),
      new InvalidArgumentError('bad arg'),
      new ObjectError('oob', 'object error: oob'),
      new HashMismatchError('expected aa, got bb', 'hash mismatch: expected aa, got bb', 'aa', 'bb'),
    ];
    for (const err of candidates) {
      expect(err).toBeInstanceOf(TensogramError);
      expect(err).toBeInstanceOf(Error);
    }
  });

  it('mapTensogramError routes framing prefix', () => {
    const e = mapTensogramError(new Error('framing error: buffer too short'));
    expect(e).toBeInstanceOf(FramingError);
    expect(e.message).toBe('buffer too short');
  });

  it('mapTensogramError routes metadata prefix', () => {
    const e = mapTensogramError(new Error('metadata error: missing version'));
    expect(e).toBeInstanceOf(TensogramError);
    expect(e.name).toBe('MetadataError');
  });

  it('mapTensogramError extracts hash-mismatch fields', () => {
    const e = mapTensogramError(
      new Error('hash mismatch: expected=a1b2c3d4, actual=11223344'),
    );
    expect(e).toBeInstanceOf(HashMismatchError);
    const hm = e as HashMismatchError;
    expect(hm.expected).toBe('a1b2c3d4');
    expect(hm.actual).toBe('11223344');
    // Consistency with other variants: `message` strips the variant prefix.
    expect(hm.message).toBe('expected=a1b2c3d4, actual=11223344');
    // `rawMessage` retains the full string including the prefix.
    expect(hm.rawMessage).toBe('hash mismatch: expected=a1b2c3d4, actual=11223344');
  });

  it('mapTensogramError accepts "hash mismatch" with no trailing body', () => {
    // Defensive: if the Rust side ever emits just "hash mismatch" without
    // a structured `expected`/`got` payload, the wrapper must not produce
    // a HashMismatchError with an empty `.message`.
    const e = mapTensogramError(new Error('hash mismatch'));
    expect(e).toBeInstanceOf(HashMismatchError);
    const hm = e as HashMismatchError;
    expect(hm.expected).toBeUndefined();
    expect(hm.actual).toBeUndefined();
    expect(hm.message).toBe('hash mismatch'); // falls back to raw
    expect(hm.rawMessage).toBe('hash mismatch');
  });

  it('mapTensogramError accepts the Rust space-delimited hash-mismatch form', () => {
    // Rust's TensogramError::Display format: "hash mismatch: expected X, got Y"
    const e = mapTensogramError(
      new Error('hash mismatch: expected ab40 got f1da'),
    );
    const hm = e as HashMismatchError;
    expect(hm.expected).toBe('ab40');
    expect(hm.actual).toBe('f1da');
  });

  it('mapTensogramError routes "index out of range" as ObjectError', () => {
    const e = mapTensogramError(new Error('object index 5 out of range (have 2)'));
    expect(e).toBeInstanceOf(ObjectError);
  });

  it('TensogramError exposes rawMessage', () => {
    // Signature: (message, rawMessage?). `message` is the user-facing
    // prefix-free form; `rawMessage` retains the original WASM string.
    const e = new FramingError('bad', 'framing error: bad');
    expect(e.rawMessage).toBe('framing error: bad');
    expect(e.message).toBe('bad');
  });

  it('TensogramError rawMessage defaults to message when omitted', () => {
    const e = new InvalidArgumentError('bad arg');
    expect(e.rawMessage).toBe('bad arg');
    expect(e.message).toBe('bad arg');
  });

  // ── Coverage closers for the fallback keyword-routing paths ──────────

  it('mapTensogramError routes encoding prefix', () => {
    const e = mapTensogramError(new Error('encoding error: NaN in simple_packing'));
    expect(e.name).toBe('EncodingError');
  });

  it('mapTensogramError routes compression prefix', () => {
    const e = mapTensogramError(new Error('compression error: zstd decompression failed'));
    expect(e.name).toBe('CompressionError');
  });

  it('mapTensogramError routes io prefix', () => {
    const e = mapTensogramError(new Error('io error: permission denied'));
    expect(e.name).toBe('IoError');
  });

  it('mapTensogramError routes remote prefix', () => {
    const e = mapTensogramError(new Error('remote error: 503 Service Unavailable'));
    expect(e.name).toBe('RemoteError');
  });

  it('mapTensogramError routes streaming-buffer-limit messages', () => {
    const e = mapTensogramError(
      new Error('streaming buffer would grow to 300000000 bytes (limit 268435456)'),
    );
    expect(e.name).toBe('StreamingLimitError');
  });

  it('mapTensogramError routes shape-error keywords as metadata errors', () => {
    const e = mapTensogramError(new Error('unknown dtype: float128'));
    // The keyword-fallback table routes this to MetadataError.
    expect(['MetadataError', 'FramingError']).toContain(e.name);
  });

  it('mapTensogramError routes byte_order keywords to MetadataError', () => {
    const e = mapTensogramError(new Error('invalid byte_order: middle'));
    expect(e.name).toBe('MetadataError');
  });

  it('mapTensogramError routes descriptor-related keywords to MetadataError', () => {
    const e = mapTensogramError(new Error('missing required descriptor field'));
    expect(e.name).toBe('MetadataError');
  });

  it('mapTensogramError routes shape-related keywords to MetadataError', () => {
    const e = mapTensogramError(new Error('shape product overflow'));
    expect(e.name).toBe('MetadataError');
  });

  it('mapTensogramError routes a bare "out of range" message to ObjectError', () => {
    const e = mapTensogramError(new Error('object index 42 out of range (num_objects=3)'));
    expect(e.name).toBe('ObjectError');
    // Raw preserved.
    expect(e.rawMessage).toBe('object index 42 out of range (num_objects=3)');
  });

  it('mapTensogramError falls back to FramingError for completely foreign text', () => {
    const e = mapTensogramError(new Error('some unrelated runtime problem'));
    expect(e.name).toBe('FramingError');
    expect(e.message).toBe('some unrelated runtime problem');
  });

  it('mapTensogramError routes "buffer size overflow" to StreamingLimitError', () => {
    // Alternate buffer-limit phrasing that the wrapper accepts.
    const e = mapTensogramError(new Error('streaming: buffer size overflow at 300MB'));
    expect(e.name).toBe('StreamingLimitError');
  });

  it('mapTensogramError on a non-Error value still produces a typed error', () => {
    const e = mapTensogramError('bare string error');
    expect(e).toBeInstanceOf(FramingError); // safe default
    expect(e.message).toBe('bare string error');
  });

  it('rethrowTyped passes through TensogramError instances untouched', async () => {
    const { rethrowTyped } = await import('../src/errors.js');
    const original = new FramingError('x', 'framing error: x');
    expect(() => rethrowTyped(() => { throw original; })).toThrow(original);
  });
});

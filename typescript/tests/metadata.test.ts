// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import {
  cborValuesEqual,
  computeCommon,
  decode,
  decodeMetadata,
  encode,
  getMetaKey,
  init,
} from '../src/index.js';
import { makeDescriptor } from './helpers.js';

describe('Phase 2 — metadata helpers', () => {
  it('getMetaKey finds a key in base[0]', async () => {
    await init();
    const msg = encode(
      {
base: [
          { mars: { param: '2t', class: 'od' } },
        ],
      },
      [
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([0]),
        },
      ],
    );
    const meta = decodeMetadata(msg);
    expect(getMetaKey(meta, 'mars.param')).toBe('2t');
    expect(getMetaKey(meta, 'mars.class')).toBe('od');
  });

  it('getMetaKey uses first-match semantics across base entries', async () => {
    await init();
    const msg = encode(
      {
base: [
          { mars: { param: '2t' } },
          { mars: { param: 'msl' } },
        ],
      },
      [
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([0]),
        },
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([1]),
        },
      ],
    );
    const meta = decodeMetadata(msg);
    expect(getMetaKey(meta, 'mars.param')).toBe('2t');
  });

  it('getMetaKey falls back to _extra_', async () => {
    await init();
    const msg = encode(
      {_extra_: { source: 'ifs-cycle49r2' } },
      [],
    );
    const meta = decodeMetadata(msg);
    expect(getMetaKey(meta, 'source')).toBe('ifs-cycle49r2');
    expect(getMetaKey(meta, '_extra_.source')).toBe('ifs-cycle49r2');
    expect(getMetaKey(meta, 'extra.source')).toBe('ifs-cycle49r2');
  });

  it('getMetaKey hides _reserved_', async () => {
    await init();
    // The encoder populates _reserved_ itself; we use it here via decode.
    const msg = encode({}, []);
    const meta = decodeMetadata(msg);
    // _reserved_ exists on the decoded side, but the lookup must NOT see it.
    expect(meta._reserved_).toBeDefined();
    expect(getMetaKey(meta, '_reserved_')).toBeUndefined();
    expect(getMetaKey(meta, '_reserved_.encoder.name')).toBeUndefined();
    expect(getMetaKey(meta, 'reserved.encoder')).toBeUndefined();
  });

  it('getMetaKey returns undefined for missing keys', () => {
    const meta = {base: [{ foo: 1 }], _extra_: {} };
    expect(getMetaKey(meta, 'nonexistent')).toBeUndefined();
    expect(getMetaKey(meta, 'foo.bar.baz')).toBeUndefined();
    expect(getMetaKey(meta, '')).toBeUndefined();
  });

  it('computeCommon extracts shared keys across base entries', () => {
    const meta = {
base: [
        { class: 'od', type: 'fc', param: '2t' },
        { class: 'od', type: 'fc', param: 'msl' },
        { class: 'od', type: 'fc', param: '10u' },
      ],
    };
    const common = computeCommon(meta);
    expect(common).toEqual({ class: 'od', type: 'fc' });
    expect('param' in common).toBe(false);
  });

  it('computeCommon returns {} for an empty base', () => {
    expect(computeCommon({})).toEqual({});
    expect(computeCommon({base: [] })).toEqual({});
  });

  it('computeCommon returns the full entry if there is only one base entry', () => {
    expect(
      computeCommon({
base: [{ mars: { param: '2t' }, foo: 'bar' }],
      }),
    ).toEqual({ mars: { param: '2t' }, foo: 'bar' });
  });

  it('computeCommon skips _reserved_', () => {
    expect(
      computeCommon({
base: [
          { class: 'od', _reserved_: { tensor: { ndim: 1 } } },
          { class: 'od', _reserved_: { tensor: { ndim: 2 } } },
        ],
      }),
    ).toEqual({ class: 'od' });
  });

  it('computeCommon handles nested maps correctly', () => {
    expect(
      computeCommon({
base: [
          { mars: { class: 'od', param: '2t' } },
          { mars: { class: 'od', param: 'msl' } },
        ],
      }),
    ).toEqual({});
    // The nested mars objects differ → no common top-level key.
  });

  it('cborValuesEqual handles NaN bit-patterns', () => {
    expect(cborValuesEqual(NaN, NaN)).toBe(true);
    expect(cborValuesEqual(1.5, 1.5)).toBe(true);
    expect(cborValuesEqual(1.5, 1.6)).toBe(false);
  });

  it('cborValuesEqual handles nested arrays and objects', () => {
    expect(cborValuesEqual([1, 2, 3], [1, 2, 3])).toBe(true);
    expect(cborValuesEqual([1, 2, 3], [1, 2])).toBe(false);
    expect(cborValuesEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
    expect(cborValuesEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    // Mismatched types
    expect(cborValuesEqual(1, '1')).toBe(false);
    expect(cborValuesEqual([1], { 0: 1 })).toBe(false);
    // null/bool edges
    expect(cborValuesEqual(null, null)).toBe(true);
    expect(cborValuesEqual(true, false)).toBe(false);
  });

  it('getMetaKey returns undefined for empty path', () => {
    const meta = {_extra_: { foo: 'bar' } };
    expect(getMetaKey(meta, '')).toBeUndefined();
  });

  it('getMetaKey _reserved_ prefix always hidden', () => {
    const meta = {_extra_: { reserved: 'visible' } };
    // 'reserved' (without underscore) as an _extra_ key IS visible
    expect(getMetaKey(meta, '_extra_.reserved')).toBe('visible');
    // But '_reserved_' or 'reserved.X' at the top path is always hidden
    expect(getMetaKey(meta, '_reserved_')).toBeUndefined();
    expect(getMetaKey(meta, 'reserved')).toBeUndefined();
  });

  it('decoded metadata has plain objects, not ES Maps', async () => {
    await init();
    const msg = encode(
      {
base: [{ mars: { param: '2t', class: 'od' } }],
      },
      [
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([0]),
        },
      ],
    );
    const decoded = decode(msg);
    const meta = decoded.metadata;
    expect(meta.base).toBeDefined();
    expect(Array.isArray(meta.base)).toBe(true);
    const entry = meta.base![0] as Record<string, unknown>;
    // This is the headline assertion: the metadata is navigable as a
    // plain object, not as an ES Map (would require .get()).
    expect(entry.mars).toBeDefined();
    expect((entry.mars as Record<string, unknown>).param).toBe('2t');
    decoded.close();
  });
});

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
  getMetaBool,
  getMetaBoolAt,
  getMetaFloat,
  getMetaFloatAt,
  getMetaInt,
  getMetaIntAt,
  getMetaKey,
  getMetaKeyAt,
  getMetaString,
  getMetaStringAt,
  hasMetaKey,
  hasMetaKeyAt,
  init,
} from '../src/index.js';
import type { GlobalMetadata } from '../src/index.js';
import { makeDescriptor } from './helpers.js';

describe('metadata helpers (getMetaKey / computeCommon)', () => {
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

describe('symmetry helpers (existence, per-object, typed getters)', () => {
  // A hand-built fixture that exercises every relevant kind: a key present in
  // one object but absent in another, an empty-string value, a zero value, a
  // bool/float/int, _reserved_ (which must stay hidden from path getters), and
  // an _extra_ section (message-level fallback).
  const meta: GlobalMetadata = {
base: [
      {
        mars: { param: '2t', class: 'od' },
        count: 0,
        ratio: 1.5,
        active: false,
        note: '',
        _reserved_: { tensor: { ndim: 1 } },
      },
      {
        mars: { param: 'msl' },
        only1: 'second-object',
        whole: 7,
      },
    ],
    _extra_: { source: 'ifs-cycle49r2', level: 42 },
  };

  // ── hasMetaKey — present / absent ───────────────────────────────────────
  it('hasMetaKey reports presence and absence', () => {
    expect(hasMetaKey(meta, 'mars.param')).toBe(true);
    expect(hasMetaKey(meta, 'source')).toBe(true); // via _extra_ fallback
    expect(hasMetaKey(meta, 'nonexistent')).toBe(false);
    expect(hasMetaKey(meta, '')).toBe(false);
  });

  it('hasMetaKey treats a stored empty-string / zero / false as present', () => {
    expect(hasMetaKey(meta, 'note')).toBe(true); // ""
    expect(hasMetaKey(meta, 'count')).toBe(true); // 0
    expect(hasMetaKey(meta, 'active')).toBe(true); // false
  });

  // ── getMetaKeyAt / hasMetaKeyAt — per-object scoping ────────────────────
  it('getMetaKeyAt scopes to a single base entry (no cross-object match)', () => {
    // `only1` lives in base[1] only.
    expect(getMetaKeyAt(meta, 1, 'only1')).toBe('second-object');
    expect(getMetaKeyAt(meta, 0, 'only1')).toBeUndefined();
    // Per-object first-match must NOT bleed across objects: base[0].mars.param
    // is '2t', base[1].mars.param is 'msl'.
    expect(getMetaKeyAt(meta, 0, 'mars.param')).toBe('2t');
    expect(getMetaKeyAt(meta, 1, 'mars.param')).toBe('msl');
  });

  it('getMetaKeyAt does NOT fall back to _extra_', () => {
    // `source` only exists in _extra_; per-object lookup must not see it.
    expect(getMetaKey(meta, 'source')).toBe('ifs-cycle49r2');
    expect(getMetaKeyAt(meta, 0, 'source')).toBeUndefined();
    expect(getMetaKeyAt(meta, 1, 'source')).toBeUndefined();
  });

  it('getMetaKeyAt guards empty path and out-of-range obj', () => {
    expect(getMetaKeyAt(meta, 0, '')).toBeUndefined();
    expect(getMetaKeyAt(meta, 99, 'mars.param')).toBeUndefined();
    expect(getMetaKeyAt(meta, -1, 'mars.param')).toBeUndefined();
    expect(getMetaKeyAt({}, 0, 'mars.param')).toBeUndefined();
  });

  it('hasMetaKeyAt mirrors getMetaKeyAt scoping', () => {
    expect(hasMetaKeyAt(meta, 1, 'only1')).toBe(true);
    expect(hasMetaKeyAt(meta, 0, 'only1')).toBe(false);
    expect(hasMetaKeyAt(meta, 0, 'note')).toBe(true); // "" is present
    expect(hasMetaKeyAt(meta, 0, 'count')).toBe(true); // 0 is present
    expect(hasMetaKeyAt(meta, 99, 'mars.param')).toBe(false);
  });

  // ── _reserved_ hidden from both message-level and per-object getters ────
  it('_reserved_ stays hidden from message-level and per-object path getters', () => {
    // Present on the object itself…
    expect(meta.base![0]._reserved_).toBeDefined();
    // …but never visible through the path getters.
    expect(getMetaKey(meta, '_reserved_')).toBeUndefined();
    expect(getMetaKey(meta, '_reserved_.tensor.ndim')).toBeUndefined();
    expect(getMetaKey(meta, 'reserved')).toBeUndefined();
    expect(hasMetaKey(meta, '_reserved_.tensor')).toBe(false);
    expect(getMetaKeyAt(meta, 0, '_reserved_')).toBeUndefined();
    expect(getMetaKeyAt(meta, 0, '_reserved_.tensor.ndim')).toBeUndefined();
    expect(hasMetaKeyAt(meta, 0, '_reserved_.tensor')).toBe(false);
  });

  // ── explicit extra. prefix still works via getMetaKey ───────────────────
  it('explicit extra. / _extra_. prefix still targets _extra_ via getMetaKey', () => {
    expect(getMetaKey(meta, 'extra.source')).toBe('ifs-cycle49r2');
    expect(getMetaKey(meta, '_extra_.source')).toBe('ifs-cycle49r2');
    expect(hasMetaKey(meta, 'extra.source')).toBe(true);
    expect(hasMetaKey(meta, '_extra_.level')).toBe(true);
  });

  // ── typed getters — right type returns value, wrong type → undefined ────
  it('getMetaString returns strings, undefined for wrong type / absent', () => {
    expect(getMetaString(meta, 'mars.param')).toBe('2t');
    expect(getMetaString(meta, 'note')).toBe(''); // absent-vs-empty: "" is a value
    expect(getMetaString(meta, 'count')).toBeUndefined(); // number, not string
    expect(getMetaString(meta, 'active')).toBeUndefined(); // bool, not string
    expect(getMetaString(meta, 'nonexistent')).toBeUndefined();
  });

  it('getMetaInt returns integers only, undefined for float / wrong type / absent', () => {
    expect(getMetaInt(meta, 'count')).toBe(0); // absent-vs-zero: 0 is a value
    expect(getMetaInt(meta, '_extra_.level')).toBe(42);
    expect(getMetaInt(meta, 'ratio')).toBeUndefined(); // 1.5 is not an integer
    expect(getMetaInt(meta, 'mars.param')).toBeUndefined(); // string
    expect(getMetaInt(meta, 'active')).toBeUndefined(); // bool
    expect(getMetaInt(meta, 'nonexistent')).toBeUndefined();
  });

  it('getMetaFloat returns any finite number incl. integers (int-widened)', () => {
    expect(getMetaFloat(meta, 'ratio')).toBe(1.5);
    expect(getMetaFloat(meta, 'count')).toBe(0); // int widens to float
    expect(getMetaFloat(meta, '_extra_.level')).toBe(42); // int widens to float
    expect(getMetaFloat(meta, 'mars.param')).toBeUndefined(); // string
    expect(getMetaFloat(meta, 'active')).toBeUndefined(); // bool
    expect(getMetaFloat(meta, 'nonexistent')).toBeUndefined();
  });

  it('numeric getters mirror core: bigint and NaN/±Infinity semantics', () => {
    // Large CBOR integers decode as bigint; NaN/±Infinity are genuine floats.
    const big: GlobalMetadata = {
      base: [{ huge: 9_007_199_254_740_993n, bad: NaN, inf: Infinity }],
      _extra_: {},
    };
    // getMetaInt spans the full integer range (mirrors as_i64/as_u64): bigint through.
    expect(getMetaInt(big, 'huge')).toBe(9_007_199_254_740_993n);
    // getMetaFloat widens the bigint to a number (mirrors `n as f64`).
    expect(getMetaFloat(big, 'huge')).toBe(Number(9_007_199_254_740_993n));
    // NaN / ±Infinity are present float values, not "absent" (mirror as_f64).
    expect(getMetaFloat(big, 'bad')).toBeNaN();
    expect(getMetaFloat(big, 'inf')).toBe(Infinity);
    // ...but NaN is not an integer.
    expect(getMetaInt(big, 'bad')).toBeUndefined();
  });

  it('getMetaBool returns bools only, undefined for wrong type / absent', () => {
    expect(getMetaBool(meta, 'active')).toBe(false); // absent-vs-false: false is a value
    expect(getMetaBool(meta, 'count')).toBeUndefined(); // 0 is not a bool
    expect(getMetaBool(meta, 'mars.param')).toBeUndefined(); // string
    expect(getMetaBool(meta, 'nonexistent')).toBeUndefined();
  });

  // ── typed per-object getters ────────────────────────────────────────────
  it('per-object typed getters honour scoping and type checks', () => {
    expect(getMetaStringAt(meta, 0, 'mars.param')).toBe('2t');
    expect(getMetaStringAt(meta, 1, 'mars.param')).toBe('msl');
    expect(getMetaStringAt(meta, 0, 'only1')).toBeUndefined(); // scoped out
    expect(getMetaStringAt(meta, 1, 'only1')).toBe('second-object');
    expect(getMetaStringAt(meta, 0, 'count')).toBeUndefined(); // number, not string

    expect(getMetaIntAt(meta, 0, 'count')).toBe(0);
    expect(getMetaIntAt(meta, 1, 'whole')).toBe(7);
    expect(getMetaIntAt(meta, 0, 'ratio')).toBeUndefined(); // 1.5 not an int
    expect(getMetaIntAt(meta, 0, 'whole')).toBeUndefined(); // whole lives in base[1]

    expect(getMetaFloatAt(meta, 0, 'ratio')).toBe(1.5);
    expect(getMetaFloatAt(meta, 0, 'count')).toBe(0); // int-widened
    expect(getMetaFloatAt(meta, 0, 'mars.param')).toBeUndefined(); // string

    expect(getMetaBoolAt(meta, 0, 'active')).toBe(false);
    expect(getMetaBoolAt(meta, 1, 'active')).toBeUndefined(); // scoped out
    expect(getMetaBoolAt(meta, 0, 'count')).toBeUndefined(); // number, not bool
  });

  it('per-object typed getters do NOT fall back to _extra_', () => {
    expect(getMetaStringAt(meta, 0, 'source')).toBeUndefined();
    expect(getMetaIntAt(meta, 0, 'level')).toBeUndefined();
  });
});

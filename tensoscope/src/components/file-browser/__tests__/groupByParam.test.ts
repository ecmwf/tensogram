// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Regression tests for `groupByParam`.
 *
 * Before the param-type-coercion fix, a GRIB-derived file whose
 * `mars.param` was an integer code (e.g. 167 for 2m temperature)
 * crashed the sidebar at render time because the sort used
 * `String.prototype.localeCompare` directly on what turned out to
 * be a `number`.  The fix coerces the param value to a string both
 * when keying the group map and when sorting the final output.
 */

import { describe, expect, it } from 'vitest';
import { groupByParam, getPackedLevels } from '../groupByParam';
import type { ObjectInfo } from '../../../tensogram';

function makeVar(
  name: string,
  shape: number[],
  marsParam: string | number | undefined,
  extra: Record<string, unknown> = {},
): ObjectInfo {
  return {
    msgIndex: 0,
    objIndex: 0,
    name,
    shape,
    dtype: 'float32',
    encoding: 'none',
    compression: 'none',
    metadata: {
      mars: marsParam !== undefined ? { param: marsParam, step: 0, ...extra } : extra,
    },
  };
}

describe('groupByParam — param type coercion', () => {
  it('does not crash when mars.param is an integer (GRIB code)', () => {
    const vars: ObjectInfo[] = [
      makeVar('2t', [100], 167),
      makeVar('msl', [100], 2),
      makeVar('t', [100], 130),
    ];
    const groups = groupByParam(vars, 100);
    expect(groups).toHaveLength(3);
    expect(groups.map((g) => g.param).sort()).toEqual(['130', '167', '2']);
  });

  it('still works when mars.param is a string (GRIB short name)', () => {
    const vars: ObjectInfo[] = [
      makeVar('2t', [100], '2t'),
      makeVar('msl', [100], 'msl'),
    ];
    const groups = groupByParam(vars, 100);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.param).sort()).toEqual(['2t', 'msl']);
  });

  it('mixes integer and string params into the same sort', () => {
    const vars: ObjectInfo[] = [
      makeVar('x', [100], 167),
      makeVar('y', [100], '2t'),
      makeVar('z', [100], 130),
    ];
    const groups = groupByParam(vars, 100);
    expect(new Set(groups.map((g) => g.param))).toEqual(new Set(['167', '2t', '130']));
  });

  it('falls back to variable.name when mars.param is absent', () => {
    const vars: ObjectInfo[] = [
      makeVar('alpha', [100], undefined),
      makeVar('beta', [100], undefined),
    ];
    const groups = groupByParam(vars, 100);
    expect(groups.map((g) => g.param).sort()).toEqual(['alpha', 'beta']);
  });

  it('groups objects with the same integer param together', () => {
    const vars: ObjectInfo[] = [
      makeVar('2t', [100], 167, { levtype: 'sfc', step: 0 }),
      makeVar('2t', [100], 167, { levtype: 'sfc', step: 6 }),
      makeVar('2t', [100], 167, { levtype: 'sfc', step: 12 }),
    ];
    const groups = groupByParam(vars, 100);
    expect(groups).toHaveLength(1);
    expect(groups[0].entries).toHaveLength(3);
    expect(groups[0].entries.map((e) => e.step)).toEqual([0, 6, 12]);
  });
});

// ── getPackedLevels: regular_ll GRIB (no anemoi metadata) ────────────────────

describe('getPackedLevels — regular_ll GRIB (meshgridded coords)', () => {
  const COORD = 721 * 1440; // 1_038_240 — meshgridded lat count

  function gribVar(shape: number[], meta: Record<string, unknown> = {}): ObjectInfo {
    return {
      msgIndex: 0, objIndex: 0, name: 'x', shape,
      dtype: 'float32', encoding: 'none', compression: 'none',
      metadata: { mars: { param: 130, step: 0, ...meta } },
    };
  }

  it('returns null for a pure [nLat, nLon] surface variable', () => {
    // product(721, 1440) === COORD → pure spatial, no levels
    expect(getPackedLevels(gribVar([721, 1440]), COORD)).toBeNull();
  });

  it('returns null for a 1-D flat variable', () => {
    expect(getPackedLevels(gribVar([COORD]), COORD)).toBeNull();
  });

  it('returns 13 level indices for [13, 1038240] (flattened spatial)', () => {
    const result = getPackedLevels(gribVar([13, COORD]), COORD);
    expect(result).toHaveLength(13);
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('returns 13 level indices for [13, 721, 1440] (rectangular spatial)', () => {
    const result = getPackedLevels(gribVar([13, 721, 1440]), COORD);
    expect(result).toHaveLength(13);
  });

  it('returns null when coordLength is 0 (coords not yet loaded)', () => {
    expect(getPackedLevels(gribVar([721, 1440]), 0)).toBeNull();
    expect(getPackedLevels(gribVar([13, COORD]), 0)).toBeNull();
  });
});

// ── getPackedLevels: anemoi metadata ────────────────────────────────────────

describe('getPackedLevels — anemoi explicit levels', () => {
  const COORD = 1_038_240;
  const ANO_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50];

  function anoVar(shape: number[], levels?: number[]): ObjectInfo {
    return {
      msgIndex: 0, objIndex: 0, name: 'x', shape,
      dtype: 'float32', encoding: 'none', compression: 'none',
      metadata: {
        mars: { param: 130, step: 0, levtype: 'pl' },
        ...(levels ? { anemoi: { levels } } : {}),
      },
    };
  }

  it('uses anemoi.levels when length matches the level dimension', () => {
    const result = getPackedLevels(anoVar([13, COORD], ANO_LEVELS), COORD);
    expect(result).toEqual(ANO_LEVELS);
  });

  it('returns anemoi.levels when coordLength is 0', () => {
    const result = getPackedLevels(anoVar([13, COORD], ANO_LEVELS), 0);
    expect(result).toEqual(ANO_LEVELS);
  });

  it('returns null when coordLength is 0 and no anemoi.levels', () => {
    expect(getPackedLevels(anoVar([13, COORD]), 0)).toBeNull();
  });
});

// ── groupByParam integration: GRIB pl per-object levels ──────────────────────

describe('groupByParam — GRIB pl per-object level detection', () => {
  const COORD = 721 * 1440;
  const PL_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50];

  function plVar(msgIndex: number, levelist: number): ObjectInfo {
    return {
      msgIndex, objIndex: 0, name: 't', shape: [721, 1440],
      dtype: 'float32', encoding: 'none', compression: 'none',
      metadata: { mars: { param: 130, step: 0, levtype: 'pl', levelist } },
    };
  }

  it('detects 13 per-object pressure levels, 1 step — not 721 packed levels', () => {
    const vars = PL_LEVELS.map((lev, i) => plVar(i, lev));
    const groups = groupByParam(vars, COORD);

    expect(groups).toHaveLength(1);
    const [g] = groups;
    expect(g.levelStrategy.kind).toBe('per-object');
    if (g.levelStrategy.kind === 'per-object') {
      expect(g.levelStrategy.levels).toHaveLength(13);
    }
    // 1 distinct step (step=0 for all)
    expect(new Set(g.entries.map((e) => e.step)).size).toBe(1);
  });

  it('surface variable [721, 1440] gets no levels and 1 step', () => {
    const vars = [
      {
        msgIndex: 0, objIndex: 0, name: '2t', shape: [721, 1440],
        dtype: 'float32', encoding: 'none', compression: 'none',
        metadata: { mars: { param: 167, step: 0, levtype: 'sfc' } },
      } satisfies ObjectInfo,
    ];
    const groups = groupByParam(vars, COORD);
    expect(groups).toHaveLength(1);
    expect(groups[0].levelStrategy.kind).toBe('none');
    expect(groups[0].entries).toHaveLength(1);
  });
});

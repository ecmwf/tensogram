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
import { groupByParam } from '../groupByParam';
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

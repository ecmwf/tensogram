// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Tests for `inferAxesFromMarsGrid` — the helper that synthesises 1-D
 * latitude + longitude axes from `mars.grid = "regular_ll"` + the
 * variable's shape, for GRIB-derived files that ship no explicit
 * coordinate objects.
 */

import { describe, expect, it } from 'vitest';
import { inferAxesFromMarsGrid } from '../index';
import type { ObjectInfo } from '../index';

function makeVar(shape: number[], mars: Record<string, unknown>): ObjectInfo {
  return {
    msgIndex: 0,
    objIndex: 0,
    name: 't',
    shape,
    dtype: 'float32',
    encoding: 'none',
    compression: 'none',
    metadata: { mars },
  };
}

describe('inferAxesFromMarsGrid', () => {
  it('infers 721×1440 global quarter-degree lat/lon from regular_ll', () => {
    const result = inferAxesFromMarsGrid([
      makeVar([721, 1440], { grid: 'regular_ll', param: 167 }),
    ]);
    expect(result).not.toBeNull();
    expect(result!.lat.length).toBe(721);
    expect(result!.lon.length).toBe(1440);
    expect(result!.lat[0]).toBe(90);
    expect(result!.lat[720]).toBe(-90);
    expect(result!.lat[360]).toBeCloseTo(0, 5);
    expect(result!.lon[0]).toBe(-180);
    expect(result!.lon[1439]).toBeCloseTo(179.75, 3);
  });

  it('honours mars.area [north, west, south, east] for regional grids', () => {
    const result = inferAxesFromMarsGrid([
      makeVar([100, 200], {
        grid: 'regular_ll',
        area: [60, -30, 20, 70],
      }),
    ]);
    expect(result).not.toBeNull();
    expect(result!.lat[0]).toBe(60);
    expect(result!.lat[99]).toBeCloseTo(20, 5);
    expect(result!.lon[0]).toBe(-30);
    expect(result!.lon[199]).toBeCloseTo(70, 5);
  });

  it('uses endpoint-exclusive longitude when area wraps a full circle', () => {
    // When no area is specified, defaults are [-180, 180) which spans 360°
    const result = inferAxesFromMarsGrid([
      makeVar([90, 180], { grid: 'regular_ll' }),
    ]);
    expect(result).not.toBeNull();
    // 360 / 180 = 2°, last sample at 178° (endpoints exclusive)
    expect(result!.lon[0]).toBe(-180);
    expect(result!.lon[179]).toBeCloseTo(178, 3);
  });

  it('returns null for unsupported grid kinds', () => {
    expect(
      inferAxesFromMarsGrid([
        makeVar([100, 200], { grid: 'reduced_gg' }),
      ]),
    ).toBeNull();
    expect(
      inferAxesFromMarsGrid([makeVar([100, 200], { grid: 'N320' })]),
    ).toBeNull();
    expect(
      inferAxesFromMarsGrid([makeVar([100, 200], { grid: 'O96' })]),
    ).toBeNull();
  });

  it('returns null when no variable has mars.grid', () => {
    expect(
      inferAxesFromMarsGrid([makeVar([100, 200], { param: 167 })]),
    ).toBeNull();
  });

  it('returns null when the only matching variable is not 2-D', () => {
    expect(
      inferAxesFromMarsGrid([
        makeVar([10, 100, 200], { grid: 'regular_ll' }),
        makeVar([500], { grid: 'regular_ll' }),
      ]),
    ).toBeNull();
  });

  it('picks the first 2-D regular_ll variable and ignores the rest', () => {
    const lat = new Float32Array([1, 2]);
    const lon = new Float32Array([10, 20, 30]);
    const variables = [
      makeVar([5, 2, 3], { grid: 'regular_ll' }),
      makeVar([73, 144], { grid: 'regular_ll' }),
      makeVar([100, 200], { grid: 'regular_ll' }),
    ];
    const result = inferAxesFromMarsGrid(variables);
    expect(result).not.toBeNull();
    expect(result!.lat.length).toBe(73);
    expect(result!.lon.length).toBe(144);
  });

  it('falls back to defaults when mars.area contains a non-finite bigint', () => {
    // Corrupt or absurd CBOR-encoded areas — a bigint that overflows a
    // JS number via Number(v) — must not silently produce Infinity
    // coordinates.  toNumber returns null in that case and the
    // defaults kick in.
    const hugeBigint = 1n << 2000n;
    const result = inferAxesFromMarsGrid([
      makeVar([10, 20], {
        grid: 'regular_ll',
        area: [hugeBigint, 0, -90, 360],
      }),
    ]);
    expect(result).not.toBeNull();
    // Defaults: north=90 (since the overflowing bigint failed toNumber).
    expect(result!.lat[0]).toBe(90);
    expect(Number.isFinite(result!.lat[0])).toBe(true);
    expect(Number.isFinite(result!.lat[result!.lat.length - 1])).toBe(true);
    expect(Number.isFinite(result!.lon[0])).toBe(true);
  });
});

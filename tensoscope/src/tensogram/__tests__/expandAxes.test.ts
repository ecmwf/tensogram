// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Tests for `expandAxesIfRectangularGrid` — the helper that promotes
 * 1-D `latitude` + `longitude` axes to per-point meshgridded arrays
 * when a file ships a regular rectangular grid (the common
 * GRIB / NetCDF layout).
 */

import { describe, expect, it } from 'vitest';
import { expandAxesIfRectangularGrid } from '../index';
import type { ObjectInfo } from '../index';

function makeVar(name: string, shape: number[]): ObjectInfo {
  return {
    msgIndex: 0,
    objIndex: 0,
    name,
    shape,
    dtype: 'float32',
    encoding: 'none',
    compression: 'none',
    metadata: {},
  };
}

describe('expandAxesIfRectangularGrid', () => {
  it('meshgrids 1-D axes when a [nLat, nLon] variable exists', () => {
    const lat = new Float32Array([90, 0, -90]);
    const lon = new Float32Array([0, 90, 180, -90]);
    const variables = [makeVar('2t', [3, 4])];

    const result = expandAxesIfRectangularGrid(lat, lon, variables);
    expect(result).not.toBeNull();
    expect(result!.lat.length).toBe(12);
    expect(result!.lon.length).toBe(12);

    // Row 0 (lat=90): every lon column repeats that lat.
    expect(Array.from(result!.lat.slice(0, 4))).toEqual([90, 90, 90, 90]);
    expect(Array.from(result!.lon.slice(0, 4))).toEqual([0, 90, 180, -90]);

    // Row 1 (lat=0): same lon columns, lat=0 throughout.
    expect(Array.from(result!.lat.slice(4, 8))).toEqual([0, 0, 0, 0]);
    expect(Array.from(result!.lon.slice(4, 8))).toEqual([0, 90, 180, -90]);

    // Row 2 (lat=-90): final row.
    expect(Array.from(result!.lat.slice(8, 12))).toEqual([-90, -90, -90, -90]);
  });

  it('returns null when no [nLat, nLon] variable exists', () => {
    const lat = new Float32Array([90, 0, -90]);
    const lon = new Float32Array([0, 90, 180, -90]);
    const variables = [
      makeVar('x', [5, 4]),
      makeVar('y', [3, 5]),
      makeVar('z', [10]),
    ];
    expect(expandAxesIfRectangularGrid(lat, lon, variables)).toBeNull();
  });

  it('returns null for already-meshgridded coords (no match)', () => {
    // Simulates the `add_coords_meshed.py` output: coords already
    // expanded to length nLat*nLon, and variables still shaped
    // [nLat, nLon] — but no variable has shape
    // [fullGridLength, nLon] so the grid-match search comes back
    // empty.
    const nLat = 4;
    const nLon = 3;
    const latFull = new Float32Array(nLat * nLon);
    const lonFull = new Float32Array(nLat * nLon);
    const variables = [makeVar('2t', [nLat, nLon])];
    expect(expandAxesIfRectangularGrid(latFull, lonFull, variables)).toBeNull();
  });

  it('returns null for empty axes', () => {
    expect(
      expandAxesIfRectangularGrid(new Float32Array(), new Float32Array([1]), []),
    ).toBeNull();
    expect(
      expandAxesIfRectangularGrid(new Float32Array([1]), new Float32Array(), []),
    ).toBeNull();
  });

  it('picks the first matching variable and ignores 3-D ones', () => {
    const lat = new Float32Array([1, 2]);
    const lon = new Float32Array([10, 20, 30]);
    const variables = [
      makeVar('levels', [5, 2, 3]),
      makeVar('2t', [2, 3]),
    ];
    const result = expandAxesIfRectangularGrid(lat, lon, variables);
    expect(result).not.toBeNull();
    expect(result!.lat.length).toBe(6);
  });
});

import { describe, it, expect } from 'vitest';
import { findNearestPointIndex, buildLevelSliceIndex } from '../usePointInspection';

describe('findNearestPointIndex', () => {
  it('returns 0 when there is only one point', () => {
    const lat = new Float32Array([10]);
    const lon = new Float32Array([20]);
    expect(findNearestPointIndex(lat, lon, 10, 20)).toBe(0);
  });

  it('finds the nearest of several points', () => {
    // Points at (0,0), (10,0), (0,10)
    const lat = new Float32Array([0, 10, 0]);
    const lon = new Float32Array([0,  0, 10]);
    // Click near (9, 1) — closest to (10, 0)
    expect(findNearestPointIndex(lat, lon, 9, 1)).toBe(1);
  });

  it('returns exact match index when click is on a grid point', () => {
    const lat = new Float32Array([0, 5, 10]);
    const lon = new Float32Array([0, 5, 10]);
    expect(findNearestPointIndex(lat, lon, 5, 5)).toBe(1);
  });

  it('handles ties by returning the first encountered index', () => {
    const lat = new Float32Array([0, 0]);
    const lon = new Float32Array([-1, 1]);
    // Click at lon=0: equidistant, expect first (index 0)
    expect(findNearestPointIndex(lat, lon, 0, 0)).toBe(0);
  });

  it('matches a [0, 360] grid point when click lon is negative', () => {
    // Grid stored in [0, 360]: 270° == -90°
    const lat = new Float32Array([0, 0, 0]);
    const lon = new Float32Array([90, 180, 270]);
    // Click at lon=-90 (equiv to 270°), lat=0
    expect(findNearestPointIndex(lat, lon, 0, -90)).toBe(2);
  });
});

describe('buildLevelSliceIndex', () => {
  it('returns 0 when selectedLevel is null', () => {
    expect(buildLevelSliceIndex(null, undefined)).toBe(0);
  });

  it('returns 0 when no anemoi levels metadata', () => {
    expect(buildLevelSliceIndex(500, undefined)).toBe(0);
  });

  it('finds the index of the selected level in anemoi.levels', () => {
    const levels = [1000, 850, 500, 250];
    expect(buildLevelSliceIndex(500, levels)).toBe(2);
  });

  it('returns 0 when selected level is not in the list', () => {
    const levels = [1000, 850, 500];
    expect(buildLevelSliceIndex(700, levels)).toBe(0);
  });
});

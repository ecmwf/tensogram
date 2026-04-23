// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Regression tests for `decideSliceDim`, the helper that picks the
 * axis of a field to slice down to 2-D before rendering.
 *
 * Covers the cases flagged in Oracle's post-implementation review:
 *
 *  1. Truly 2-D gridded data with meshed coords — no slice
 *     (the original "donut in the north" regression).
 *  2. Packed vertical levels with meshed coords
 *     `[N_lev, nLat, nLon]` with `coordLength = nLat*nLon` — slice
 *     dim 0 (the regression Oracle flagged on the first guard).
 *  3. Packed levels with a flattened spatial axis
 *     `[N_lev, nSpatial]` with `coordLength = nSpatial` — the
 *     classic case the original guard was designed for.
 *  4. Shape + coord length that don't fit — no slice, best-effort.
 */

import { describe, expect, it } from 'vitest';
import { decideSliceDim } from '../useAppStore';

describe('decideSliceDim', () => {
  it('does not slice a 2-D gridded field that matches the full coord length', () => {
    // shape [721, 1440], coord length = 721 * 1440 = 1_038_240 (meshed).
    expect(decideSliceDim([721, 1440], 721 * 1440)).toBe(-1);
  });

  it('slices dim 0 for a packed [N_lev, nLat, nLon] field with meshed coords', () => {
    // This is the Oracle-flagged regression: before the generalisation
    // to the total-is-a-multiple rule, this case returned -1 and the
    // renderer got the full 3-D tensor.
    expect(decideSliceDim([37, 721, 1440], 721 * 1440)).toBe(0);
  });

  it('slices dim 0 for a packed [N_lev, nSpatial] field with 1-D coords', () => {
    // Classic case the original Tensoscope guard was designed for.
    expect(decideSliceDim([37, 1038240], 1038240)).toBe(0);
  });

  it('does not slice when coord length is zero', () => {
    expect(decideSliceDim([721, 1440], 0)).toBe(-1);
  });

  it('does not slice 1-D data', () => {
    // coord == total → no slice regardless of rank.
    expect(decideSliceDim([1038240], 1038240)).toBe(-1);
  });

  it('returns -1 on an empty shape', () => {
    expect(decideSliceDim([], 1000)).toBe(-1);
  });

  it('does not slice when total is less than coord length', () => {
    // Degenerate: coords advertise a bigger grid than the data covers.
    expect(decideSliceDim([100], 1000)).toBe(-1);
  });

  it('does not slice when total is not a multiple of coord length', () => {
    // total = 13*1000 + 500, not divisible.  Shape and coords don't
    // fit; decode best-effort and let the renderer fail loudly rather
    // than silently picking a wrong slice.
    expect(decideSliceDim([13500], 1000)).toBe(-1);
  });

  it('slices on large multiples (hypothetical 4-D ensemble)', () => {
    // [nEnsemble, nLev, nLat, nLon] with meshed coords.  Picks dim 0
    // (ensemble), which is consistent with "slice the outermost
    // non-spatial dim" even for rank ≥ 4 — a latent feature, not
    // wired into the UI today, but guaranteed not to crash.
    expect(decideSliceDim([5, 37, 721, 1440], 721 * 1440)).toBe(0);
  });
});

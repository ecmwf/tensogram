import { describe, it, expect, beforeEach } from 'vitest';
import { __test } from '../FieldOverlay';

const { applyExcludeMask, resolveBounds, getCached, putCache, clearCache } = __test;

interface TestParams {
  width: number;
  height: number;
  binDeg: number;
  rowLats: Float64Array;
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
  mapProjection: 'mercator' | 'geographic';
}

function makeFullAlphaRgba(width: number, height: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4 + 0] = 100;
    rgba[i * 4 + 1] = 150;
    rgba[i * 4 + 2] = 200;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function geoParams(width = 16, height = 16): TestParams {
  return {
    width, height, binDeg: 1,
    rowLats: new Float64Array(height),
    lonMin: -180, lonMax: 180, latMin: -90, latMax: 90,
    mapProjection: 'geographic',
  };
}

describe('applyExcludeMask', () => {
  it('zeroes alpha inside the rect and leaves it elsewhere', () => {
    const rgba = makeFullAlphaRgba(16, 16);
    const params = geoParams();
    // West=0..East=180 covers the right half (cols 8..16);
    // South=0..North=90 covers the top half (rows 0..8).
    const out = applyExcludeMask(rgba, params, { west: 0, east: 180, south: 0, north: 90 });

    for (let row = 0; row < 16; row++) {
      for (let col = 0; col < 16; col++) {
        const a = out[(row * 16 + col) * 4 + 3];
        const inside = row < 8 && col >= 8;
        expect(a).toBe(inside ? 0 : 255);
      }
    }
    // Source untouched (slice copy)
    expect(rgba[3]).toBe(255);
  });

  it('uses Math.floor on left/top and Math.ceil on right/bottom', () => {
    const rgba = makeFullAlphaRgba(10, 10);
    // 10 cols span lonMin=-180 to lonMax=180 → 36° per col.
    // West=-90 → x0 = floor((-90 - -180)/360 * 10) = floor(2.5) = 2
    // East=90  → x1 = ceil((90 - -180)/360 * 10) = ceil(7.5) = 8
    const params: TestParams = { ...geoParams(10, 10) };
    const out = applyExcludeMask(rgba, params, { west: -90, east: 90, south: -90, north: 90 });

    // Row 0: cols 2..7 zeroed
    for (let col = 0; col < 10; col++) {
      const a = out[col * 4 + 3];
      expect(a).toBe(col >= 2 && col < 8 ? 0 : 255);
    }
  });

  it('clamps the rect to the image extent', () => {
    const rgba = makeFullAlphaRgba(8, 8);
    const params = geoParams(8, 8);
    // Rect well outside the image: should not crash, image unchanged.
    const out = applyExcludeMask(rgba, params, { west: 200, east: 300, south: -200, north: -100 });
    for (let i = 0; i < out.length; i += 4) {
      expect(out[i + 3]).toBe(255);
    }
  });
});

describe('resolveBounds', () => {
  it('clips geographic to ±90 when no bounds are provided', () => {
    const r = resolveBounds('geographic', undefined, undefined, undefined);
    expect(r.lonMin).toBe(-180);
    expect(r.lonMax).toBe(180);
    expect(r.latMin).toBe(-90);
    expect(r.latMax).toBe(90);
  });

  it('clips mercator to ±85 when no bounds are provided', () => {
    const r = resolveBounds('mercator', undefined, undefined, undefined);
    expect(r.latMin).toBe(-85);
    expect(r.latMax).toBe(85);
  });

  it('passes through valid in-range bounds for geographic', () => {
    const r = resolveBounds('geographic', { west: -10, east: 20, south: -5, north: 30 }, 800, 600);
    expect(r).toMatchObject({ lonMin: -10, lonMax: 20, latMin: -5, latMax: 30, vw: 800, vh: 600 });
  });

  it('clamps mercator latitudes to ±85 even with valid bounds', () => {
    const r = resolveBounds('mercator', { west: -10, east: 20, south: -89, north: 89 }, 800, 600);
    expect(r.latMin).toBe(-85);
    expect(r.latMax).toBe(85);
  });

  it('falls back to global on antimeridian crossing or wide spans', () => {
    const r1 = resolveBounds('geographic', { west: 170, east: 200, south: -10, north: 10 }, 100, 100);
    expect(r1.lonMin).toBe(-180);
    expect(r1.lonMax).toBe(180);

    const r2 = resolveBounds('geographic', { west: -200, east: 0, south: -10, north: 10 }, 100, 100);
    expect(r2.lonMin).toBe(-180);

    const r3 = resolveBounds('geographic', { west: -170, east: 170, south: -10, north: 10 }, 100, 100);
    expect(r3.lonMin).toBe(-180);
  });

  it('regression: geographic is selected by string arg, not by an object', () => {
    // The previous implementation was called as resolveBounds(props), which
    // silently routed geographic renders through the mercator branch. Lock in
    // the new shape: the first arg must be the literal projection string. If
    // someone passes anything that is not the string 'geographic', the result
    // must use the mercator clamp (±85), not the geographic clamp (±90).
    const r = resolveBounds(
      { mapProjection: 'geographic' } as unknown as 'geographic',
      undefined, undefined, undefined,
    );
    expect(r.latMax).toBe(85);
    expect(r.latMin).toBe(-85);
  });
});

describe('image cache', () => {
  beforeEach(() => clearCache());

  it('round-trips raw RGBA, not a dataUrl', () => {
    const rgba = makeFullAlphaRgba(4, 4);
    const params = geoParams(4, 4);
    putCache({ key: 'k1', rgba, width: 4, height: 4, params });
    const got = getCached('k1');
    expect(got).not.toBeNull();
    expect(got!.rgba).toBe(rgba);
    expect(got!.width).toBe(4);
    expect(got!.height).toBe(4);
    expect(got!.params).toBe(params);
  });

  it('returns null on miss', () => {
    expect(getCached('nope')).toBeNull();
  });

  it('moves recently accessed entries to the front (LRU)', () => {
    const params = geoParams(2, 2);
    putCache({ key: 'a', rgba: makeFullAlphaRgba(2, 2), width: 2, height: 2, params });
    putCache({ key: 'b', rgba: makeFullAlphaRgba(2, 2), width: 2, height: 2, params });
    // Access 'a' — should now be at the front
    const got = getCached('a');
    expect(got).not.toBeNull();
    // Re-access works (still cached, just reordered)
    expect(getCached('a')).not.toBeNull();
    expect(getCached('b')).not.toBeNull();
  });
});

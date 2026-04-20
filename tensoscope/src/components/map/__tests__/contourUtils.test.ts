// tensoscope/src/components/map/__tests__/contourUtils.test.ts

import { describe, it, expect } from 'vitest';
import { getNumBands, contourLutIndex, DEFAULT_NUM_BANDS } from '../contourUtils';

describe('getNumBands', () => {
  it('returns DEFAULT_NUM_BANDS for named palettes', () => {
    expect(getNumBands('viridis')).toBe(DEFAULT_NUM_BANDS);
    expect(getNumBands('plasma')).toBe(DEFAULT_NUM_BANDS);
  });

  it('returns stop count for custom palette with stops', () => {
    const stops = [
      { pos: 0, color: '#000' },
      { pos: 0.5, color: '#888' },
      { pos: 1, color: '#fff' },
    ];
    expect(getNumBands('custom', stops)).toBe(3);
  });

  it('returns DEFAULT_NUM_BANDS for custom palette with fewer than 2 stops', () => {
    expect(getNumBands('custom', [])).toBe(DEFAULT_NUM_BANDS);
    expect(getNumBands('custom', [{ pos: 0, color: '#000' }])).toBe(DEFAULT_NUM_BANDS);
    expect(getNumBands('custom', undefined)).toBe(DEFAULT_NUM_BANDS);
  });
});

describe('contourLutIndex', () => {
  it('returns 0 for numBands <= 1', () => {
    expect(contourLutIndex(0.5, 1)).toBe(0);
    expect(contourLutIndex(0.99, 0)).toBe(0);
  });

  it('maps values to correct band with 2 bands', () => {
    expect(contourLutIndex(0.0, 2)).toBe(0);    // band 0 → lut 0
    expect(contourLutIndex(0.49, 2)).toBe(0);   // band 0 → lut 0
    expect(contourLutIndex(0.5, 2)).toBe(255);  // band 1 → lut 255
    expect(contourLutIndex(1.0, 2)).toBe(255);  // band 1 → lut 255
  });

  it('maps values to correct band with 10 bands', () => {
    expect(contourLutIndex(0.05, 10)).toBe(0);
    expect(contourLutIndex(0.95, 10)).toBe(255);
    // band 4: t=0.45 → floor(0.45*10)=4 → round(4*255/9) = round(113.33) = 113
    expect(contourLutIndex(0.45, 10)).toBe(113);
  });

  it('handles boundary values t=0 and t=1', () => {
    expect(contourLutIndex(0, 10)).toBe(0);
    expect(contourLutIndex(1, 10)).toBe(255);
  });
});

/**
 * Colour map utilities for mapping float data to RGBA pixels using
 * d3-scale-chromatic palettes.
 */

import {
  interpolateViridis,
  interpolatePlasma,
  interpolateInferno,
  interpolateMagma,
  interpolateTurbo,
  interpolateRdBu,
  interpolateSpectral,
  interpolateBlues,
  interpolateGreens,
  interpolateGreys,
  interpolateYlOrRd,
  interpolateRdYlBu,
} from 'd3-scale-chromatic';

export const PALETTE_NAMES: string[] = [
  'viridis',
  'plasma',
  'inferno',
  'magma',
  'turbo',
  'spectral',
  'coolwarm',
  'rdylbu',
  'ylOrRd',
  'blues',
  'greens',
  'greys',
];

export interface CustomStop {
  pos: number;    // 0–1
  color: string;  // CSS hex, e.g. '#ff0000'
}

export interface PaletteOptions {
  reversed?: boolean;
  customStops?: CustomStop[];
}

type Interpolator = (t: number) => string;

const INTERPOLATORS: Record<string, Interpolator> = {
  viridis: interpolateViridis,
  plasma: interpolatePlasma,
  inferno: interpolateInferno,
  magma: interpolateMagma,
  turbo: interpolateTurbo,
  // spectral: reversed so cold=blue, warm=red (matches Spectral_r)
  spectral: (t: number) => interpolateSpectral(1 - t),
  // coolwarm: RdBu reversed so blue=cold, red=warm
  coolwarm: (t: number) => interpolateRdBu(1 - t),
  rdylbu: (t: number) => interpolateRdYlBu(1 - t),
  ylOrRd: interpolateYlOrRd,
  blues: interpolateBlues,
  greens: interpolateGreens,
  greys: interpolateGreys,
};

/** Parse a CSS colour string into [r, g, b]. */
function parseRgb(css: string): [number, number, number] {
  if (css[0] === '#') {
    const hex = css.length === 4
      ? css[1] + css[1] + css[2] + css[2] + css[3] + css[3]
      : css.slice(1);
    return [
      parseInt(hex.slice(0, 2), 16),
      parseInt(hex.slice(2, 4), 16),
      parseInt(hex.slice(4, 6), 16),
    ];
  }
  const m = css.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (m) return [parseInt(m[1], 10), parseInt(m[2], 10), parseInt(m[3], 10)];
  return [0, 0, 0];
}

// ── LUT cache ────────────────────────────────────────────────────────────

export const LUT_SIZE = 256;
const lutCache = new Map<string, Uint8Array>();

function buildNamedLUT(palette: string): Uint8Array {
  const interpolate = INTERPOLATORS[palette] ?? INTERPOLATORS['viridis'];
  const lut = new Uint8Array(LUT_SIZE * 3);
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = parseRgb(interpolate(i / (LUT_SIZE - 1)));
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}

function buildCustomLUT(stops: CustomStop[]): Uint8Array {
  const sorted = [...stops].sort((a, b) => a.pos - b.pos);
  const lut = new Uint8Array(LUT_SIZE * 3);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    let lo = sorted[0];
    let hi = sorted[sorted.length - 1];
    for (let j = 0; j < sorted.length - 1; j++) {
      if (sorted[j].pos <= t && sorted[j + 1].pos >= t) {
        lo = sorted[j];
        hi = sorted[j + 1];
        break;
      }
    }
    const span = hi.pos - lo.pos;
    const f = span === 0 ? 0 : Math.min(1, Math.max(0, (t - lo.pos) / span));
    const [lr, lg, lb] = parseRgb(lo.color);
    const [hr, hg, hb] = parseRgb(hi.color);
    lut[i * 3] = Math.round(lr + (hr - lr) * f);
    lut[i * 3 + 1] = Math.round(lg + (hg - lg) * f);
    lut[i * 3 + 2] = Math.round(lb + (hb - lb) * f);
  }
  return lut;
}

function reverseLUT(src: Uint8Array): Uint8Array {
  const rev = new Uint8Array(LUT_SIZE * 3);
  for (let i = 0; i < LUT_SIZE; i++) {
    const j = LUT_SIZE - 1 - i;
    rev[i * 3] = src[j * 3];
    rev[i * 3 + 1] = src[j * 3 + 1];
    rev[i * 3 + 2] = src[j * 3 + 2];
  }
  return rev;
}

/**
 * Build (or retrieve from cache) a 256-entry RGB LUT.
 * Returns a Uint8Array of length 256 * 3 (R, G, B interleaved).
 */
export function getPaletteLUT(palette: string, options?: PaletteOptions): Uint8Array {
  const { reversed = false, customStops } = options ?? {};

  const baseKey = palette === 'custom'
    ? 'custom:' + JSON.stringify([...(customStops ?? [])].sort((a, b) => a.pos - b.pos))
    : palette;
  const key = reversed ? baseKey + ':r' : baseKey;

  let lut = lutCache.get(key);
  if (lut) return lut;

  let base = lutCache.get(baseKey);
  if (!base) {
    base = palette === 'custom' && customStops && customStops.length >= 2
      ? buildCustomLUT(customStops)
      : buildNamedLUT(palette);
    lutCache.set(baseKey, base);
  }

  lut = reversed ? reverseLUT(base) : base;
  if (reversed) lutCache.set(key, lut);
  return lut;
}

const CSS_SAMPLE_POINTS = 8;

/**
 * Returns a CSS `linear-gradient(to right, ...)` string for a palette.
 * Used to render gradient strips in the UI.
 */
export function getPaletteCSS(palette: string, options?: PaletteOptions): string {
  const { reversed = false, customStops } = options ?? {};

  if (palette === 'custom' && customStops && customStops.length >= 2) {
    let sorted = [...customStops].sort((a, b) => a.pos - b.pos);
    if (reversed) {
      sorted = sorted.map(s => ({ pos: 1 - s.pos, color: s.color })).sort((a, b) => a.pos - b.pos);
    }
    const parts = sorted.map(s => `${s.color} ${(s.pos * 100).toFixed(1)}%`).join(', ');
    return `linear-gradient(to right, ${parts})`;
  }

  const lut = getPaletteLUT(palette, options);
  const parts: string[] = [];
  for (let i = 0; i < CSS_SAMPLE_POINTS; i++) {
    const idx = Math.round((i / (CSS_SAMPLE_POINTS - 1)) * (LUT_SIZE - 1));
    parts.push(`rgb(${lut[idx * 3]},${lut[idx * 3 + 1]},${lut[idx * 3 + 2]})`);
  }
  return `linear-gradient(to right, ${parts.join(', ')})`;
}

/**
 * Maps float values in `data` to RGBA pixels using a pre-computed LUT.
 *
 * NaN values are mapped to fully transparent pixels (alpha = 0).
 * Returns a Uint8ClampedArray of length data.length * 4.
 */
export function applyColormap(
  data: Float32Array,
  min: number,
  max: number,
  palette: string,
  options?: PaletteOptions,
): Uint8ClampedArray {
  const lut = getPaletteLUT(palette, options);
  const range = max - min;
  const invRange = range === 0 ? 0 : (LUT_SIZE - 1) / range;
  const rgba = new Uint8ClampedArray(data.length * 4);

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    const base = i * 4;

    if (v !== v) continue; // NaN → transparent

    const idx = Math.max(0, Math.min(LUT_SIZE - 1, ((v - min) * invRange) | 0));
    const lutBase = idx * 3;
    rgba[base] = lut[lutBase];
    rgba[base + 1] = lut[lutBase + 1];
    rgba[base + 2] = lut[lutBase + 2];
    rgba[base + 3] = 255;
  }

  return rgba;
}

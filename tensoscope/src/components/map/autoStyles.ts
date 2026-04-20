/**
 * Per-variable default visualisation styles, derived from
 * earthkit-plots auto-styles (v1.0.0rc3).
 *
 * Maps MARS param short names to a default palette, colour range, and
 * units string. The palette names must exist in colormaps.ts.
 */

export interface AutoStyle {
  palette: string;
  min: number;
  max: number;
  units: string;
}

const STYLES: Record<string, AutoStyle> = {
  // ── Temperature ────────────────────────────────────────────────────
  '2t':   { palette: 'spectral', min: 230, max: 310, units: 'K' },
  't2m':  { palette: 'spectral', min: 230, max: 310, units: 'K' },
  'mn2t': { palette: 'spectral', min: 225, max: 310, units: 'K' },
  'mx2t': { palette: 'spectral', min: 230, max: 320, units: 'K' },
  'sst':  { palette: 'spectral', min: 270, max: 305, units: 'K' },
  '2d':   { palette: 'spectral', min: 220, max: 310, units: 'K' },

  // ── Pressure ───────────────────────────────────────────────────────
  'msl':  { palette: 'coolwarm', min: 97000, max: 105000, units: 'Pa' },
  'sp':   { palette: 'coolwarm', min: 50000, max: 105000, units: 'Pa' },

  // ── Wind speed ─────────────────────────────────────────────────────
  '10si': { palette: 'viridis', min: 0, max: 30, units: 'm/s' },
  'ws':   { palette: 'viridis', min: 0, max: 50, units: 'm/s' },

  // ── Convection ─────────────────────────────────────────────────────
  'cape': { palette: 'inferno', min: 0, max: 4500, units: 'J/kg' },
  'cin':  { palette: 'plasma',  min: -500, max: 0, units: 'J/kg' },

  // ── Precipitation / water ──────────────────────────────────────────
  // earthkit-plots optimal style for total precipitation uses turbo
  'tp':   { palette: 'turbo', min: 0, max: 0.05, units: 'm' },
  'cp':   { palette: 'turbo', min: 0, max: 0.03, units: 'm' },
  'sf':   { palette: 'blues', min: 0, max: 0.05, units: 'm' },

  // ── Cloud cover ────────────────────────────────────────────────────
  'tcc':  { palette: 'greys', min: 0, max: 1, units: '' },
  'lcc':  { palette: 'greys', min: 0, max: 1, units: '' },
  'mcc':  { palette: 'greys', min: 0, max: 1, units: '' },
  'hcc':  { palette: 'greys', min: 0, max: 1, units: '' },

  // ── Surface fields ─────────────────────────────────────────────────
  'lsm':  { palette: 'greens', min: 0, max: 1, units: '' },
  'z':    { palette: 'turbo',  min: -500, max: 60000, units: 'm2/s2' },

  // ── Sea ice ────────────────────────────────────────────────────────
  // earthkit-plots uses solid grey for sea ice cover
  'ci':     { palette: 'greys', min: 0, max: 1, units: '' },
  'siconc': { palette: 'greys', min: 0, max: 1, units: '' },
};

/**
 * Look up the default style for a MARS param.
 * Returns undefined if no style is defined for this param.
 */
export function getAutoStyle(param: string | undefined): AutoStyle | undefined {
  if (!param) return undefined;
  return STYLES[param.toLowerCase()] ?? STYLES[param];
}

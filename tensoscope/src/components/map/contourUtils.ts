// tensoscope/src/components/map/contourUtils.ts

export const DEFAULT_NUM_BANDS = 10;

/**
 * Number of contour bands for the given palette.
 * Custom palettes use their stop count; all others default to DEFAULT_NUM_BANDS.
 */
export function getNumBands(
  palette: string,
  customStops?: { pos: number; color: string }[],
): number {
  if (palette === 'custom' && customStops && customStops.length >= 2) {
    return customStops.length;
  }
  return DEFAULT_NUM_BANDS;
}

/**
 * Maps a normalised value t ∈ [0, 1] to a LUT index ∈ [0, 255]
 * for the given number of discrete contour bands.
 *
 * Band boundaries are at t = k/numBands for k = 1…numBands-1.
 * The LUT index is sampled evenly so band 0 → index 0 and
 * band numBands-1 → index 255.
 */
export function contourLutIndex(t: number, numBands: number): number {
  if (numBands <= 1) return 0;
  const bandIdx = Math.min(numBands - 1, Math.floor(t * numBands));
  return Math.round((bandIdx * 255) / (numBands - 1));
}

import { useState, useEffect } from 'react';
import type { CoordinateData, DecodedField, FileIndex, Tensoscope } from '../../tensogram';
import type { AnimationFrame } from '../animation/useAnimationSequence';
import { decideSliceDim } from '../../store/useAppStore';

// ── Types ────────────────────────────────────────────────────────────────────

export interface TimeSeriesEntry {
  step: number;
  label: string;
  value: number;
}

export interface InspectionData {
  pointLat: number;
  pointLon: number;
  /** Non-empty when animation frames exist. */
  entries: TimeSeriesEntry[];
  /** Set when no time series (entries will be empty). NaN if lookup failed. */
  singleValue: number | null;
  loading: boolean;
}

export interface ClickPoint {
  lat: number;
  lon: number;
  screenX: number;
  screenY: number;
}

// ── Nearest point ────────────────────────────────────────────────────────────

/**
 * Returns the index of the closest point in `lat`/`lon` to the click
 * coordinates using squared distance in degrees. Longitude distance is
 * computed as the minimum of the direct and wrap-around paths so grids
 * stored in [0, 360] match clicks returned in [-180, 180].
 * Exported for testing.
 */
export function findNearestPointIndex(
  lat: Float32Array,
  lon: Float32Array,
  clickLat: number,
  clickLon: number,
): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < lat.length; i++) {
    const dlat = lat[i] - clickLat;
    const dlon = ((lon[i] - clickLon + 540) % 360) - 180;
    const dist = dlat * dlat + dlon * dlon;
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

// ── Level slice index ────────────────────────────────────────────────────────

/**
 * Resolves which slice index along the level dimension corresponds to
 * `selectedLevel`, using `anemoi.levels` metadata when available.
 * Exported for testing.
 */
export function buildLevelSliceIndex(
  selectedLevel: number | null,
  anemoiLevels: number[] | undefined,
): number {
  if (selectedLevel == null || !anemoiLevels) return 0;
  const idx = anemoiLevels.indexOf(selectedLevel);
  return idx >= 0 ? idx : 0;
}

// ── Time series fetch ────────────────────────────────────────────────────────

/**
 * Decodes each animation frame at `pointIndex` and returns the value series.
 * Frames are decoded sequentially to respect remote concurrency limits.
 */
export async function fetchTimeSeries(
  viewer: Tensoscope,
  fileIndex: FileIndex,
  coordinates: CoordinateData,
  frames: AnimationFrame[],
  pointIndex: number,
  selectedLevel: number | null,
): Promise<TimeSeriesEntry[]> {
  const results: TimeSeriesEntry[] = [];
  const coordLength = coordinates.lat.length;

  for (const frame of frames) {
    const varInfo = fileIndex.variables.find(
      (v) => v.msgIndex === frame.msgIdx && v.objIndex === frame.objIdx,
    );
    const shape = varInfo?.shape ?? [];
    const sliceDim = decideSliceDim(shape, coordLength);

    let field: DecodedField;
    if (sliceDim >= 0) {
      const anemoi = varInfo?.metadata?.anemoi as Record<string, unknown> | undefined;
      const anemoiLevels = anemoi?.levels as number[] | undefined;
      const sliceIdx = buildLevelSliceIndex(selectedLevel, anemoiLevels);
      field = await viewer.decodeFieldSlice(frame.msgIdx, frame.objIdx, sliceDim, sliceIdx);
    } else {
      field = await viewer.decodeField(frame.msgIdx, frame.objIdx);
    }

    const value = field.data[pointIndex];
    if (!Number.isNaN(value)) {
      results.push({ step: frame.step, label: frame.label, value });
    }
  }
  return results;
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export interface UsePointInspectionParams {
  point: ClickPoint | null;
  coordinates: CoordinateData | null;
  fieldData: Float32Array | null;
  viewer: Tensoscope | null;
  fileIndex: FileIndex | null;
  frames: AnimationFrame[];
  selectedLevel: number | null;
}

export function usePointInspection(params: UsePointInspectionParams): InspectionData | null {
  const { point, coordinates, fieldData, viewer, fileIndex, frames, selectedLevel } = params;
  const [result, setResult] = useState<InspectionData | null>(null);

  useEffect(() => {
    if (!point || !coordinates || coordinates.lat.length === 0) {
      setResult(null);
      return;
    }

    const pointIndex = findNearestPointIndex(
      coordinates.lat,
      coordinates.lon,
      point.lat,
      point.lon,
    );
    const pointLat = coordinates.lat[pointIndex];
    const rawLon = coordinates.lon[pointIndex];
    const pointLon = rawLon > 180 ? rawLon - 360 : rawLon < -180 ? rawLon + 360 : rawLon;

    if (frames.length <= 1) {
      const raw = fieldData ? fieldData[pointIndex] : NaN;
      const singleValue = Number.isNaN(raw) ? null : raw;
      setResult({ pointLat, pointLon, entries: [], singleValue, loading: false });
      return;
    }

    if (!viewer || !fileIndex) return;
    setResult({ pointLat, pointLon, entries: [], singleValue: null, loading: true });

    fetchTimeSeries(viewer, fileIndex, coordinates, frames, pointIndex, selectedLevel)
      .then((entries) => setResult({ pointLat, pointLon, entries, singleValue: null, loading: false }))
      .catch(() => setResult({ pointLat, pointLon, entries: [], singleValue: null, loading: false }));
  }, [point, coordinates, fieldData, viewer, fileIndex, frames, selectedLevel]);

  return result;
}

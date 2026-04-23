/** Global application state managed via Zustand. */

import { create } from 'zustand';
import { Tensoscope } from '../tensogram';
import type { FileIndex, DecodedField, CoordinateData } from '../tensogram';
import { getAutoStyle } from '../components/map/autoStyles';
import type { CustomStop } from '../components/map/colormaps';

/**
 * Decide which axis of a field to slice down to 2-D before rendering.
 *
 * `coordLength` is the length of the per-point (meshgridded) lat array
 * Tensoscope caches; it equals the flattened spatial grid size.
 *
 * Three cases, in order:
 *
 *  - `total === coordLength` — the field IS the spatial grid (e.g.
 *    `[nLat, nLon]` meshed, or `[nSpatial]` already flattened).  No
 *    slicing; the renderer consumes the full tensor.
 *  - `total` is a positive integer multiple of `coordLength` — one
 *    or more leading "level-like" dims stack the spatial grid.  Slice
 *    dim 0 (matches both `[N_lev, nLat, nLon]` and `[N_lev, nSpatial]`).
 *  - Otherwise — shape and coords do not fit together.  Decode the
 *    full tensor as best-effort; rendering will fail further
 *    downstream if the mismatch is real.
 *
 * Exported for unit testing.  @internal
 */
export function decideSliceDim(shape: readonly number[], coordLength: number): number {
  if (coordLength <= 0 || shape.length === 0) return -1;
  let total = 1;
  for (const s of shape) total *= s;
  if (total === coordLength) return -1;
  if (total > coordLength && total % coordLength === 0) return 0;
  return -1;
}

export interface FieldStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface ColorScaleConfig {
  min: number;
  max: number;
  palette: string;
  logScale: boolean;
  paletteReversed: boolean;
  customStops: CustomStop[];
  nativeUnits: string;
  displayUnit: string;
}

interface AppState {
  viewer: Tensoscope | null;
  fileIndex: FileIndex | null;
  selectedObject: { msgIdx: number; objIdx: number } | null;
  selectedLevel: number | null;
  fieldData: Float32Array | null;
  fieldShape: number[];
  fieldStats: FieldStats | null;
  coordinates: CoordinateData | null;
  colorScale: ColorScaleConfig;
  loading: boolean;
  error: string | null;

  openLocalFile: (file: File) => Promise<void>;
  openUrl: (url: string) => Promise<void>;
  selectField: (msgIdx: number, objIdx: number) => Promise<void>;
  setSelectedLevel: (level: number | null) => void;
  fetchSlice: (dim: number, idx: number) => Promise<void>;
  setColorScale: (config: Partial<ColorScaleConfig>) => void;
}

const DEFAULT_COLOR_SCALE: ColorScaleConfig = {
  min: 0,
  max: 1,
  palette: 'viridis',
  logScale: false,
  paletteReversed: false,
  customStops: [{ pos: 0, color: '#000000' }, { pos: 1, color: '#ffffff' }],
  nativeUnits: '',
  displayUnit: '',
};

/** Shared logic after a Tensoscope is created. */
async function initViewer(
  viewer: Tensoscope,
  set: (partial: Partial<AppState>) => void,
) {
  const index = await viewer.buildIndex();
  const coords = await viewer.fetchCoordinates(0);
  set({
    viewer,
    fileIndex: index,
    coordinates: coords,
    selectedObject: null,
    selectedLevel: null,
    fieldData: null,
    fieldShape: [],
    fieldStats: null,
    loading: false,
    error: null,
  });
}

export const useAppStore = create<AppState>((set, get) => ({
  viewer: null,
  fileIndex: null,
  selectedObject: null,
  selectedLevel: null,
  fieldData: null,
  fieldShape: [],
  fieldStats: null,
  coordinates: null,
  colorScale: DEFAULT_COLOR_SCALE,
  loading: false,
  error: null,

  openLocalFile: async (file: File) => {
    set({ loading: true, error: null });
    try {
      const prev = get().viewer;
      if (prev) prev.close();
      const viewer = await Tensoscope.fromFile(file);
      await initViewer(viewer, set);
      const first = get().fileIndex?.variables[0];
      if (first) await get().selectField(first.msgIndex, first.objIndex);
    } catch (err) {
      set({ loading: false, error: String(err) });
    }
  },

  openUrl: async (url: string) => {
    set({ loading: true, error: null });
    try {
      const prev = get().viewer;
      if (prev) prev.close();
      const viewer = await Tensoscope.fromUrl(url);
      await initViewer(viewer, set);
      const first = get().fileIndex?.variables[0];
      if (first) await get().selectField(first.msgIndex, first.objIndex);
    } catch (err) {
      set({ loading: false, error: String(err) });
    }
  },

  selectField: async (msgIdx: number, objIdx: number) => {
    const { viewer, fileIndex, coordinates, selectedLevel } = get();
    if (!viewer || !fileIndex) return;
    set({ loading: true, error: null, selectedObject: { msgIdx, objIdx } });
    // Yield so React renders the loading state before the synchronous WASM decode blocks the thread.
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    try {
      const varInfo = fileIndex.variables.find(
        (v) => v.msgIndex === msgIdx && v.objIndex === objIdx,
      );
      const originalShape = varInfo?.shape ?? [];
      const coordLength = coordinates?.lat.length ?? 0;

      const sliceDim = decideSliceDim(originalShape, coordLength);

      let result: DecodedField;
      if (sliceDim >= 0) {
        let sliceIdx = 0;
        if (selectedLevel != null) {
          const anemoi = varInfo?.metadata?.anemoi as Record<string, unknown> | undefined;
          const levels = anemoi?.levels as number[] | undefined;
          if (levels) {
            const idx = levels.indexOf(selectedLevel);
            if (idx >= 0) sliceIdx = idx;
          }
        }
        result = await viewer.decodeFieldSlice(msgIdx, objIdx, sliceDim, sliceIdx);
      } else {
        result = await viewer.decodeField(msgIdx, objIdx);
      }

      // Apply per-variable auto-style if available, otherwise use data range.
      // mars.param can be either a short string ("2t", "t", ...) or a
      // GRIB integer code (167, 130, ...); getAutoStyle handles both.
      const mars = varInfo?.metadata?.mars as Record<string, unknown> | undefined;
      const marsParam = mars?.param as string | number | undefined;
      const param = marsParam ?? varInfo?.name;
      const style = getAutoStyle(param);

      set({
        fieldData: result.data,
        fieldShape: originalShape,
        fieldStats: result.stats,
        colorScale: style
          ? { ...DEFAULT_COLOR_SCALE, palette: style.palette, min: style.min, max: style.max, logScale: false, nativeUnits: style.units, displayUnit: style.units }
          : { ...get().colorScale, min: result.stats.min, max: result.stats.max, nativeUnits: '', displayUnit: '' },
        loading: false,
      });
    } catch (err) {
      set({ loading: false, error: String(err) });
    }
  },

  setSelectedLevel: (level: number | null) => {
    set({ selectedLevel: level });
  },

  fetchSlice: async (dim: number, idx: number) => {
    const { viewer, selectedObject } = get();
    if (!viewer || !selectedObject) return;
    set({ loading: true, error: null });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    try {
      const result = await viewer.decodeFieldSlice(
        selectedObject.msgIdx,
        selectedObject.objIdx,
        dim,
        idx,
      );
      set({
        fieldData: result.data,
        // Keep original fieldShape so DimensionBrowser stays visible
        fieldStats: result.stats,
        loading: false,
      });
    } catch (err) {
      set({ loading: false, error: String(err) });
    }
  },

  setColorScale: (config: Partial<ColorScaleConfig>) => {
    set((state) => ({ colorScale: { ...state.colorScale, ...config } }));
  },
}));

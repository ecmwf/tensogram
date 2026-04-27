/** Global application state managed via Zustand. */

import { create } from 'zustand';
import { Tensoscope } from '../tensogram';
import type { FileIndex, DecodedField, CoordinateData } from '../tensogram';
import { getAutoStyle } from '../components/map/autoStyles';
import type { CustomStop } from '../components/map/colormaps';
import { getFrameCache, initFrameCache } from '../tensogram/frameCache';

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
  colorScaleLocked: boolean;
  colorScaleParam: string | number | undefined;
  loading: boolean;
  frameLoading: boolean;
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

/**
 * Shared logic after a Tensoscope is created.
 *
 * `coordinates` is left null here — `selectField` fetches the coords
 * for the specific message it is decoding and sets them atomically
 * with `fieldData`, so the store's coordinates slot consistently
 * means "coords for the currently-decoded field".  Heterogeneous
 * multi-message files (different grids per message) would otherwise
 * keep msg-0's coords forever.
 */
async function initViewer(
  viewer: Tensoscope,
  set: (partial: Partial<AppState>) => void,
) {
  const index = await viewer.buildIndex();
  set({
    viewer,
    fileIndex: index,
    coordinates: null,
    selectedObject: null,
    selectedLevel: null,
    fieldData: null,
    fieldShape: [],
    fieldStats: null,
    colorScaleLocked: false,
    colorScaleParam: undefined,
    frameLoading: false,
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
  colorScaleLocked: false,
  colorScaleParam: undefined,
  loading: false,
  frameLoading: false,
  error: null,

  openLocalFile: async (file: File) => {
    set({ loading: true, error: null });
    try {
      const prev = get().viewer;
      if (prev) prev.close();
      const viewer = await Tensoscope.fromFile(file);
      initFrameCache(false);
      await initViewer(viewer, set);
      const first = get().fileIndex?.variables[0];
      if (first) await get().selectField(first.msgIndex, first.objIndex);
    } catch (err) {
      set({ error: String(err) });
    } finally {
      set({ loading: false });
    }
  },

  openUrl: async (url: string) => {
    set({ loading: true, error: null });
    try {
      const prev = get().viewer;
      if (prev) prev.close();
      const viewer = await Tensoscope.fromUrl(url);
      initFrameCache(true);
      await initViewer(viewer, set);
      const first = get().fileIndex?.variables[0];
      if (first) await get().selectField(first.msgIndex, first.objIndex);
    } catch (err) {
      set({ error: String(err) });
    } finally {
      set({ loading: false });
    }
  },

  selectField: async (msgIdx: number, objIdx: number) => {
    const { viewer, fileIndex, selectedLevel } = get();
    if (!viewer || !fileIndex) return;

    set({ error: null, selectedObject: { msgIdx, objIdx } });

    try {
      const varInfo = fileIndex.variables.find(
        (v) => v.msgIndex === msgIdx && v.objIndex === objIdx,
      );
      const mars = varInfo?.metadata?.mars as Record<string, unknown> | undefined;
      const marsParam = (mars?.param as string | number | undefined) ?? varInfo?.name;
      const style = getAutoStyle(marsParam);

      const frameCache = getFrameCache();
      const cached = frameCache?.get(msgIdx, objIdx);

      let coords: CoordinateData | null;
      let result: DecodedField;
      let originalShape: number[];

      if (cached) {
        coords = cached.coordinates;
        originalShape = cached.shape;
        result = { data: cached.data, shape: cached.shape, stats: cached.stats };
      } else {
        set({ frameLoading: true });
        // Yield to prevent WASM from blocking the main thread during decode.
        await new Promise<void>((resolve) => setTimeout(resolve, 0));

        coords = await viewer.fetchCoordinates(msgIdx);
        originalShape = varInfo?.shape ?? [];
        const coordLength = coords?.lat.length ?? 0;
        const sliceDim = decideSliceDim(originalShape, coordLength);

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

        frameCache?.put(msgIdx, objIdx, {
          data: result.data,
          coordinates: coords,
          stats: result.stats,
          shape: originalShape,
        });
      }

      // Re-read lock state after the async decode — user may have changed it.
      const { colorScaleLocked, colorScaleParam } = get();
      const paramChanged = marsParam !== colorScaleParam;

      let newColorScale: ColorScaleConfig;
      let nextLocked: boolean;

      if (paramChanged) {
        // New param: apply auto-style and clear any user lock.
        newColorScale = style
          ? { ...DEFAULT_COLOR_SCALE, palette: style.palette, min: style.min, max: style.max, logScale: false, nativeUnits: style.units, displayUnit: style.units }
          : { ...get().colorScale, min: result.stats.min, max: result.stats.max, nativeUnits: '', displayUnit: '' };
        nextLocked = false;
      } else if (colorScaleLocked) {
        // Same param, user has customised: preserve their settings.
        newColorScale = get().colorScale;
        nextLocked = true;
      } else {
        // Same param, auto mode: keep fixed auto-style range or auto-scale data range.
        newColorScale = style
          ? get().colorScale
          : { ...get().colorScale, min: result.stats.min, max: result.stats.max };
        nextLocked = false;
      }

      // Commit data and coords together so consumers never see a field
      // decoded against one message paired with coords from another.
      set({
        frameLoading: false,
        fieldData: result.data,
        fieldShape: originalShape,
        fieldStats: result.stats,
        coordinates: coords,
        colorScale: newColorScale,
        colorScaleLocked: nextLocked,
        colorScaleParam: marsParam,
      });
    } catch (err) {
      set({ frameLoading: false, error: String(err) });
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
    set((state) => ({
      colorScale: { ...state.colorScale, ...config },
      colorScaleLocked: true,
    }));
  },
}));

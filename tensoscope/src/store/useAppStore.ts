/** Global application state managed via Zustand. */

import { create } from 'zustand';
import { Tensoscope } from '../tensogram';
import type { FileIndex, DecodedField, CoordinateData } from '../tensogram';
import { getAutoStyle } from '../components/map/autoStyles';
import type { CustomStop } from '../components/map/colormaps';

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
    colorScale: DEFAULT_COLOR_SCALE,
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

      let result: DecodedField;
      if (originalShape.length > 1 && coordLength > 0) {
        const sliceDim = originalShape.findIndex((s) => s !== coordLength);
        if (sliceDim >= 0) {
          // Resolve slice index from selectedLevel if possible
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
      } else {
        result = await viewer.decodeField(msgIdx, objIdx);
      }

      // Apply per-variable auto-style if available, otherwise use data range
      const mars = varInfo?.metadata?.mars as Record<string, unknown> | undefined;
      const param = (mars?.param as string) ?? varInfo?.name;
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

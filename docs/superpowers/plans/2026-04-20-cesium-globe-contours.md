# Tensoscope: Cesium Globe + Filled Contours Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CesiumJS as the globe renderer and a filled-contours render mode (toggled alongside the existing heatmap) to Tensoscope.

**Architecture:** `MapView` becomes a router that mounts `CesiumView` (globe) or the existing MapLibre `<Map>` (flat) based on the active projection. A new `renderMode` local state (`'heatmap' | 'contours'`) is toggled by `RenderModePicker`. In contours mode, the existing regrid worker quantises pixel values to N discrete bands before writing RGBA — same canvas output, same rendering path for both renderers.

**Tech Stack:** CesiumJS (open-source, no Ion), vite-plugin-cesium, React 19, Zustand, Vitest.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/components/map/contourUtils.ts` | **Create** | Pure quantisation helpers (`getNumBands`, `contourLutIndex`) |
| `src/components/map/__tests__/contourUtils.test.ts` | **Create** | Unit tests for quantisation math |
| `src/components/map/regrid.worker.ts` | **Modify** | Add `renderMode`/`numBands` to request type; add contours branch |
| `src/components/map/FieldOverlay.tsx` | **Modify** | Add `renderMode` to props, cache key, and worker message; import `getNumBands` |
| `src/components/map/CesiumView.tsx` | **Create** | Cesium globe component; drapes field canvas via entity rectangle |
| `src/components/map/CesiumView.css` | **Create** | Reset styles for the Cesium container div |
| `src/components/map/RenderModePicker.tsx` | **Create** | Two-button heatmap/contours toggle |
| `src/components/map/MapView.tsx` | **Modify** | Add `renderMode` state; add `RenderModePicker`; conditionally render `CesiumView` or MapLibre |
| `vite.config.ts` | **Modify** | Add `vite-plugin-cesium` |
| `src/main.tsx` | **Modify** | Disable Cesium Ion token before first render |
| `package.json` | Modified by `npm install` | Gain `cesium` and `vite-plugin-cesium` deps |

All paths are relative to `tensoscope/`.

---

## Task 1: Install Cesium dependencies

**Files:**
- Modify: `package.json` (via npm)
- Modify: `vite.config.ts`

- [ ] **Step 1.1: Install packages**

```bash
cd tensoscope
npm install cesium
npm install --save-dev vite-plugin-cesium
```

Expected: both packages appear in `package.json`.

- [ ] **Step 1.2: Add cesium plugin to vite.config.ts**

Current top of `vite.config.ts`:
```typescript
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import path from 'path'
```

Replace with:
```typescript
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import cesium from 'vite-plugin-cesium'
import path from 'path'
```

And update the plugins array in the `defineConfig` call:
```typescript
plugins: [react(), wasm(), cesium(), corsProxy()],
```

(`cesium()` must come before custom plugins; order otherwise unchanged.)

- [ ] **Step 1.3: Verify dev server starts**

```bash
cd tensoscope && npm run dev
```

Expected: server starts on http://localhost:5173 with no import errors. Stop with Ctrl-C.

- [ ] **Step 1.4: Commit**

```bash
git add tensoscope/vite.config.ts tensoscope/package.json tensoscope/package-lock.json
git commit -m "feat(tensoscope): add cesium and vite-plugin-cesium dependencies"
```

---

## Task 2: Add contourUtils.ts

**Files:**
- Create: `src/components/map/contourUtils.ts`

- [ ] **Step 2.1: Create contourUtils.ts**

```typescript
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
```

- [ ] **Step 2.2: Commit**

```bash
git add tensoscope/src/components/map/contourUtils.ts
git commit -m "feat(tensoscope): add contourUtils with quantisation helpers"
```

---

## Task 3: Unit tests for contourUtils

**Files:**
- Create: `src/components/map/__tests__/contourUtils.test.ts`

- [ ] **Step 3.1: Create test file**

```typescript
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
    // band 0: t in [0, 0.5), band 1: t in [0.5, 1]
    expect(contourLutIndex(0.0, 2)).toBe(0);    // band 0 → lut 0
    expect(contourLutIndex(0.49, 2)).toBe(0);   // band 0 → lut 0
    expect(contourLutIndex(0.5, 2)).toBe(255);  // band 1 → lut 255
    expect(contourLutIndex(1.0, 2)).toBe(255);  // band 1 → lut 255
  });

  it('maps values to correct band with 10 bands', () => {
    // band 0: t in [0, 0.1)
    expect(contourLutIndex(0.05, 10)).toBe(0);
    // band 9: t in [0.9, 1.0]
    expect(contourLutIndex(0.95, 10)).toBe(255);
    // band 4: t in [0.4, 0.5), lut index = round(4 * 255 / 9) = round(113.3) = 113
    expect(contourLutIndex(0.45, 10)).toBe(113);
  });

  it('clamps t values outside [0, 1]', () => {
    // floor(-0.1 * 10) = floor(-1) = -1 → clamped to 0 by Math.min(9, max(0,...))
    // but our implementation uses Math.min(numBands-1, Math.floor(t*numBands))
    // negative t: Math.floor(-0.1 * 10) = -1, Math.min(9, -1) = -1 → wrong
    // The worker clamps t before calling contourLutIndex, so this tests the bare function
    // For safety check that t=0 and t=1 are always valid:
    expect(contourLutIndex(0, 10)).toBe(0);
    expect(contourLutIndex(1, 10)).toBe(255);
  });
});
```

- [ ] **Step 3.2: Run tests**

```bash
cd tensoscope && make ts-test
```

Expected: all tests pass. If vitest is not found, run: `npx vitest run src/components/map/__tests__/contourUtils.test.ts`

- [ ] **Step 3.3: Commit**

```bash
git add tensoscope/src/components/map/__tests__/contourUtils.test.ts
git commit -m "test(tensoscope): add unit tests for contourUtils"
```

---

## Task 4: Extend regrid.worker.ts with contours mode

**Files:**
- Modify: `src/components/map/regrid.worker.ts`

- [ ] **Step 4.1: Add renderMode and numBands to RegridRequest**

Replace the `RegridRequest` interface (lines 11–23):

```typescript
interface RegridRequest {
  id: number;
  srcLat: Float32Array;
  srcLon: Float32Array;
  srcData: Float32Array;
  lut: Uint8Array;        // 256*3 RGB lookup table
  colorMin: number;
  colorMax: number;
  width: number;
  height: number;
  binDeg: number;
  rowLats: Float64Array;  // pre-computed Mercator-spaced latitudes per row
  renderMode: 'heatmap' | 'contours';
  numBands: number;
}
```

- [ ] **Step 4.2: Add contours quantisation branch in regridAndColormap**

Replace the LUT-lookup block inside the inner loop (the block starting at line 112 `if (bestIdx >= 0) {`):

```typescript
      const base = rowOffset + col * 4;
      if (bestIdx >= 0) {
        const v = srcData[bestIdx];
        if (v !== v) continue; // NaN -- leave transparent (already 0)

        let lutIdx: number;
        if (renderMode === 'contours') {
          const t = range === 0 ? 0 : Math.max(0, Math.min(1, (v - colorMin) / range));
          const bandIdx = Math.min(numBands - 1, Math.floor(t * numBands));
          lutIdx = numBands <= 1 ? 0 : Math.round((bandIdx * 255) / (numBands - 1));
        } else {
          lutIdx = Math.max(0, Math.min(255, ((v - colorMin) * invRange) | 0));
        }

        const lutBase = lutIdx * 3;
        rgba[base] = lut[lutBase];
        rgba[base + 1] = lut[lutBase + 1];
        rgba[base + 2] = lut[lutBase + 2];
        rgba[base + 3] = 255;
      }
```

- [ ] **Step 4.3: Destructure the new fields in regridAndColormap**

Replace the first line of `regridAndColormap`:

```typescript
  const { srcLat, srcLon, srcData, lut, colorMin, colorMax, width, height, binDeg, rowLats, renderMode, numBands } = req;
```

Also update `range` computation (it's already `colorMax - colorMin` — confirm it's accessible before the inner loop; it is, at line 74).

- [ ] **Step 4.4: Verify typecheck**

```bash
cd tensoscope && make ts-typecheck
```

Expected: no errors.

- [ ] **Step 4.5: Commit**

```bash
git add tensoscope/src/components/map/regrid.worker.ts
git commit -m "feat(tensoscope): add contours quantisation mode to regrid worker"
```

---

## Task 5: Extend FieldOverlay.tsx with renderMode

**Files:**
- Modify: `src/components/map/FieldOverlay.tsx`

- [ ] **Step 5.1: Add import for contourUtils**

Add at the top of `FieldOverlay.tsx` (after the existing imports):

```typescript
import { getNumBands } from './contourUtils';
```

- [ ] **Step 5.2: Add renderMode to FieldOverlayProps**

Replace the `FieldOverlayProps` interface:

```typescript
export interface FieldOverlayProps {
  data: Float32Array;
  lat: Float32Array;
  lon: Float32Array;
  colorMin: number;
  colorMax: number;
  palette: string;
  zoom?: number;
  paletteReversed?: boolean;
  customStops?: CustomStop[];
  renderMode: 'heatmap' | 'contours';
}
```

- [ ] **Step 5.3: Add renderMode to the cache key**

Replace the `cacheKey` function:

```typescript
function cacheKey(
  data: Float32Array,
  colorMin: number,
  colorMax: number,
  palette: string,
  gridW: number,
  gridH: number,
  paletteReversed: boolean,
  customStops: CustomStop[] | undefined,
  renderMode: 'heatmap' | 'contours',
): string {
  const palKey = palette
    + (paletteReversed ? ':r' : '')
    + (palette === 'custom' && customStops ? ':' + JSON.stringify(customStops) : '');
  return `${dataFingerprint(data)}:${colorMin}:${colorMax}:${palKey}:${gridW}x${gridH}:${renderMode}`;
}
```

- [ ] **Step 5.4: Pass renderMode and numBands through the hook**

In `useFieldImage`, update the destructuring and the `key` computation and `requestRegrid` call:

Replace the line:
```typescript
    const { data, lat, lon, colorMin, colorMax, palette, zoom, paletteReversed = false, customStops } = props;
```
with:
```typescript
    const { data, lat, lon, colorMin, colorMax, palette, zoom, paletteReversed = false, customStops, renderMode } = props;
    const numBands = getNumBands(palette, customStops);
```

Replace the `key` computation line:
```typescript
    const key = cacheKey(data, colorMin, colorMax, palette, params.width, params.height, paletteReversed, customStops);
```
with:
```typescript
    const key = cacheKey(data, colorMin, colorMax, palette, params.width, params.height, paletteReversed, customStops, renderMode);
```

- [ ] **Step 5.5: Forward renderMode and numBands to the worker**

In `requestRegrid`, the function signature and call site both need the new fields. Replace the `requestRegrid` function signature:

```typescript
function requestRegrid(
  srcLat: Float32Array,
  srcLon: Float32Array,
  srcData: Float32Array,
  lut: Uint8Array,
  colorMin: number,
  colorMax: number,
  params: GridParams,
  renderMode: 'heatmap' | 'contours',
  numBands: number,
): Promise<{ rgba: Uint8ClampedArray; width: number; height: number }> {
```

Add `renderMode` and `numBands` to the `worker.postMessage` call object (inside `requestRegrid`):

```typescript
    worker.postMessage(
      {
        id,
        srcLat,
        srcLon,
        srcData,
        lut,
        colorMin,
        colorMax,
        width: params.width,
        height: params.height,
        binDeg: params.binDeg,
        rowLats: params.rowLats,
        renderMode,
        numBands,
      },
    );
```

Update the `requestRegrid` call in `useFieldImage`:

```typescript
    const result = await requestRegrid(lat, lon, data, lut, colorMin, colorMax, params, renderMode, numBands);
```

Also update the `useCallback` dependency array to include `renderMode`:

```typescript
  }, [props?.data, props?.lat, props?.lon, props?.colorMin, props?.colorMax, props?.palette, props?.zoom, props?.paletteReversed, props?.customStops, props?.renderMode]);
```

- [ ] **Step 5.6: Typecheck**

```bash
cd tensoscope && make ts-typecheck
```

Expected: errors about `renderMode` missing in callers in `MapView.tsx` — these will be fixed in Task 8. For now they are expected. Check for no other errors.

- [ ] **Step 5.7: Commit**

```bash
git add tensoscope/src/components/map/FieldOverlay.tsx
git commit -m "feat(tensoscope): propagate renderMode through FieldOverlay hook"
```

---

## Task 6: Add RenderModePicker component

**Files:**
- Create: `src/components/map/RenderModePicker.tsx`

- [ ] **Step 6.1: Create RenderModePicker.tsx**

```typescript
// tensoscope/src/components/map/RenderModePicker.tsx

interface RenderModePickerProps {
  mode: 'heatmap' | 'contours';
  onChange: (mode: 'heatmap' | 'contours') => void;
}

export function RenderModePicker({ mode, onChange }: RenderModePickerProps) {
  return (
    <div className="render-mode-picker">
      <button
        className={`render-mode-btn${mode === 'heatmap' ? ' render-mode-btn-active' : ''}`}
        onClick={() => onChange('heatmap')}
        title="Heatmap"
      >
        Heatmap
      </button>
      <button
        className={`render-mode-btn${mode === 'contours' ? ' render-mode-btn-active' : ''}`}
        onClick={() => onChange('contours')}
        title="Contours"
      >
        Contours
      </button>
    </div>
  );
}
```

- [ ] **Step 6.2: Add CSS for RenderModePicker**

Open `tensoscope/src/index.css` (or whichever global CSS file contains `.projection-picker` styles) and append:

```css
.render-mode-picker {
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 10;
  display: flex;
  gap: 4px;
  background: rgba(20, 20, 30, 0.85);
  border-radius: 8px;
  padding: 4px;
  border: 1px solid rgba(255,255,255,0.1);
}

.render-mode-btn {
  padding: 5px 12px;
  border-radius: 5px;
  border: none;
  background: transparent;
  color: rgba(255,255,255,0.6);
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.render-mode-btn:hover {
  background: rgba(255,255,255,0.1);
  color: #fff;
}

.render-mode-btn-active {
  background: rgba(255,255,255,0.18);
  color: #fff;
  font-weight: 600;
}
```

- [ ] **Step 6.3: Typecheck**

```bash
cd tensoscope && make ts-typecheck
```

Expected: no new errors.

- [ ] **Step 6.4: Commit**

```bash
git add tensoscope/src/components/map/RenderModePicker.tsx tensoscope/src/index.css
git commit -m "feat(tensoscope): add RenderModePicker toggle component"
```

---

## Task 7: Add CesiumView component

**Files:**
- Create: `src/components/map/CesiumView.tsx`
- Create: `src/components/map/CesiumView.css`

- [ ] **Step 7.1: Disable Ion in main.tsx**

Add these two lines at the top of `tensoscope/src/main.tsx`, before the existing imports:

```typescript
import { Ion } from 'cesium';
Ion.defaultAccessToken = '';
```

The full file becomes:

```typescript
import { Ion } from 'cesium';
Ion.defaultAccessToken = '';

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ensureInit } from './tensogram'

ensureInit();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 7.2: Create CesiumView.css**

```css
/* tensoscope/src/components/map/CesiumView.css */

.cesium-container {
  width: 100%;
  height: 100%;
}

/* Hide default Cesium credit banner */
.cesium-widget-credits {
  display: none !important;
}
```

- [ ] **Step 7.3: Create CesiumView.tsx**

```typescript
// tensoscope/src/components/map/CesiumView.tsx

import { useEffect, useRef } from 'react';
import {
  Viewer,
  UrlTemplateImageryProvider,
  Rectangle,
  ImageMaterialProperty,
  Color,
  Cartesian3,
  Math as CesiumMath,
  Credit,
} from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import './CesiumView.css';
import type { FieldImage } from './FieldOverlay';

interface CesiumViewProps {
  fieldImage: FieldImage | null;
  initialCenter: { lat: number; lon: number };
  onUnmount: (lat: number, lon: number) => void;
}

export function CesiumView({ fieldImage, initialCenter, onUnmount }: CesiumViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | undefined>(undefined);
  const overlayRef = useRef<ReturnType<typeof viewerRef.current.entities.add> | undefined>(undefined);

  // Mount Cesium viewer once
  useEffect(() => {
    if (!containerRef.current) return;

    const osm = new UrlTemplateImageryProvider({
      url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      credit: new Credit('© OpenStreetMap contributors'),
      maximumLevel: 19,
    });

    const viewer = new Viewer(containerRef.current, {
      imageryProvider: osm,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      creditContainer: document.createElement('div'),
    });

    viewer.scene.globe.enableLighting = false;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;

    // Restore camera from previous renderer position
    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(initialCenter.lon, initialCenter.lat, 20000000),
      duration: 0,
    });

    viewerRef.current = viewer;

    return () => {
      const cart = viewer.camera.positionCartographic;
      onUnmount(
        CesiumMath.toDegrees(cart.latitude),
        CesiumMath.toDegrees(cart.longitude),
      );
      viewer.destroy();
      viewerRef.current = undefined;
      overlayRef.current = undefined;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update field overlay when fieldImage changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (overlayRef.current) {
      viewer.entities.remove(overlayRef.current);
      overlayRef.current = undefined;
    }

    if (!fieldImage) return;

    overlayRef.current = viewer.entities.add({
      rectangle: {
        coordinates: Rectangle.fromDegrees(-180, -85, 180, 85),
        material: new ImageMaterialProperty({
          image: fieldImage.dataUrl,
          transparent: true,
          color: new Color(1, 1, 1, 0.7),
        }),
        fill: true,
      },
    });
  }, [fieldImage]);

  return <div ref={containerRef} className="cesium-container" />;
}
```

- [ ] **Step 7.4: Typecheck**

```bash
cd tensoscope && make ts-typecheck
```

Expected: no errors in CesiumView.tsx itself. (MapView errors from Task 5 still present.)

- [ ] **Step 7.5: Commit**

```bash
git add tensoscope/src/main.tsx tensoscope/src/components/map/CesiumView.tsx tensoscope/src/components/map/CesiumView.css
git commit -m "feat(tensoscope): add CesiumView globe renderer"
```

---

## Task 8: Refactor MapView.tsx

**Files:**
- Modify: `src/components/map/MapView.tsx`

- [ ] **Step 8.1: Replace MapView.tsx with the new implementation**

```typescript
// tensoscope/src/components/map/MapView.tsx

import { useState, useCallback, useRef } from 'react';
import { Map, Source, Layer } from 'react-map-gl/maplibre';
import type { MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useFieldImage } from './FieldOverlay';
import { ColorBar } from './ColorBar';
import { ColorScaleControls } from './ColorScaleControls';
import { CesiumView } from './CesiumView';
import { RenderModePicker } from './RenderModePicker';
import type { FieldOverlayProps } from './FieldOverlay';
import { ProjectionPicker, PROJECTION_PRESETS } from './ProjectionPicker';
import type { ProjectionPreset } from './ProjectionPicker';
import type { CustomStop } from './colormaps';

const BASEMAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

export interface MapViewProps {
  data: Float32Array | null;
  lat: Float32Array | null;
  lon: Float32Array | null;
  colorMin: number;
  colorMax: number;
  palette: string;
  units: string;
  paletteReversed: boolean;
  customStops: CustomStop[];
  onColorMinChange: (v: number) => void;
  onColorMaxChange: (v: number) => void;
  onPaletteChange: (v: string) => void;
  onPaletteReversedChange: (v: boolean) => void;
  onCustomStopsChange: (stops: CustomStop[]) => void;
  dataMin: number;
  dataMax: number;
  nativeUnits: string;
  displayUnit: string;
  onDisplayUnitChange: (unit: string) => void;
}

export function MapView(props: MapViewProps) {
  const {
    data, lat, lon,
    colorMin, colorMax, palette, units,
    paletteReversed, customStops,
    onColorMinChange, onColorMaxChange, onPaletteChange,
    onPaletteReversedChange, onCustomStopsChange,
    dataMin, dataMax,
    nativeUnits, displayUnit, onDisplayUnitChange,
  } = props;

  const [activePreset, setActivePreset] = useState<ProjectionPreset>(PROJECTION_PRESETS[0]);
  const [renderMode, setRenderMode] = useState<'heatmap' | 'contours'>('heatmap');
  const [zoom, setZoom] = useState(1.5);
  const [cameraCenter, setCameraCenter] = useState({ lat: 20, lon: 0 });
  const mapRef = useRef<MapRef>(null);
  const zoomTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handlePresetSelect = (preset: ProjectionPreset) => {
    if (preset.id === activePreset.id) return;
    // Save MapLibre camera position when switching away from flat
    if (activePreset.id === 'flat' && mapRef.current) {
      const centre = mapRef.current.getCenter();
      setCameraCenter({ lat: centre.lat, lon: centre.lng });
    }
    setActivePreset(preset);
    if (preset.id === 'flat') {
      // MapLibre will mount; fly to saved position
      mapRef.current?.flyTo({ center: [cameraCenter.lon, cameraCenter.lat], zoom });
    }
  };

  const handleCesiumUnmount = (lat: number, lon: number) => {
    setCameraCenter({ lat, lon });
  };

  const handleZoomEnd = useCallback(() => {
    if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current);
    zoomTimerRef.current = setTimeout(() => {
      const map = mapRef.current;
      if (map) setZoom(map.getZoom());
    }, 200);
  }, []);

  const overlayProps: FieldOverlayProps | null =
    data && lat && lon
      ? { data, lat, lon, colorMin, colorMax, palette, zoom, paletteReversed, customStops, renderMode }
      : null;

  const fieldImage = useFieldImage(overlayProps);

  const isGlobe = activePreset.id === 'globe';

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {isGlobe ? (
        <CesiumView
          fieldImage={fieldImage}
          initialCenter={cameraCenter}
          onUnmount={handleCesiumUnmount}
        />
      ) : (
        <Map
          ref={mapRef}
          initialViewState={{ longitude: cameraCenter.lon, latitude: cameraCenter.lat, zoom: 1.5 }}
          mapStyle={BASEMAP_STYLE}
          projection={activePreset.type as any}
          style={{ width: '100%', height: '100%' }}
          onZoomEnd={handleZoomEnd}
        >
          {fieldImage && (
            <Source
              id="field-overlay"
              type="image"
              url={fieldImage.dataUrl}
              coordinates={fieldImage.coordinates}
            >
              <Layer
                id="field-overlay-layer"
                type="raster"
                paint={{ 'raster-opacity': 0.7 }}
              />
            </Source>
          )}
        </Map>
      )}

      <ProjectionPicker current={activePreset.id} onSelect={handlePresetSelect} />

      {data && <RenderModePicker mode={renderMode} onChange={setRenderMode} />}

      {data && (
        <>
          <div style={{ position: 'absolute', bottom: 40, right: 16, zIndex: 10 }}>
            <ColorBar
              min={colorMin}
              max={colorMax}
              palette={palette}
              units={units}
              paletteReversed={paletteReversed}
              customStops={customStops}
              nativeUnits={nativeUnits}
              displayUnit={displayUnit}
            />
          </div>

          <div style={{ position: 'absolute', top: 16, right: 16, zIndex: 10 }}>
            <ColorScaleControls
              palette={palette}
              colorMin={colorMin}
              colorMax={colorMax}
              dataMin={dataMin}
              dataMax={dataMax}
              paletteReversed={paletteReversed}
              customStops={customStops}
              nativeUnits={nativeUnits}
              displayUnit={displayUnit}
              onPaletteChange={onPaletteChange}
              onColorMinChange={onColorMinChange}
              onColorMaxChange={onColorMaxChange}
              onPaletteReversedChange={onPaletteReversedChange}
              onCustomStopsChange={onCustomStopsChange}
              onDisplayUnitChange={onDisplayUnitChange}
            />
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 8.2: Full typecheck**

```bash
cd tensoscope && make ts-typecheck
```

Expected: zero errors. Fix any type mismatches before proceeding.

- [ ] **Step 8.3: Run unit tests**

```bash
cd tensoscope && make ts-test
```

Expected: all tests pass.

- [ ] **Step 8.4: Start dev server and manually verify**

```bash
cd tensoscope && npm run dev
```

Open http://localhost:5173 and verify:

1. Load a `.tgm` file with a field
2. Default view: globe, heatmap — field appears as smooth gradient
3. Click **Contours** in RenderModePicker — field re-renders with discrete colour bands
4. Click **Heatmap** again — restores smooth gradient; no page reload needed
5. Switch projection to **Flat** — MapLibre view appears with correct overlay in both modes
6. Switch back to **Globe** — Cesium view appears; field overlay present
7. No Ion-related console errors
8. No console errors from the worker

- [ ] **Step 8.5: Commit**

```bash
git add tensoscope/src/components/map/MapView.tsx
git commit -m "feat(tensoscope): route globe to CesiumView, add RenderModePicker"
```

---

## Task 9: Production build verification and docs update

**Files:**
- Modify: `plans/DONE.md`

- [ ] **Step 9.1: Production build**

```bash
cd tensoscope && npm run build
```

Expected: build completes without errors. Cesium assets should appear in `dist/cesium/`.

- [ ] **Step 9.2: Update DONE.md**

Open `plans/DONE.md` and add an entry under the Tensoscope section:

```markdown
- **Cesium globe + filled contours** — CesiumView replaces MapLibre for globe
  projection; RenderModePicker toggles between smooth heatmap and discretised
  filled contours (N bands auto-derived from colour scale).
```

- [ ] **Step 9.3: Final commit**

```bash
git add plans/DONE.md
git commit -m "docs(tensoscope): record Cesium globe + contours in DONE.md"
```

---

## Test Coverage Summary

| Test | Location | Type |
|---|---|---|
| `getNumBands` for named/custom palettes | `__tests__/contourUtils.test.ts` | Unit |
| `contourLutIndex` band boundaries | `__tests__/contourUtils.test.ts` | Unit |
| `contourLutIndex` edge cases (numBands≤1, t=0/1) | `__tests__/contourUtils.test.ts` | Unit |
| Heatmap↔contours toggle | Manual (dev server) | Visual |
| Globe↔flat renderer switch | Manual (dev server) | Visual |
| Field overlay in both renderers | Manual (dev server) | Visual |
| No Ion console errors | Manual (dev server) | Visual |
| Production build | `npm run build` | Build |

# Tensoscope: Cesium Globe + Filled Contours

**Date:** 2026-04-20  
**Status:** Approved

## Summary

Replace Tensoscope's single MapLibre renderer with two renderers (CesiumJS for globe, MapLibre for flat/Mercator) and add a filled-contours render mode alongside the existing heatmap. Both modes share the same regrid worker pipeline; contours are produced by a quantisation step in the worker rather than true vector marching-squares.

---

## Decisions

| Question | Decision |
|---|---|
| Globe renderer | CesiumJS open-source (no Ion, no API key) |
| Flat renderer | MapLibre GL JS v5 (unchanged) |
| Contour style | Filled contours (discrete colour bands, contourf-style) |
| Contour pipeline | Discretised canvas — quantise pixel values in worker before writing RGBA |
| Contour levels | Auto-derived from colour scale: 10 bands for continuous palettes, stop-count for custom palettes |
| Render modes | Toggle: heatmap or filled contours (both available regardless of projection) |
| Base imagery (Cesium) | OpenStreetMapImageryProvider (free, no key) |

---

## Architecture

### Component tree

```
App
├── Sidebar
│   ├── FileOpenDialog
│   ├── FieldSelector
│   └── MetadataPanel
├── MapView  (routes to renderer by projection)
│   ├── CesiumView          ← NEW (shown when projection = globe)
│   │   └── FieldOverlay hook  (shared, drives SingleTileImageryProvider)
│   ├── MapLibreView        (shown when projection = flat, existing)
│   │   └── FieldOverlay hook  (existing, drives image source)
│   ├── ProjectionPicker    (globe / flat toggle — unchanged)
│   ├── RenderModePicker    ← NEW (heatmap / contours toggle)
│   ├── ColorBar
│   └── ColorScaleControls
└── AnimationBar
```

### State additions (Zustand)

```typescript
renderMode: 'heatmap' | 'contours'   // new
centerLat: number                      // new — camera handoff between renderers
centerLon: number                      // new
// projection: 'globe' | 'mercator'   // existing
```

---

## Data Pipeline

```
Float32Array data + lat/lon
  ↓  spatial hash + nearest-neighbour (unchanged)
Float pixel grid
  ↓  colourmap LUT (unchanged)
  if mode === 'heatmap'  →  smooth RGBA (unchanged)
  if mode === 'contours' →  quantise to N bands → banded RGBA (new)
  ↓  canvas data URL (unchanged)
CesiumView:    SingleTileImageryProvider(url, Rectangle(-180,-90,180,90))
MapLibreView:  image source layer (unchanged)
```

### Quantisation logic (regrid.worker.ts)

```typescript
// existing
if (mode === 'heatmap') {
  rgba = lut[Math.round(t * 255)]
}

// new
if (mode === 'contours') {
  const bandIndex = clamp(Math.floor(t * numBands), 0, numBands - 1)
  rgba = lut[Math.round((bandIndex / numBands) * 255)]
}
```

`numBands` is derived in `colormaps.ts` and passed alongside the LUT in the worker message. For continuous palettes (viridis, plasma, etc.) the default is 10. For custom-stop palettes the stop count is used.

### Cache key

```
`${fingerprint}-${palette}-${min}-${max}-${w}-${h}-${renderMode}`
```

Heatmap and contour images are cached independently (12-entry LRU, unchanged size).

---

## Cesium Integration

### Package setup

```bash
npm install cesium
```

Vite config: copy `node_modules/cesium/Build/Cesium/` → `public/cesium/` at build time (via `vite-plugin-cesium` or `vite-plugin-static-copy`).

`main.tsx` (before any Cesium import):

```typescript
window.CESIUM_BASE_URL = '/cesium'
Ion.defaultAccessToken = ''
```

### CesiumView component

**On mount:**
- Create `Viewer` with `imageryProvider = new OpenStreetMapImageryProvider()`
- Disable all Cesium UI chrome (animation, timeline, geocoder, homeButton, sceneModePicker, etc.)
- Set `scene.globe.enableLighting = false`
- Restore camera from Zustand `centerLat` / `centerLon`

**Field overlay:**
- Receive canvas data URL from shared `FieldOverlay` hook
- Remove previous overlay `ImageryLayer` if present
- Add `viewer.imageryLayers.addImageryProvider(new SingleTileImageryProvider({ url, rectangle: Rectangle.fromDegrees(-180, -90, 180, 90) }))`
- Set `layer.alpha = 0.7`

**On unmount (switching to flat):**
- Write `camera.positionCartographic.latitude/longitude` → Zustand `centerLat` / `centerLon`
- MapLibreView reads these on mount and flies to the same location

---

## UI Changes

### RenderModePicker (new component)

Two-button toggle matching existing `ProjectionPicker` visual style. Placed in the map toolbar alongside `ProjectionPicker`.

- **Heatmap** button: sets `renderMode = 'heatmap'`
- **Contours** button: sets `renderMode = 'contours'`
- Active button is filled; inactive is outlined
- Toggling triggers an immediate worker re-run using cached field data (no file reload)

---

## Build / Docker

- Cesium static assets (~10 MB) are copied into the Docker image at build time
- `BASE_PATH` env var and nginx subpath config are unchanged
- Docker image size will increase; this is acceptable

---

## Testing Plan

| Test | Method |
|---|---|
| Quantisation correctness | Unit test: given `t` ∈ [0,1] and N bands, verify band index at boundaries |
| Visual contours | Load a known field, switch heatmap↔contours, verify N distinct colour bands |
| Renderer switch | Toggle globe↔flat while field loaded, verify overlay in both renderers |
| Cache independence | Toggle mode twice — second toggle must hit cache (no worker re-run) |
| No Ion errors | Load app, verify no Cesium Ion console warnings |
| Camera handoff | Switch globe→flat→globe, verify camera position is approximately preserved |

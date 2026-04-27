import { useState, useCallback, useRef, useEffect } from 'react';
import { Map, Source, Layer } from 'react-map-gl/maplibre';
import type { MapRef, MapLayerMouseEvent } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { ClickPoint } from './usePointInspection';
import { useFieldImage } from './FieldOverlay';
import { ColorBar } from './ColorBar';
import { ColorScaleControls } from './ColorScaleControls';
import { CesiumView } from './CesiumView';
import { RenderModePicker } from './RenderModePicker';
import type { FieldOverlayProps, ViewBounds } from './FieldOverlay';
import { ProjectionPicker, PROJECTION_PRESETS } from './ProjectionPicker';
import type { ProjectionPreset } from './ProjectionPicker';
import type { CustomStop } from './colormaps';

const BASEMAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

// Persistent state keyed in localStorage (works with the user's "store in a
// cookie" intent — same scope, simpler API, no server round-trip). Stored
// values are JSON-encoded so booleans, numbers and small objects all round-
// trip safely. Failures (Safari private mode, quota) silently fall back to
// the in-memory value.
function usePersistedState<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) return JSON.parse(raw) as T;
    } catch { /* ignore */ }
    return initial;
  });
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
  }, [key, value]);
  return [value, setValue];
}

// Front render covers the visible viewport extended by this factor, so small
// pans within the buffered rect do not require a re-render. The back layer
// remains visible (never the basemap) while the front catches up.
const FRONT_BUFFER_FACTOR = 1.5;

// Once the buffered viewport spans more than this many longitude degrees the
// front layer is skipped: the back already covers the visible area at its
// full resolution, so adding a front would only introduce a rasterised-quad
// seam at its outer edge for no resolution gain. resolveBounds also promotes
// to global at 330°, so this matches its threshold.
const FRONT_SKIP_LON_SPAN_DEG = 330;

// Map maxZoom is derived from data spacing so the user cannot zoom past the
// resolution of the underlying field. The formula `log2(1.4 / avgSpacing)`
// is the zoom at which one MapLibre tile pixel equals one data cell at the
// equator (360° / 256 px ≈ 1.4 °/px). Adding ZOOM_HEADROOM levels gives
// some upsampling space (cells visible but not blocky).
const MAX_ZOOM_HEADROOM = 4;
const MAX_ZOOM_ABSOLUTE_FLOOR = 2;
const MAX_ZOOM_ABSOLUTE_CEILING = 12;

function computeMaxZoom(nPoints: number | null): number | undefined {
  if (!nPoints || nPoints < 4) return undefined;
  const avgSpacing = Math.sqrt(360 * 170 / nPoints);
  const z = Math.ceil(Math.log2(1.40625 / avgSpacing)) + MAX_ZOOM_HEADROOM;
  return Math.max(MAX_ZOOM_ABSOLUTE_FLOOR, Math.min(MAX_ZOOM_ABSOLUTE_CEILING, z));
}

// Back-layer resolution at full extent. Used to align the Cesium front rect
// to the back's bilinear blend zone (half-pixel extension) on the globe.
const BACK_GEO_W = 2048;
const BACK_GEO_H = 1024;
const BACK_FLAT_W = 1440;
const BACK_FLAT_H = 720;

function bufferBounds(
  bounds: ViewBounds,
  factor: number,
  latMin: number,
  latMax: number,
  lonMin: number,
  lonMax: number,
): ViewBounds {
  const lonPad = ((factor - 1) / 2) * (bounds.east - bounds.west);
  const latPad = ((factor - 1) / 2) * (bounds.north - bounds.south);
  return {
    west: Math.max(lonMin, bounds.west - lonPad),
    east: Math.min(lonMax, bounds.east + lonPad),
    south: Math.max(latMin, bounds.south - latPad),
    north: Math.min(latMax, bounds.north + latPad),
  };
}

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
  colorScaleInitialMinimised?: boolean;
  onMapClick?: (point: ClickPoint) => void;
  selectedPoint?: { lat: number; lon: number } | null;
  selectedPointGridSpacing?: number | null;
  onSelectedPointScreen?: (x: number, y: number) => void;
  onSelectedPointOutOfView?: () => void;
  isLoading?: boolean;
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
    colorScaleInitialMinimised,
    onMapClick,
    selectedPoint,
    selectedPointGridSpacing,
    onSelectedPointScreen,
    onSelectedPointOutOfView,
    isLoading,
  } = props;

  const [activePreset, setActivePreset] = useState<ProjectionPreset>(PROJECTION_PRESETS[0]);
  const [renderMode, setRenderMode] = useState<'heatmap' | 'contours'>('heatmap');
  const [highResEnabled, setHighResEnabled] = usePersistedState('tensoscope.display.highRes', true);
  const [showLabels, setShowLabels] = usePersistedState('tensoscope.display.labels', false);
  const [showLines, setShowLines] = usePersistedState('tensoscope.display.lines', true);
  const [configOpen, setConfigOpen] = useState(false);
  // Ids of basemap line/symbol layers above the field anchor — populated
  // on map load and used by the toggle effects to flip layer visibility
  // without re-walking the style every time.
  const basemapLineIdsRef = useRef<string[]>([]);
  const basemapSymbolIdsRef = useRef<string[]>([]);
  const [cameraCenter, setCameraCenter] = useState({ lat: 20, lon: 0 });
  const mapRef = useRef<MapRef>(null);
  const justSwitchedToFlatRef = useRef(false);

  // Flat map viewport state
  const [viewportBounds, setViewportBounds] = useState<ViewBounds | null>(null);
  const [viewportSize, setViewportSize] = useState<{ width: number; height: number } | null>(null);

  // Id of the topmost basemap line/symbol layer; the field overlay is
  // inserted beneath it so coastlines, country borders and place labels
  // render on top of the field. Resolved on map load by walking the active
  // style; null when the basemap is still loading or the style has no
  // matching layer (in which case the field stays on top — same as before).
  const [overlayBeforeId, setOverlayBeforeId] = useState<string | null>(null);

  // Globe viewport state (from Cesium camera change callback)
  const [cesiumBounds, setCesiumBounds] = useState<ViewBounds | null>(null);
  const [cesiumViewSize, setCesiumViewSize] = useState<{ width: number; height: number }>({ width: 1024, height: 512 });

  const selectedPointRef = useRef(selectedPoint);
  const onSelectedPointScreenRef = useRef(onSelectedPointScreen);
  const onSelectedPointOutOfViewRef = useRef(onSelectedPointOutOfView);
  useEffect(() => { selectedPointRef.current = selectedPoint; }, [selectedPoint]);
  useEffect(() => { onSelectedPointScreenRef.current = onSelectedPointScreen; }, [onSelectedPointScreen]);
  useEffect(() => { onSelectedPointOutOfViewRef.current = onSelectedPointOutOfView; }, [onSelectedPointOutOfView]);

  const captureViewport = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const b = map.getBounds();
    setViewportBounds({ west: b.getWest(), east: b.getEast(), south: b.getSouth(), north: b.getNorth() });
    const canvas = map.getCanvas();
    setViewportSize({ width: canvas.width, height: canvas.height });

    const pt = selectedPointRef.current;
    if (!pt) return;
    const screen = map.project([pt.lon, pt.lat]);
    onSelectedPointScreenRef.current?.(screen.x, screen.y);
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const margin = 80;
    if (screen.x < -margin || screen.x > cw + margin || screen.y < -margin || screen.y > ch + margin) {
      onSelectedPointOutOfViewRef.current?.();
    }
  }, []);

  const handleMapLoad = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const style = map.getStyle();
    const layers = style?.layers ?? [];
    const ml = map.getMap();

    // Hide the water fill entirely (it would otherwise cover the field in
    // oceans), and replace its visual contribution with an explicit
    // coastline line layer: traces the boundary of every water polygon
    // using the same vector source the water fill consumes.
    const waterIdRe = /\b(water|ocean|sea)\b/i;
    let waterFill: typeof layers[number] | null = null;
    for (const l of layers) {
      if ((l.type === 'fill' || l.type === 'fill-extrusion') && waterIdRe.test(l.id)) {
        ml.setPaintProperty(l.id, l.type === 'fill' ? 'fill-opacity' : 'fill-extrusion-opacity', 0);
        if (!waterFill) waterFill = l;
      }
    }
    if (waterFill && 'source' in waterFill && 'source-layer' in waterFill) {
      const coastlineId = 'coastline-overlay';
      if (!ml.getLayer(coastlineId)) {
        // Insert above the first symbol layer so the line sits on top of
        // the field (and other line layers) but below labels — matches
        // the rest of the basemap line treatment.
        const beforeSymbol = layers.find((l) => l.type === 'symbol')?.id;
        ml.addLayer({
          id: coastlineId,
          type: 'line',
          source: (waterFill as { source: string }).source,
          'source-layer': (waterFill as { 'source-layer': string })['source-layer'],
          paint: { 'line-color': '#ffffff', 'line-opacity': 0.55, 'line-width': 0.6 },
        }, beforeSymbol);
      }
    }

    // Anchor the field beneath the first line/symbol layer so coastlines,
    // country borders, roads and place labels render on top of the field.
    // Falls back to "no anchor" (field on top) if the style exposes none.
    const anchorIdx = layers.findIndex((l) => l.type === 'line' || l.type === 'symbol');
    setOverlayBeforeId(anchorIdx >= 0 ? layers[anchorIdx].id : null);

    // Restyle basemap lines and labels for legibility over the opaque
    // colour field: dark-matter's defaults (light grey on near-black) are
    // invisible once the field renders bright colours underneath. Force
    // white with a dark halo for symbols, white for lines, and keep
    // opacity moderate so they read as overlays rather than dominating.
    const startIdx = anchorIdx >= 0 ? anchorIdx : 0;
    const lineIds: string[] = [];
    const symbolIds: string[] = [];
    for (let i = startIdx; i < layers.length; i++) {
      const l = layers[i];
      if (l.type === 'line') {
        lineIds.push(l.id);
        ml.setPaintProperty(l.id, 'line-color', '#ffffff');
        ml.setPaintProperty(l.id, 'line-opacity', 0.55);
      } else if (l.type === 'symbol') {
        symbolIds.push(l.id);
        ml.setPaintProperty(l.id, 'text-color', '#ffffff');
        ml.setPaintProperty(l.id, 'text-halo-color', 'rgba(0, 0, 0, 0.85)');
        ml.setPaintProperty(l.id, 'text-halo-width', 1.5);
        ml.setPaintProperty(l.id, 'text-opacity', 0.9);
        ml.setPaintProperty(l.id, 'icon-opacity', 0.85);
      }
    }
    if (ml.getLayer('coastline-overlay')) lineIds.push('coastline-overlay');
    basemapLineIdsRef.current = lineIds;
    basemapSymbolIdsRef.current = symbolIds;

    captureViewport();
  }, [captureViewport]);

  const isGlobe = activePreset.id === 'globe';

  const maxZoom = computeMaxZoom(lat?.length ?? null);

  // Apply showLabels / showLines by toggling MapLibre layer visibility.
  // Runs whenever a toggle changes; no-op if handleMapLoad hasn't populated
  // the layer-id caches yet.
  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (!map) return;
    for (const id of basemapLineIdsRef.current) {
      if (!map.getLayer(id)) continue;
      map.setLayoutProperty(id, 'visibility', showLines ? 'visible' : 'none');
    }
    for (const id of basemapSymbolIdsRef.current) {
      if (!map.getLayer(id)) continue;
      map.setLayoutProperty(id, 'visibility', showLabels ? 'visible' : 'none');
    }
  }, [showLabels, showLines, isGlobe]);

  const handlePresetSelect = (preset: ProjectionPreset) => {
    if (preset.id === activePreset.id) return;
    if (activePreset.id === 'flat' && mapRef.current) {
      const centre = mapRef.current.getCenter();
      setCameraCenter({ lat: centre.lat, lon: centre.lng });
    }
    if (preset.id === 'flat') justSwitchedToFlatRef.current = true;
    if (preset.id !== 'flat') setViewportBounds(null);
    setActivePreset(preset);
  };

  const handleCesiumUnmount = (lat: number, lon: number) => {
    setCameraCenter({ lat, lon });
  };

  const handleMapClick = useCallback((e: MapLayerMouseEvent) => {
    if (!onMapClick) return;
    onMapClick({
      lat: e.lngLat.lat,
      lon: e.lngLat.lng,
      screenX: e.point.x,
      screenY: e.point.y,
    });
  }, [onMapClick]);

  useEffect(() => {
    if (!isGlobe && justSwitchedToFlatRef.current && mapRef.current) {
      justSwitchedToFlatRef.current = false;
      mapRef.current.flyTo({ center: [cameraCenter.lon, cameraCenter.lat], zoom: mapRef.current.getZoom(), duration: 0 });
    }
  }, [cameraCenter, isGlobe]);

  // Recompute screen position when the selected point changes (snapping update)
  useEffect(() => {
    if (isGlobe || !selectedPoint || !mapRef.current) return;
    const screen = mapRef.current.project([selectedPoint.lon, selectedPoint.lat]);
    onSelectedPointScreenRef.current?.(screen.x, screen.y);
  }, [selectedPoint, isGlobe]);

  // ── Overlay props ────────────────────────────────────────────────────
  //
  // Two-layer rendering per view: a low-resolution full-globe back layer that
  // is always present, and a screen-resolution front layer covering the
  // visible viewport extended by FRONT_BUFFER_FACTOR. The back's RGBA has
  // alpha forced to zero in the rectangle covered by the front, so every
  // pixel has at most one source contributing colour (no opacity stacking).
  //
  // Both layers display at raster-opacity 0.7 / Color(1,1,1,0.7).

  const bufferedFlatBounds: ViewBounds | null =
    !isGlobe && viewportBounds
      ? bufferBounds(viewportBounds, FRONT_BUFFER_FACTOR, -85, 85, -180, 180)
      : null;

  const bufferedGlobeBounds: ViewBounds | null =
    isGlobe && cesiumBounds
      ? bufferBounds(cesiumBounds, FRONT_BUFFER_FACTOR, -90, 90, -180, 180)
      : null;

  const flatFrontWorthwhile = highResEnabled && !!bufferedFlatBounds
    && (bufferedFlatBounds.east - bufferedFlatBounds.west) < FRONT_SKIP_LON_SPAN_DEG;
  const globeFrontWorthwhile = highResEnabled && !!bufferedGlobeBounds
    && (bufferedGlobeBounds.east - bufferedGlobeBounds.west) < FRONT_SKIP_LON_SPAN_DEG;

  // Front layers — viewport at screen resolution. Skipped when the buffered
  // span is close to the global extent: at that zoom the back already covers
  // the visible area at its full resolution and adding a front layer would
  // only introduce a rasterised-quad seam at its outer edge for no gain.
  const viewportFlatProps: FieldOverlayProps | null =
    !isGlobe && data && lat && lon && bufferedFlatBounds && viewportSize && flatFrontWorthwhile
      ? {
          data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
          renderMode, mapProjection: 'mercator',
          bounds: bufferedFlatBounds,
          viewportWidth: viewportSize.width,
          viewportHeight: viewportSize.height,
        }
      : null;

  const globeProps: FieldOverlayProps | null =
    isGlobe && data && lat && lon && bufferedGlobeBounds && globeFrontWorthwhile
      ? {
          data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
          renderMode, mapProjection: 'geographic',
          bounds: bufferedGlobeBounds,
          viewportWidth: cesiumViewSize.width,
          viewportHeight: cesiumViewSize.height,
        }
      : null;

  // Hooks must always be called in the same order regardless of isGlobe.
  // The two layers are mutually exclusive at display time: when the front
  // image fully contains the visible viewport, only the front is shown
  // (single-layer 0.7 opacity, no boundary, no flashing from re-masking).
  // Otherwise the front is hidden and the back covers the globe at its low
  // resolution. No alpha-mask compositing is performed on either RGBA.
  const { image: globeImage, isRendering: isGlobeFrontRendering } = useFieldImage(globeProps);
  const { image: viewportFlatImage, isRendering: isFlatFrontRendering } = useFieldImage(viewportFlatProps);

  const globalGlobeProps: FieldOverlayProps | null = isGlobe && data && lat && lon ? {
    data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
    renderMode, mapProjection: 'geographic',
    viewportWidth: BACK_GEO_W, viewportHeight: BACK_GEO_H,
  } : null;

  const globalFlatProps: FieldOverlayProps | null = !isGlobe && data && lat && lon ? {
    data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
    renderMode, mapProjection: 'mercator',
    viewportWidth: BACK_FLAT_W, viewportHeight: BACK_FLAT_H,
  } : null;

  const { image: globalGlobeImage, isRendering: isGlobeBackRendering } = useFieldImage(globalGlobeProps);
  const { image: globalFlatImage, isRendering: isFlatBackRendering } = useFieldImage(globalFlatProps);

  const backFlatImage = globalFlatImage;
  const frontFlatImage = viewportFlatImage;
  const backGlobeImage = globalGlobeImage;
  const frontGlobeImage = globeImage;

  const showSpinner = isLoading
    || isGlobeBackRendering || isFlatBackRendering
    || isGlobeFrontRendering || isFlatFrontRendering;

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {isGlobe ? (
        <CesiumView
          fieldImage={frontGlobeImage}
          backFieldImage={backGlobeImage}
          showLabels={showLabels}
          showLines={showLines}
          initialCenter={cameraCenter}
          onUnmount={handleCesiumUnmount}
          onViewChange={(b, w, h) => { setCesiumBounds(b); setCesiumViewSize({ width: w, height: h }); }}
          onMapClick={onMapClick}
          selectedPoint={selectedPoint}
          selectedPointGridSpacing={selectedPointGridSpacing}
          onSelectedPointScreen={onSelectedPointScreen}
          onSelectedPointOutOfView={onSelectedPointOutOfView}
        />
      ) : (
        <Map
          ref={mapRef}
          initialViewState={{ longitude: cameraCenter.lon, latitude: cameraCenter.lat, zoom: 1.5 }}
          maxZoom={maxZoom}
          mapStyle={BASEMAP_STYLE}
          projection={activePreset.type as any}
          style={{ width: '100%', height: '100%' }}
          onLoad={handleMapLoad}
          onMoveEnd={captureViewport}
          onZoomEnd={captureViewport}
          onClick={handleMapClick}
        >
          {backFlatImage && (
            <Source id="field-overlay-back" type="image" url={backFlatImage.dataUrl} coordinates={backFlatImage.coordinates}>
              <Layer
                id="field-overlay-back-layer"
                type="raster"
                beforeId={overlayBeforeId ?? undefined}
                paint={{
                  'raster-opacity': 1,
                  'raster-fade-duration': 0,
                  'raster-resampling': 'nearest',
                }}
              />
            </Source>
          )}
          {frontFlatImage && (
            <Source id="field-overlay-front" type="image" url={frontFlatImage.dataUrl} coordinates={frontFlatImage.coordinates}>
              <Layer
                id="field-overlay-front-layer"
                type="raster"
                beforeId={overlayBeforeId ?? undefined}
                paint={{
                  'raster-opacity': 1,
                  'raster-fade-duration': 0,
                  'raster-resampling': 'nearest',
                }}
              />
            </Source>
          )}
          {selectedPoint && (() => {
            const spacing = selectedPointGridSpacing ?? 2;
            const baseRadius = spacing * 0.356;
            return (
              <Source
                id="selected-point"
                type="geojson"
                data={{ type: 'Feature', geometry: { type: 'Point', coordinates: [selectedPoint.lon, selectedPoint.lat] }, properties: {} }}
              >
                <Layer
                  id="selected-point-circle"
                  type="circle"
                  paint={{
                    'circle-radius': ['interpolate', ['exponential', 2], ['zoom'], 0, Math.max(baseRadius, 2), 10, Math.max(baseRadius * 1024, 4)],
                    'circle-color': 'transparent',
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff',
                    'circle-stroke-opacity': 0.9,
                  }}
                />
              </Source>
            );
          })()}
        </Map>
      )}

      <div className="map-controls-bar">
        <ProjectionPicker current={activePreset.id} onSelect={handlePresetSelect} />
        {data && <RenderModePicker mode={renderMode} onChange={setRenderMode} />}
        {data && (
          <div className="map-config-pill">
            <button
              className={`map-picker-btn${configOpen ? ' map-picker-btn-active' : ''}`}
              onClick={() => setConfigOpen((o) => !o)}
              title="Display options"
              aria-expanded={configOpen}
            >
              Display
              <span className="map-config-chevron" aria-hidden>{configOpen ? '▾' : '▸'}</span>
            </button>
            {configOpen && (
              <div className="map-config-panel">
                <label className="map-config-row">
                  <input
                    type="checkbox"
                    checked={highResEnabled}
                    onChange={(e) => setHighResEnabled(e.target.checked)}
                  />
                  <span>High-res viewport</span>
                  <span className="map-config-hint">expensive when zoomed in</span>
                </label>
                <label className="map-config-row">
                  <input
                    type="checkbox"
                    checked={showLabels}
                    onChange={(e) => setShowLabels(e.target.checked)}
                  />
                  <span>Place labels</span>
                </label>
                <label className="map-config-row">
                  <input
                    type="checkbox"
                    checked={showLines}
                    onChange={(e) => setShowLines(e.target.checked)}
                  />
                  <span>Borders &amp; coastlines</span>
                </label>
              </div>
            )}
          </div>
        )}
      </div>

      {data && (
        <>
          <div style={{ position: 'absolute', bottom: 'calc(var(--sheet-height, 32px) + 8px)', right: 16, zIndex: 10 }}>
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
              initialMinimised={colorScaleInitialMinimised}
            />
          </div>
        </>
      )}
      {showSpinner && <div className="map-frame-spinner" />}
    </div>
  );
}

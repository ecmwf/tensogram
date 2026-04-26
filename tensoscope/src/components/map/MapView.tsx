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
  const [cameraCenter, setCameraCenter] = useState({ lat: 20, lon: 0 });
  const mapRef = useRef<MapRef>(null);
  const justSwitchedToFlatRef = useRef(false);

  // Flat map viewport state
  const [viewportBounds, setViewportBounds] = useState<ViewBounds | null>(null);
  const [viewportSize, setViewportSize] = useState<{ width: number; height: number } | null>(null);

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

  const isGlobe = activePreset.id === 'globe';

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

  const flatFrontWorthwhile = bufferedFlatBounds
    ? (bufferedFlatBounds.east - bufferedFlatBounds.west) < FRONT_SKIP_LON_SPAN_DEG
    : false;
  const globeFrontWorthwhile = bufferedGlobeBounds
    ? (bufferedGlobeBounds.east - bufferedGlobeBounds.west) < FRONT_SKIP_LON_SPAN_DEG
    : false;

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
          mapStyle={BASEMAP_STYLE}
          projection={activePreset.type as any}
          style={{ width: '100%', height: '100%' }}
          onLoad={captureViewport}
          onMoveEnd={captureViewport}
          onZoomEnd={captureViewport}
          onClick={handleMapClick}
        >
          {backFlatImage && (
            <Source id="field-overlay-back" type="image" url={backFlatImage.dataUrl} coordinates={backFlatImage.coordinates}>
              <Layer
                id="field-overlay-back-layer"
                type="raster"
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

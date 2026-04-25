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
  // Single-layer rendering: one image shown at a time per view mode.
  //
  // Globe: viewport-aware render (cesiumBounds) when the camera position is
  //   known, falling back to a full-extent 2048×1024 render on first load.
  //   A separate full-extent render is kept as a low-res fallback shown via
  //   the ?? operator so the globe is never blank.
  //
  // Flat: screen-resolution render of the visible viewport when bounds are
  //   known, falling back to a full-extent 1440×720 render otherwise.
  //   Only one MapLibre source/layer is active at any time.

  // High-res globe viewport render disabled:
  // const globeProps: FieldOverlayProps | null = isGlobe && data && lat && lon ? {
  //   data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
  //   renderMode, mapProjection: 'geographic',
  //   ...(cesiumBounds
  //     ? { bounds: cesiumBounds, viewportWidth: cesiumViewSize.width, viewportHeight: cesiumViewSize.height }
  //     : { viewportWidth: 2048, viewportHeight: 1024 }),
  // } : null;
  const globeProps: FieldOverlayProps | null = null;

  const globalGlobeProps: FieldOverlayProps | null = isGlobe && data && lat && lon ? {
    data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
    renderMode, mapProjection: 'geographic',
    viewportWidth: 2048, viewportHeight: 1024,
  } : null;

  const globalFlatProps: FieldOverlayProps | null = !isGlobe && data && lat && lon ? {
    data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
    renderMode, mapProjection: 'mercator',
    viewportWidth: 1440, viewportHeight: 720,
  } : null;

  // High-res flat viewport render disabled:
  // const viewportFlatProps: FieldOverlayProps | null =
  //   !isGlobe && data && lat && lon && viewportBounds && viewportSize ? {
  //     data, lat, lon, colorMin, colorMax, palette, paletteReversed, customStops,
  //     renderMode, mapProjection: 'mercator',
  //     bounds: viewportBounds,
  //     viewportWidth: viewportSize.width,
  //     viewportHeight: viewportSize.height,
  //   } : null;
  const viewportFlatProps: FieldOverlayProps | null = null;

  // Hooks must always be called in the same order regardless of isGlobe.
  const { image: globeImage } = useFieldImage(globeProps);
  const { image: globalGlobeImage, isRendering: isGlobeRendering } = useFieldImage(globalGlobeProps);
  const { image: globalFlatImage, isRendering: isFlatRendering } = useFieldImage(globalFlatProps);
  const { image: viewportFlatImage } = useFieldImage(viewportFlatProps);

  // High-res viewport rendering disabled; re-enable by restoring:
  //   const flatImage = viewportFlatImage ?? globalFlatImage;
  //   const cesiumImage = globeImage ?? globalGlobeImage;
  const flatImage = globalFlatImage;
  const cesiumImage = globalGlobeImage;

  const showSpinner = isLoading || isGlobeRendering || isFlatRendering;

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {isGlobe ? (
        <CesiumView
          fieldImage={cesiumImage}
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
          {flatImage && (
            <Source id="field-overlay" type="image" url={flatImage.dataUrl} coordinates={flatImage.coordinates}>
              <Layer id="field-overlay-layer" type="raster" paint={{ 'raster-opacity': 0.7, 'raster-fade-duration': 0 }} />
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

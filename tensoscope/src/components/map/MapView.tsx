import { useState, useCallback, useRef, useEffect } from 'react';
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
  const justSwitchedToFlatRef = useRef(false);

  const handlePresetSelect = (preset: ProjectionPreset) => {
    if (preset.id === activePreset.id) return;
    if (activePreset.id === 'flat' && mapRef.current) {
      const centre = mapRef.current.getCenter();
      setCameraCenter({ lat: centre.lat, lon: centre.lng });
    }
    if (preset.id === 'flat') {
      justSwitchedToFlatRef.current = true;
    }
    setActivePreset(preset);
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

  const isGlobe = activePreset.id === 'globe';

  // After Cesium unmounts and saves its camera position, fly MapLibre to that position.
  // initialViewState on <Map> is mount-only, so we must use flyTo after the switch.
  useEffect(() => {
    if (!isGlobe && justSwitchedToFlatRef.current && mapRef.current) {
      justSwitchedToFlatRef.current = false;
      mapRef.current.flyTo({ center: [cameraCenter.lon, cameraCenter.lat], zoom, duration: 0 });
    }
  }, [cameraCenter, isGlobe, zoom]);

  const overlayProps: FieldOverlayProps | null =
    data && lat && lon
      ? { data, lat, lon, colorMin, colorMax, palette, zoom, paletteReversed, customStops, renderMode, mapProjection: isGlobe ? 'geographic' : 'mercator' }
      : null;

  const fieldImage = useFieldImage(overlayProps);

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
                paint={{ 'raster-opacity': 0.7, 'raster-fade-duration': 0 }}
              />
            </Source>
          )}
        </Map>
      )}

      <div className="map-controls-bar">
        <ProjectionPicker current={activePreset.id} onSelect={handlePresetSelect} />
        {data && <RenderModePicker mode={renderMode} onChange={setRenderMode} />}
      </div>

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

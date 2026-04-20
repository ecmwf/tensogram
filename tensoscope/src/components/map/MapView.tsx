/**
 * MapView -- main map container.
 *
 * Uses react-map-gl/maplibre with a native image source for the field overlay.
 * This renders as part of the map so alignment and projection are handled by
 * MapLibre itself.
 */

import { useState, useCallback, useRef } from 'react';
import { Map, Source, Layer } from 'react-map-gl/maplibre';
import type { MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useFieldImage } from './FieldOverlay';
import { ColorBar } from './ColorBar';
import { ColorScaleControls } from './ColorScaleControls';
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
  const [zoom, setZoom] = useState(1.5);
  const mapRef = useRef<MapRef>(null);
  const zoomTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handlePresetSelect = (preset: ProjectionPreset) => {
    setActivePreset(preset);
    mapRef.current?.flyTo({ center: preset.center, zoom: preset.zoom });
  };

  // Debounce zoom changes to avoid spamming re-renders during smooth zoom
  const handleZoomEnd = useCallback(() => {
    if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current);
    zoomTimerRef.current = setTimeout(() => {
      const map = mapRef.current;
      if (map) setZoom(map.getZoom());
    }, 200);
  }, []);

  const overlayProps: FieldOverlayProps | null =
    data && lat && lon
      ? { data, lat, lon, colorMin, colorMax, palette, zoom, paletteReversed, customStops }
      : null;

  const fieldImage = useFieldImage(overlayProps);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Map
        ref={mapRef}
        initialViewState={{ longitude: 0, latitude: 20, zoom: 1.5 }}
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

      <ProjectionPicker current={activePreset.id} onSelect={handlePresetSelect} />

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

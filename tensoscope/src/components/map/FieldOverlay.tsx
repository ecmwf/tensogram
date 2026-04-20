/**
 * FieldOverlay -- renders a colormapped unstructured field via MapLibre's
 * native image source.
 *
 * MapLibre interpolates image sources in Web Mercator projection space,
 * so we regrid into Mercator y-spacing for correct geographic alignment.
 *
 * Heavy computation (regridding + colormapping) runs in a Web Worker to
 * keep the main thread responsive. Results are cached by an LRU so that
 * revisiting a previously rendered field/zoom combination is instant.
 *
 * Grid resolution adapts to both the source data density and the current
 * map zoom level: coarser when zoomed out, sharper when zoomed in.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { getPaletteLUT } from './colormaps';
import type { CustomStop } from './colormaps';
import RegridWorker from './regrid.worker?worker';
import { getNumBands } from './contourUtils';

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
  mapProjection?: 'mercator' | 'geographic';
}

export interface FieldImage {
  dataUrl: string;
  coordinates: [[number, number], [number, number], [number, number], [number, number]];
}

const LAT_MAX = 85;
const LAT_MIN = -85;

const COORDINATES: FieldImage['coordinates'] = [
  [-180, LAT_MAX],
  [180, LAT_MAX],
  [180, LAT_MIN],
  [-180, LAT_MIN],
];

// ── Mercator helpers ──────────────────────────────────────────────────

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function latToMercY(lat: number): number {
  return Math.log(Math.tan(Math.PI / 4 + lat * DEG2RAD / 2));
}

function mercYToLat(y: number): number {
  return (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * RAD2DEG;
}

const Y_TOP = latToMercY(LAT_MAX);
const Y_RANGE = Y_TOP - latToMercY(LAT_MIN);

// ── Adaptive grid parameters ──────────────────────────────────────────

interface GridParams {
  width: number;
  height: number;
  binDeg: number;
  rowLats: Float64Array;
}

/**
 * Compute output grid dimensions.
 *
 * `scaleFactor` ranges from ~0.3 (zoomed out) to 1.0 (zoomed in enough
 * to see full native resolution). It multiplies the native-resolution
 * grid size so that zoomed-out views render smaller, faster images.
 */
function computeGridParams(
  nPoints: number,
  scaleFactor: number,
  mapProjection: 'mercator' | 'geographic',
): GridParams {
  const avgSpacing = Math.sqrt(360 * 170 / Math.max(1, nPoints));
  const cellDeg = Math.max(0.1, Math.min(2, avgSpacing * 0.7));
  const nativeWidth = Math.ceil(360 / cellDeg);

  const width = Math.min(1440, Math.max(180, Math.round(nativeWidth * scaleFactor)));
  const heightRaw = mapProjection === 'geographic'
    ? width * (LAT_MAX - LAT_MIN) / 360
    : width * Y_RANGE / (2 * Math.PI);
  const height = Math.min(1440, Math.max(180, Math.round(heightRaw)));
  const binDeg = Math.max(1, Math.ceil(avgSpacing * 1.5));

  const rowLats = new Float64Array(height);
  if (mapProjection === 'geographic') {
    for (let row = 0; row < height; row++) {
      rowLats[row] = LAT_MAX - (row + 0.5) / height * (LAT_MAX - LAT_MIN);
    }
  } else {
    for (let row = 0; row < height; row++) {
      const mercY = Y_TOP - (row + 0.5) / height * Y_RANGE;
      rowLats[row] = mercYToLat(mercY);
    }
  }

  return { width, height, binDeg, rowLats };
}

/**
 * Map zoom level to a grid scale factor.
 * zoom 0-1  -> 0.3 (very coarse)
 * zoom 2-3  -> 0.5
 * zoom 4+   -> 1.0 (full native res)
 */
function zoomToScale(zoom: number | undefined): number {
  if (zoom == null) return 0.6; // sensible default
  if (zoom <= 1) return 0.3;
  if (zoom >= 4) return 1.0;
  // Linear interpolation between zoom 1 (0.3) and zoom 4 (1.0)
  return 0.3 + (zoom - 1) * (0.7 / 3);
}

// ── Image cache ──────────────────────────────────────────────────────

interface CacheEntry {
  dataUrl: string;
  key: string;
}

const IMAGE_CACHE_SIZE = 12;
const imageCache: CacheEntry[] = [];

/** Cheap fingerprint: length + a few sampled values to distinguish different fields of equal size. */
function dataFingerprint(data: Float32Array): string {
  const n = data.length;
  if (n === 0) return '0';
  // Sample 4 evenly spaced values
  const a = data[0];
  const b = data[n >> 2];
  const c = data[n >> 1];
  const d = data[n - 1];
  return `${n}:${a}:${b}:${c}:${d}`;
}

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
  mapProjection: 'mercator' | 'geographic',
): string {
  const palKey = palette
    + (paletteReversed ? ':r' : '')
    + (palette === 'custom' && customStops ? ':' + JSON.stringify(customStops) : '');
  return `${dataFingerprint(data)}:${colorMin}:${colorMax}:${palKey}:${gridW}x${gridH}:${renderMode}:${mapProjection}`;
}

function getCached(key: string): string | null {
  const idx = imageCache.findIndex((e) => e.key === key);
  if (idx < 0) return null;
  // Move to front (most recently used)
  const [entry] = imageCache.splice(idx, 1);
  imageCache.unshift(entry);
  return entry.dataUrl;
}

function putCache(key: string, dataUrl: string): void {
  // Evict oldest if full
  if (imageCache.length >= IMAGE_CACHE_SIZE) imageCache.pop();
  imageCache.unshift({ key, dataUrl });
}

// ── Shared worker instance ───────────────────────────────────────────

let sharedWorker: Worker | null = null;
let workerSeq = 0;
const pendingCallbacks = new Map<number, (rgba: Uint8ClampedArray, w: number, h: number) => void>();

function getWorker(): Worker {
  if (!sharedWorker) {
    sharedWorker = new RegridWorker();
    sharedWorker.onmessage = (e: MessageEvent) => {
      const { id, rgba, width, height } = e.data;
      const cb = pendingCallbacks.get(id);
      if (cb) {
        pendingCallbacks.delete(id);
        cb(rgba, width, height);
      }
    };
  }
  return sharedWorker;
}

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
  return new Promise((resolve) => {
    const id = ++workerSeq;
    pendingCallbacks.set(id, (rgba, w, h) => resolve({ rgba, width: w, height: h }));
    const worker = getWorker();
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
      // Transfer arrays to avoid copying (worker gets ownership)
      // We clone them first since the caller may still need them
    );
  });
}

// ── Canvas helper ────────────────────────────────────────────────────

function rgbaToDataUrl(rgba: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = new ImageData(width, height);
  imageData.data.set(rgba);
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}

// ── React hook ────────────────────────────────────────────────────────

export function useFieldImage(props: FieldOverlayProps | null): FieldImage | null {
  const [image, setImage] = useState<FieldImage | null>(null);
  const requestIdRef = useRef(0);

  const render = useCallback(async () => {
    if (!props || !props.data || props.data.length === 0) {
      setImage(null);
      return;
    }
    const { data, lat, lon, colorMin, colorMax, palette, zoom, paletteReversed = false, customStops, renderMode, mapProjection = 'mercator' } = props;
    const numBands = getNumBands(palette, customStops);

    const scale = zoomToScale(zoom);
    const params = computeGridParams(data.length, scale, mapProjection);
    const key = cacheKey(data, colorMin, colorMax, palette, params.width, params.height, paletteReversed, customStops, renderMode, mapProjection);

    // Check cache first
    const cached = getCached(key);
    if (cached) {
      setImage({ dataUrl: cached, coordinates: COORDINATES });
      return;
    }

    const reqId = ++requestIdRef.current;
    const lut = getPaletteLUT(palette, { reversed: paletteReversed, customStops });

    const result = await requestRegrid(lat, lon, data, lut, colorMin, colorMax, params, renderMode, numBands);

    // Only apply if this is still the most recent request (prevents stale results)
    if (reqId !== requestIdRef.current) return;

    const dataUrl = rgbaToDataUrl(result.rgba, result.width, result.height);
    putCache(key, dataUrl);
    setImage({ dataUrl, coordinates: COORDINATES });
  }, [props?.data, props?.lat, props?.lon, props?.colorMin, props?.colorMax, props?.palette, props?.zoom, props?.paletteReversed, props?.customStops, props?.renderMode, props?.mapProjection]);

  useEffect(() => {
    render();
  }, [render]);

  return image;
}

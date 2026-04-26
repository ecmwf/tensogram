/**
 * FieldOverlay -- renders a colormapped unstructured field via MapLibre's
 * native image source.
 *
 * Heavy computation (regridding + colormapping) runs in a Web Worker to
 * keep the main thread responsive. Results are cached by an LRU so that
 * revisiting a previously rendered field/zoom combination is instant.
 *
 * When viewport bounds and dimensions are supplied the grid covers only the
 * visible area at screen resolution, giving sharp rendering at any zoom
 * level. Without bounds it falls back to a full-globe render (used by Cesium
 * and as a background fallback on the flat map).
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { getPaletteLUT } from './colormaps';
import type { CustomStop } from './colormaps';
import RegridWorker from './regrid.worker?worker';
import { getNumBands } from './contourUtils';

export interface ViewBounds {
  west: number;
  east: number;
  south: number;
  north: number;
}

export interface FieldOverlayProps {
  data: Float32Array;
  lat: Float32Array;
  lon: Float32Array;
  colorMin: number;
  colorMax: number;
  palette: string;
  paletteReversed?: boolean;
  customStops?: CustomStop[];
  renderMode: 'heatmap' | 'contours';
  mapProjection?: 'mercator' | 'geographic';
  viewportWidth?: number;
  viewportHeight?: number;
  bounds?: ViewBounds;
  excludeBounds?: ViewBounds | null;
}

export interface FieldImage {
  dataUrl: string;
  coordinates: [[number, number], [number, number], [number, number], [number, number]];
}

const LAT_MAX_MERCATOR = 85;
const LAT_MIN_MERCATOR = -85;
const LAT_MAX_GEOGRAPHIC = 90;
const LAT_MIN_GEOGRAPHIC = -90;

// ── Mercator helpers ──────────────────────────────────────────────────

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function latToMercY(lat: number): number {
  return Math.log(Math.tan(Math.PI / 4 + lat * DEG2RAD / 2));
}

function mercYToLat(y: number): number {
  return (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * RAD2DEG;
}

// ── Adaptive grid parameters ──────────────────────────────────────────

interface GridParams {
  width: number;
  height: number;
  binDeg: number;
  rowLats: Float64Array;
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
  mapProjection: 'mercator' | 'geographic';
}

function computeGridParams(
  nPoints: number,
  mapProjection: 'mercator' | 'geographic',
  lonMin: number,
  lonMax: number,
  latMin: number,
  latMax: number,
  viewportWidth: number,
  viewportHeight: number,
): GridParams {
  const avgSpacing = Math.sqrt(360 * 170 / Math.max(1, nPoints));
  const binDeg = Math.max(1, Math.ceil(avgSpacing * 1.5));

  const width = Math.min(4096, Math.max(64, viewportWidth));
  const height = Math.min(4096, Math.max(64, viewportHeight));

  const latMax_ = mapProjection === 'geographic' ? LAT_MAX_GEOGRAPHIC : LAT_MAX_MERCATOR;
  const latMin_ = mapProjection === 'geographic' ? LAT_MIN_GEOGRAPHIC : LAT_MIN_MERCATOR;
  const clampedLatMax = Math.min(latMax_, latMax);
  const clampedLatMin = Math.max(latMin_, latMin);

  const rowLats = new Float64Array(height);
  if (mapProjection === 'geographic') {
    for (let row = 0; row < height; row++) {
      rowLats[row] = clampedLatMax - (row + 0.5) / height * (clampedLatMax - clampedLatMin);
    }
  } else {
    const yTop = latToMercY(clampedLatMax);
    const yBot = latToMercY(clampedLatMin);
    const yRange = yTop - yBot;
    for (let row = 0; row < height; row++) {
      const mercY = yTop - (row + 0.5) / height * yRange;
      rowLats[row] = mercYToLat(mercY);
    }
  }

  return { width, height, binDeg, rowLats, lonMin, lonMax, latMin: clampedLatMin, latMax: clampedLatMax, mapProjection };
}

// ── Image cache ──────────────────────────────────────────────────────
//
// Stores raw RGBA + grid params, not dataUrls. The dataUrl is regenerated
// on every render (~5 ms for 2K×1K, negligible compared to the worker
// pass) so the exclude-mask can be re-applied cheaply when excludeBounds
// changes without re-running the worker.

interface CacheEntry {
  key: string;
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  params: GridParams;
}

const IMAGE_CACHE_SIZE = 64;
const imageCache: CacheEntry[] = [];

function dataFingerprint(data: Float32Array): string {
  const n = data.length;
  if (n === 0) return '0';
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
  lonMin: number,
  lonMax: number,
  latMin: number,
  latMax: number,
  paletteReversed: boolean,
  customStops: CustomStop[] | undefined,
  renderMode: 'heatmap' | 'contours',
  mapProjection: 'mercator' | 'geographic',
): string {
  const palKey = palette
    + (paletteReversed ? ':r' : '')
    + (palette === 'custom' && customStops ? ':' + JSON.stringify(customStops) : '');
  // Quantise bounds to 0.5° so minor pan jitter doesn't bust the cache
  const q = (v: number) => (Math.round(v * 2) / 2).toFixed(1);
  return `${dataFingerprint(data)}:${colorMin}:${colorMax}:${palKey}:${gridW}x${gridH}:${q(lonMin)},${q(lonMax)},${q(latMin)},${q(latMax)}:${renderMode}:${mapProjection}`;
}

function getCached(key: string): CacheEntry | null {
  const idx = imageCache.findIndex((e) => e.key === key);
  if (idx < 0) return null;
  const [entry] = imageCache.splice(idx, 1);
  imageCache.unshift(entry);
  return entry;
}

function putCache(entry: CacheEntry): void {
  if (imageCache.length >= IMAGE_CACHE_SIZE) imageCache.pop();
  imageCache.unshift(entry);
}

// Exposed for tests.
export const __test = {
  applyExcludeMask,
  resolveBounds,
  getCached,
  putCache,
  clearCache: () => { imageCache.length = 0; },
};

// ── Exclude-region masking ───────────────────────────────────────────
//
// Used by the global (full-extent) render to zero-out pixels covered by the
// viewport-specific render so both layers read at the same opacity (0.7)
// with no additive stacking in the overlap region.

function applyExcludeMask(
  rgba: Uint8ClampedArray,
  params: GridParams,
  excl: ViewBounds,
): Uint8ClampedArray {
  const out = rgba.slice();
  const { width, height, lonMin, lonMax, latMin, latMax, mapProjection } = params;
  const lonSpan = lonMax - lonMin;

  const x0 = Math.max(0, Math.floor((excl.west  - lonMin) / lonSpan * width));
  const x1 = Math.min(width, Math.ceil((excl.east  - lonMin) / lonSpan * width));

  let y0: number, y1: number;
  if (mapProjection === 'geographic') {
    const latSpan = latMax - latMin;
    y0 = Math.max(0, Math.floor((latMax - Math.min(latMax, excl.north)) / latSpan * height));
    y1 = Math.min(height, Math.ceil((latMax - Math.max(latMin, excl.south)) / latSpan * height));
  } else {
    const yTop = latToMercY(latMax);
    const yBot = latToMercY(latMin);
    const yRange = yTop - yBot;
    y0 = Math.max(0, Math.floor((yTop - latToMercY(Math.min(latMax, excl.north))) / yRange * height));
    y1 = Math.min(height, Math.ceil((yTop - latToMercY(Math.max(latMin, excl.south))) / yRange * height));
  }

  for (let row = y0; row < y1; row++) {
    const base = row * width * 4;
    for (let col = x0; col < x1; col++) {
      out[base + col * 4 + 3] = 0;
    }
  }
  return out;
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
    worker.postMessage({
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
      lonMin: params.lonMin,
      lonMax: params.lonMax,
      renderMode,
      numBands,
    });
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

// ── Bounds resolution ─────────────────────────────────────────────────

/**
 * Resolve the effective rendering window.
 *
 * When bounds cross the antimeridian (east > 180 or west < -180) or span
 * more than 330° in longitude, fall back to the full global extent so there
 * are no gaps near the dateline.
 */
function resolveBounds(
  mapProjection: 'mercator' | 'geographic',
  bounds: ViewBounds | undefined,
  viewportWidth: number | undefined,
  viewportHeight: number | undefined,
): { lonMin: number; lonMax: number; latMin: number; latMax: number; vw: number; vh: number } {
  const latMax_ = mapProjection === 'geographic' ? LAT_MAX_GEOGRAPHIC : LAT_MAX_MERCATOR;
  const latMin_ = mapProjection === 'geographic' ? LAT_MIN_GEOGRAPHIC : LAT_MIN_MERCATOR;

  const globalVw = mapProjection === 'geographic' ? 2048 : 1440;
  const globalVh = mapProjection === 'geographic' ? 1024 : 720;

  if (!bounds) {
    return { lonMin: -180, lonMax: 180, latMin: latMin_, latMax: latMax_, vw: viewportWidth ?? globalVw, vh: viewportHeight ?? globalVh };
  }

  const { west, east, south, north } = bounds;
  const lonSpan = east - west;

  // Antimeridian crossing or nearly-global view: use full extent
  if (west < -180 || east > 180 || lonSpan > 330) {
    return { lonMin: -180, lonMax: 180, latMin: latMin_, latMax: latMax_, vw: viewportWidth ?? globalVw, vh: viewportHeight ?? globalVh };
  }

  return {
    lonMin: Math.max(-180, west),
    lonMax: Math.min(180, east),
    latMin: Math.max(latMin_, south),
    latMax: Math.min(latMax_, north),
    vw: viewportWidth ?? globalVw,
    vh: viewportHeight ?? globalVh,
  };
}

// ── React hook ────────────────────────────────────────────────────────

interface RawRender {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  coords: FieldImage['coordinates'];
  params: GridParams;
}

export function useFieldImage(props: FieldOverlayProps | null): { image: FieldImage | null; isRendering: boolean } {
  const [image, setImage] = useState<FieldImage | null>(null);
  const [isRendering, setIsRendering] = useState(false);
  const requestIdRef = useRef(0);
  // Stores the last unmasked render result so the exclude mask can be
  // re-applied cheaply when the viewport bounds change without re-running
  // the worker.
  const rawRef = useRef<RawRender | null>(null);

  const applyAndSet = useCallback((raw: RawRender, excl: ViewBounds | null | undefined) => {
    const rgba = excl ? applyExcludeMask(raw.rgba, raw.params, excl) : raw.rgba;
    setImage({ dataUrl: rgbaToDataUrl(rgba, raw.width, raw.height), coordinates: raw.coords });
  }, []);

  const render = useCallback(async () => {
    if (!props || !props.data || props.data.length === 0) {
      setImage(null);
      setIsRendering(false);
      rawRef.current = null;
      return;
    }
    const {
      data, lat, lon, colorMin, colorMax, palette,
      paletteReversed = false, customStops, renderMode,
      mapProjection = 'mercator', excludeBounds,
    } = props;
    const numBands = getNumBands(palette, customStops);

    const { lonMin, lonMax, latMin, latMax, vw, vh } = resolveBounds(
      mapProjection, props.bounds, props.viewportWidth, props.viewportHeight,
    );

    const params = computeGridParams(data.length, mapProjection, lonMin, lonMax, latMin, latMax, vw, vh);
    const key = cacheKey(
      data, colorMin, colorMax, palette,
      params.width, params.height,
      lonMin, lonMax, latMin, latMax,
      paletteReversed, customStops, renderMode, mapProjection,
    );

    const coords: FieldImage['coordinates'] = [
      [lonMin, params.latMax],
      [lonMax, params.latMax],
      [lonMax, params.latMin],
      [lonMin, params.latMin],
    ];

    const cached = getCached(key);
    if (cached) {
      const raw: RawRender = {
        rgba: cached.rgba, width: cached.width, height: cached.height,
        coords, params: cached.params,
      };
      rawRef.current = raw;
      setIsRendering(false);
      applyAndSet(raw, excludeBounds);
      return;
    }

    const reqId = ++requestIdRef.current;
    setIsRendering(true);
    // Do not clear image state here -- keep previous image visible while the
    // new render runs (double-buffer: no blank frame on bounds/zoom change).
    const lut = getPaletteLUT(palette, { reversed: paletteReversed, customStops });

    const result = await requestRegrid(lat, lon, data, lut, colorMin, colorMax, params, renderMode, numBands);
    if (reqId !== requestIdRef.current) return;

    setIsRendering(false);
    const raw: RawRender = { rgba: result.rgba, width: result.width, height: result.height, coords, params };
    rawRef.current = raw;
    putCache({ key, rgba: result.rgba, width: result.width, height: result.height, params });
    applyAndSet(raw, excludeBounds);
  }, [
    props?.data, props?.lat, props?.lon,
    props?.colorMin, props?.colorMax, props?.palette,
    props?.paletteReversed, props?.customStops, props?.renderMode, props?.mapProjection,
    props?.viewportWidth, props?.viewportHeight,
    props?.bounds?.west, props?.bounds?.east, props?.bounds?.south, props?.bounds?.north,
    applyAndSet,
  ]);

  useEffect(() => { render(); }, [render]);

  // Re-mask the last render when excludeBounds changes (e.g. user pans the
  // viewport) without re-running the expensive worker.
  const excl = props?.excludeBounds;
  useEffect(() => {
    if (!rawRef.current) return;
    applyAndSet(rawRef.current, excl ?? null);
  }, [excl?.west, excl?.east, excl?.south, excl?.north, applyAndSet]);

  return { image, isRendering };
}

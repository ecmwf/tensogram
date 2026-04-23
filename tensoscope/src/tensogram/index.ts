/**
 * Tensogram wrapper for the viewer.
 *
 * Uses @ecmwf.int/tensogram (WASM) directly in the browser -- no backend needed.
 * Provides file opening, metadata indexing, field decoding, and coordinate
 * extraction.
 */

import {
  init as tgInit,
  TensogramFile,
} from '@ecmwf.int/tensogram';
import type {
  BaseEntry,
  CborValue,
} from '@ecmwf.int/tensogram';

/**
 * Cap on the per-batch prefetch size.  We chunk through the message
 * indices so a 10 000-message file doesn't try to start 10 000
 * Promises at once even though the per-host concurrency limiter
 * would only let 6 of them run.
 */
const PREFETCH_CHUNK_SIZE = 64;
/** Bounded-concurrency cap for prefetch and per-object fetches. */
const REMOTE_CONCURRENCY = 6;

// ── Public types ────────────────────────────────────────────────────────

export interface ObjectInfo {
  msgIndex: number;
  objIndex: number;
  name: string;
  shape: number[];
  dtype: string;
  encoding: string;
  compression: string;
  metadata: Record<string, unknown>;
}

export interface CoordinateInfo {
  msgIndex: number;
  objIndex: number;
  name: string;
  shape: number[];
  dtype: string;
}

export interface FileIndex {
  source: string;
  messageCount: number;
  variables: ObjectInfo[];
  coordinates: CoordinateInfo[];
}

export interface CoordinateData {
  lat: Float32Array;
  lon: Float32Array;
}

export interface FieldStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface DecodedField {
  data: Float32Array;
  shape: number[];
  stats: FieldStats;
}

// ── Coordinate detection ────────────────────────────────────────────────

const LAT_NAMES = new Set(['latitude', 'lat', 'grid_latitude']);
const LON_NAMES = new Set(['longitude', 'lon', 'grid_longitude']);
const COORD_NAMES = new Set([
  ...LAT_NAMES, ...LON_NAMES,
  'level', 'levels', 'pressure', 'time', 'step',
]);

// ── WASM init ───────────────────────────────────────────────────────────

let initialised = false;

export async function ensureInit(): Promise<void> {
  if (initialised) return;
  await tgInit();
  initialised = true;
}


// ── Viewer handle ───────────────────────────────────────────────────────

export class Tensoscope {
  private file: TensogramFile;
  private _source: string;
  private _index: FileIndex | null = null;
  /**
   * Per-message coordinate cache.  Heterogeneous multi-message files
   * may carry different grids in different messages; keying by
   * `msgIdx` lets each call to {@link fetchCoordinates} return the
   * coordinates for that specific message without silently reusing
   * message 0's geometry.
   */
  private readonly _coords = new Map<number, CoordinateData>();

  private constructor(file: TensogramFile, source: string) {
    this.file = file;
    this._source = source;
  }

  /** Open from a browser File object (file picker / drag-drop). */
  static async fromFile(fileObj: File): Promise<Tensoscope> {
    await ensureInit();
    const buf = await fileObj.arrayBuffer();
    const file = TensogramFile.fromBytes(new Uint8Array(buf));
    return new Tensoscope(file, fileObj.name);
  }

  /**
   * Open from a URL (http, https).
   *
   * Routes through the nginx CORS proxy so cross-origin files are reachable.
   * The proxy forwards Range headers, so TensogramFile.fromUrl will use lazy
   * per-message Range requests when the upstream server supports them, falling
   * back to a full eager GET otherwise.
   */
  static async fromUrl(url: string): Promise<Tensoscope> {
    await ensureInit();
    // Do NOT use encodeURIComponent here: nginx's $arg_url returns the raw
    // (non-decoded) query-string value, so encoding "://" as "%3A%2F%2F"
    // breaks both the regex safety-check and the proxy_pass in production.
    // We only encode "&" and "#" which would otherwise truncate the parameter.
    const safeUrl = url.replace(/[&#]/g, encodeURIComponent);
    const proxyUrl = `${import.meta.env.BASE_URL}api/proxy?url=${safeUrl}`;
    const file = await TensogramFile.fromUrl(proxyUrl);
    return new Tensoscope(file, url);
  }

  // ── Index ───────────────────────────────────────────────────────────

  /**
   * Build the file index from metadata only (fast, no payload decode).
   *
   * For remote files this calls `prefetchLayouts` in chunks of
   * `PREFETCH_CHUNK_SIZE` so the per-host browser concurrency limit
   * (6) is honoured and very large files don't queue thousands of
   * Promises at once.  The subsequent `messageMetadata(i)` loop then
   * hits the warm layout cache and makes no network calls.
   */
  async buildIndex(): Promise<FileIndex> {
    if (this._index) return this._index;

    const variables: ObjectInfo[] = [];
    const coordinates: CoordinateInfo[] = [];

    if (this.file.source === 'remote') {
      const indices = Array.from({ length: this.file.messageCount }, (_, i) => i);
      for (let off = 0; off < indices.length; off += PREFETCH_CHUNK_SIZE) {
        const chunk = indices.slice(off, off + PREFETCH_CHUNK_SIZE);
        await this.file.prefetchLayouts(chunk, { concurrency: REMOTE_CONCURRENCY });
      }
    }

    for (let msgIdx = 0; msgIdx < this.file.messageCount; msgIdx++) {
      const meta = await this.file.messageMetadata(msgIdx);
      const base = meta.base ?? [];

      for (let objIdx = 0; objIdx < base.length; objIdx++) {
        const entry = base[objIdx];
        const tensor = getTensor(entry);
        const name = resolveName(entry, objIdx);
        const shape = tensor?.shape ?? [];
        const dtype = String(tensor?.dtype ?? 'unknown');

        const info = {
          msgIndex: msgIdx,
          objIndex: objIdx,
          name,
          shape: [...(shape as unknown as Iterable<number>)],
          dtype,
        };

        if (COORD_NAMES.has(name.toLowerCase())) {
          coordinates.push(info);
        } else {
          // Build user-facing metadata (exclude _reserved_)
          const metadata: Record<string, unknown> = {};
          for (const [k, v] of Object.entries(entry)) {
            if (k !== '_reserved_') metadata[k] = v as unknown;
          }
          variables.push({
            ...info,
            encoding: String(tensor?.encoding ?? 'unknown'),
            compression: String(tensor?.compression ?? 'unknown'),
            metadata,
          });
        }
      }
    }

    this._index = {
      source: this._source,
      messageCount: this.file.messageCount,
      variables,
      coordinates,
    };
    return this._index;
  }

  // ── Field decode ──────────────────────────────────────────────────────

  /**
   * Decode a single object and return as Float32Array with stats.
   *
   * For remote files this calls `messageObject(msgIdx, objIdx)`,
   * which on the lazy HTTP backend issues exactly one Range GET for
   * the target object's frame instead of downloading the whole
   * message — a major bandwidth saving for multi-tensor messages.
   */
  async decodeField(msgIdx: number, objIdx: number): Promise<DecodedField> {
    const msg = await this.file.messageObject(msgIdx, objIdx);
    try {
      const obj = msg.objects[0];
      const typed = obj.data();
      const data = typed instanceof Float32Array
        ? typed
        : new Float32Array(typed as unknown as ArrayLike<number>);
      const stats = computeStats(data);
      return { data, shape: [...obj.descriptor.shape], stats };
    } finally {
      msg.close();
    }
  }

  /** Decode a field and slice along a dimension. */
  async decodeFieldSlice(msgIdx: number, objIdx: number, dim: number, idx: number): Promise<DecodedField> {
    const full = await this.decodeField(msgIdx, objIdx);
    const sliced = sliceArray(full.data, full.shape, dim, idx);
    const stats = computeStats(sliced);
    const newShape = [...full.shape];
    newShape.splice(dim, 1);
    return { data: sliced, shape: newShape, stats };
  }

  // ── Coordinates ───────────────────────────────────────────────────────

  /**
   * Decode and cache lat/lon coordinate arrays for a message.
   *
   * When a file ships 1-D latitude + longitude axes and one or more
   * data variables shaped `[nLat, nLon]` (the natural
   * regular-rectangular-grid layout for GRIB / NetCDF climate data),
   * the renderer's downstream worker wants per-point coordinates —
   * one `(lat, lon)` pair per data cell.  We detect that layout here
   * and expand the axes into a full meshgrid so callers never have
   * to worry about it.
   *
   * If the file already ships per-point coords (lat and lon of
   * length `nLat * nLon`, as produced by
   * `examples/.../add_coords_meshed.py`), the expansion is a no-op
   * and we return the file's coords as-is.
   */
  async fetchCoordinates(msgIdx: number): Promise<CoordinateData | null> {
    const cached = this._coords.get(msgIdx);
    if (cached) return cached;

    const index = await this.buildIndex();
    const latInfo = index.coordinates.find(
      (c) => c.msgIndex === msgIdx && LAT_NAMES.has(c.name.toLowerCase()),
    );
    const lonInfo = index.coordinates.find(
      (c) => c.msgIndex === msgIdx && LON_NAMES.has(c.name.toLowerCase()),
    );

    // Scope inference + expansion to the variables of the requested
    // message.  Heterogeneous multi-message files may have different
    // grids in different messages and should not pick up an axis from
    // some other message.
    const msgVariables = index.variables.filter((v) => v.msgIndex === msgIdx);

    let latAxis: Float32Array;
    let lonAxis: Float32Array;
    if (latInfo && lonInfo) {
      latAxis = (await this.decodeField(latInfo.msgIndex, latInfo.objIndex)).data;
      lonAxis = (await this.decodeField(lonInfo.msgIndex, lonInfo.objIndex)).data;
    } else {
      // Fallback: no explicit coordinate objects.  Try to infer axes
      // from `mars.grid` + data shape.  This covers GRIB-derived
      // files that ship only data, relying on the grid-kind metadata
      // hint for geolocation.
      const inferred = inferAxesFromMarsGrid(msgVariables);
      if (!inferred) return null;
      latAxis = inferred.lat;
      lonAxis = inferred.lon;
    }

    const expanded = expandAxesIfRectangularGrid(latAxis, lonAxis, msgVariables);
    const coords = expanded ?? { lat: latAxis, lon: lonAxis };
    this._coords.set(msgIdx, coords);
    return coords;
  }

  close(): void {
    this.file.close();
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function getTensor(entry: BaseEntry): Record<string, CborValue> | undefined {
  const reserved = entry._reserved_ as Record<string, CborValue> | undefined;
  return reserved?.tensor as Record<string, CborValue> | undefined;
}

function resolveName(entry: BaseEntry, objIdx: number): string {
  if (typeof entry.name === 'string') return entry.name;
  const mars = entry.mars as Record<string, unknown> | undefined;
  if (typeof mars?.param === 'string') return mars.param;
  return `object_${objIdx}`;
}

function computeStats(data: Float32Array): FieldStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
    count++;
  }

  const mean = count > 0 ? sum / count : 0;
  const variance = count > 0 ? sumSq / count - mean * mean : 0;
  return {
    min: count > 0 ? min : 0,
    max: count > 0 ? max : 0,
    mean,
    std: Math.sqrt(Math.max(0, variance)),
  };
}

/**
 * Detect a regular rectangular grid and expand 1-D lat/lon axes to
 * per-point arrays.  Exported for unit testing.
 *
 * Returns `null` when no expansion is needed — either because a
 * matching `[nLat, nLon]` variable is not present, or because the
 * coords are already full-length.
 *
 * @internal
 */
export function expandAxesIfRectangularGrid(
  lat: Float32Array,
  lon: Float32Array,
  variables: readonly ObjectInfo[],
): CoordinateData | null {
  const nLat = lat.length;
  const nLon = lon.length;
  if (nLat === 0 || nLon === 0) return null;
  // When the file already ships meshgridded coords (lat.length === full
  // grid size), no data variable will have shape [lat.length, nLon] —
  // the `gridMatch` search below then returns undefined, which is
  // exactly the no-op answer we want.  No extra guard needed here.
  const gridMatch = variables.find(
    (v) => v.shape.length === 2 && v.shape[0] === nLat && v.shape[1] === nLon,
  );
  if (!gridMatch) return null;

  const n = nLat * nLon;
  const latFull = new Float32Array(n);
  const lonFull = new Float32Array(n);
  for (let i = 0; i < nLat; i++) {
    const base = i * nLon;
    const latVal = lat[i];
    for (let j = 0; j < nLon; j++) {
      latFull[base + j] = latVal;
      lonFull[base + j] = lon[j];
    }
  }
  return { lat: latFull, lon: lonFull };
}

/**
 * Infer 1-D latitude + longitude axes from `mars.grid` metadata + the
 * variable's shape, for files that ship no explicit coordinate
 * objects.  Currently supports `regular_ll` (the common
 * constant-step global lat/lon grid); returns `null` for any other
 * grid kind (`reduced_gg`, octahedral `O*`, Gaussian `N*`, etc.) —
 * those require either a per-point lat/lon pair in the file or a
 * grid-definition lookup that this thin viewer layer can't do
 * standalone.
 *
 * Exported for unit-testing.  @internal
 */
export function inferAxesFromMarsGrid(
  variables: readonly ObjectInfo[],
): { lat: Float32Array; lon: Float32Array } | null {
  for (const v of variables) {
    if (v.shape.length !== 2) continue;
    const mars = v.metadata?.mars as Record<string, unknown> | undefined;
    if (!mars) continue;
    const grid = mars.grid;
    if (typeof grid !== 'string') continue;
    const gridKind = grid.toLowerCase();
    if (gridKind !== 'regular_ll') continue;

    const [nLat, nLon] = v.shape;
    // MARS "area": [north, west, south, east].  Default to a full
    // global domain when absent — matches ECMWF operational data.
    const area = Array.isArray(mars.area) ? (mars.area as unknown[]) : null;
    const north = toNumber(area?.[0]) ?? 90;
    const west = toNumber(area?.[1]) ?? 0;
    const south = toNumber(area?.[2]) ?? -90;
    const east = toNumber(area?.[3]) ?? 360;

    const lat = new Float32Array(nLat);
    for (let i = 0; i < nLat; i++) {
      lat[i] = north + (i * (south - north)) / Math.max(nLat - 1, 1);
    }

    // Longitude endpoint handling: when the area spans a full 360°
    // circle the last sample sits one step before `east` because
    // east ≡ west (mod 360).  For partial longitudinal bands the
    // last sample sits exactly at `east`.
    const lon = new Float32Array(nLon);
    const spansCircle = Math.abs(east - west) >= 360 - 1e-6;
    const denom = spansCircle ? nLon : Math.max(nLon - 1, 1);
    for (let j = 0; j < nLon; j++) {
      lon[j] = west + (j * (east - west)) / denom;
    }
    return { lat, lon };
  }
  return null;
}

/**
 * Convert a CborValue to a plain finite number when possible.
 *
 * Guards against `bigint` values that overflow a `number` — converting
 * a very large `bigint` via `Number(v)` silently returns `Infinity`,
 * which would then propagate into generated lat/lon arrays and
 * destroy the grid.  Return `null` on overflow so the caller falls
 * back to a safe default.
 */
function toNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'bigint') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/** Slice a flat array along one dimension of a multi-dimensional shape. */
function sliceArray(data: Float32Array, shape: number[], dim: number, idx: number): Float32Array {
  if (shape.length <= 1) return data;

  const innerSize = shape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const dimStride = shape[dim] * innerSize;
  const outerCount = shape.slice(0, dim).reduce((a, b) => a * b, 1);

  const result = new Float32Array(outerCount * innerSize);
  for (let outer = 0; outer < outerCount; outer++) {
    const srcBase = outer * dimStride + idx * innerSize;
    const dstBase = outer * innerSize;
    for (let i = 0; i < innerSize; i++) {
      result[dstBase + i] = data[srcBase + i];
    }
  }
  return result;
}

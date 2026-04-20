/**
 * Tensogram wrapper for the viewer.
 *
 * Uses @ecmwf/tensogram (WASM) directly in the browser -- no backend needed.
 * Provides file opening, metadata indexing, field decoding, and coordinate
 * extraction.
 */

import {
  init as tgInit,
  TensogramFile,
  decodeObject,
} from '@ecmwf/tensogram';
import type {
  BaseEntry,
  CborValue,
} from '@ecmwf/tensogram';

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
  private _coords: CoordinateData | null = null;

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
    const proxyUrl = `${import.meta.env.BASE_URL}api/proxy?url=${encodeURIComponent(url)}`;
    const file = await TensogramFile.fromUrl(proxyUrl);
    return new Tensoscope(file, url);
  }

  // ── Index ───────────────────────────────────────────────────────────

  /** Build the file index from metadata only (fast, no payload decode). */
  async buildIndex(): Promise<FileIndex> {
    if (this._index) return this._index;

    const variables: ObjectInfo[] = [];
    const coordinates: CoordinateInfo[] = [];

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

  /** Decode a single object and return as Float32Array with stats. */
  decodeField(msgIdx: number, objIdx: number): DecodedField {
    const raw = this.file.rawMessage(msgIdx);
    const msg = decodeObject(raw, objIdx);
    try {
      const obj = msg.objects[0];
      const typed = obj.data();
      // Convert to Float32Array if needed
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
  decodeFieldSlice(msgIdx: number, objIdx: number, dim: number, idx: number): DecodedField {
    const full = this.decodeField(msgIdx, objIdx);
    const sliced = sliceArray(full.data, full.shape, dim, idx);
    const stats = computeStats(sliced);
    const newShape = [...full.shape];
    newShape.splice(dim, 1);
    return { data: sliced, shape: newShape, stats };
  }

  // ── Coordinates ───────────────────────────────────────────────────────

  /** Decode and cache lat/lon coordinate arrays for a message. */
  async fetchCoordinates(msgIdx: number): Promise<CoordinateData | null> {
    if (this._coords) return this._coords;

    const index = await this.buildIndex();
    const latInfo = index.coordinates.find(
      (c) => c.msgIndex === msgIdx && LAT_NAMES.has(c.name.toLowerCase()),
    );
    const lonInfo = index.coordinates.find(
      (c) => c.msgIndex === msgIdx && LON_NAMES.has(c.name.toLowerCase()),
    );
    if (!latInfo || !lonInfo) return null;

    const lat = this.decodeField(latInfo.msgIndex, latInfo.objIndex).data;
    const lon = this.decodeField(lonInfo.msgIndex, lonInfo.objIndex).data;
    this._coords = { lat, lon };
    return this._coords;
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

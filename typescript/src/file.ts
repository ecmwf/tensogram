// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `TensogramFile` — random access over a `.tgm` file.
 *
 * Three open paths:
 *
 * - {@link TensogramFile.open} — Node local file system.  Pre-reads
 *   the full file so random-access reads are O(1) subarray slices.
 *   Supports {@link TensogramFile.append} (Node-only, local-path only).
 * - {@link TensogramFile.fromUrl} — HTTP(S).  Auto-detects Range
 *   support: on a server that advertises `Accept-Ranges: bytes` and a
 *   finite `Content-Length`, message payloads are fetched on demand
 *   via `Range` requests (no full download).  Servers that don't
 *   advertise Range transparently fall back to eager download.
 * - {@link TensogramFile.fromBytes} — wrap an already-loaded buffer.
 *
 * After open, the public API is identical across all three paths; the
 * backend differences are invisible to consumers.
 *
 * ### API changes from Scope B
 *
 * {@link TensogramFile.rawMessage} is `async` — previously sync — so
 * the lazy HTTP backend can issue the Range request before returning
 * bytes.  Existing call sites add `await`.
 */

import { decode, decodeMetadata, scan } from './decode.js';
import { encode } from './encode.js';
import {
  InvalidArgumentError,
  IoError,
  ObjectError,
  rethrowTyped,
} from './errors.js';
import { errorMessage, withCause } from './internal/io.js';
import type {
  AppendOptions,
  DecodedMessage,
  DecodeOptions,
  EncodeInput,
  FileSource,
  FromUrlOptions,
  GlobalMetadata,
  MessagePosition,
  OpenFileOptions,
} from './types.js';

// ── Module-level constants ────────────────────────────────────────────────

/** Bytes in the fixed Tensogram preamble. */
const PREAMBLE_BYTES = 24;
/** Bytes in the fixed Tensogram postamble. */
const POSTAMBLE_BYTES = 16;
/** Minimum wire-format message length — preamble + postamble with no frames. */
const MIN_MESSAGE_BYTES = PREAMBLE_BYTES + POSTAMBLE_BYTES;
/** Lazy HTTP backend's per-message LRU cache size. */
const LAZY_CACHE_CAPACITY = 32;
/** The magic header every Tensogram preamble starts with. */
const PREAMBLE_MAGIC = new TextEncoder().encode('TENSOGRM');

// ── Internal backend types ────────────────────────────────────────────────

/** Backing storage + fetch strategy for a `TensogramFile`. */
type Backend = InMemoryBackend | LocalFileBackend | LazyHttpBackend;

/** Fully in-memory — fromBytes + eager-HTTP. */
interface InMemoryBackend {
  readonly kind: 'buffer' | 'remote-eager';
  readonly source: FileSource;
  bytes: Uint8Array;
  positions: MessagePosition[];
}

/** Node local filesystem; file is mirrored in memory for O(1) reads. */
interface LocalFileBackend {
  readonly kind: 'local';
  readonly source: 'local';
  path: string | URL;
  bytes: Uint8Array;
  positions: MessagePosition[];
}

/** HTTP with Range request support — message bytes fetched on demand. */
interface LazyHttpBackend {
  readonly kind: 'remote-lazy';
  readonly source: 'remote';
  url: string | URL;
  fetchFn: typeof globalThis.fetch;
  baseHeaders?: HeadersInit;
  signal?: AbortSignal;
  byteLength: number;
  positions: MessagePosition[];
  /** Tiny LRU of already-fetched message bytes. */
  cache: Map<number, Uint8Array>;
  cacheCap: number;
}

// ── TensogramFile ─────────────────────────────────────────────────────────

/**
 * Random-access view over a `.tgm` file.  Async iteration yields one
 * `DecodedMessage` per message in the file.  The caller closes each
 * yielded message to release WASM memory.
 */
export class TensogramFile implements AsyncIterable<DecodedMessage> {
  readonly #backend: Backend;
  #closed = false;

  private constructor(backend: Backend) {
    this.#backend = backend;
  }

  // ── Factories ──

  /**
   * Open a local `.tgm` file.  Node-only — the `node:fs/promises`
   * import is dynamic so browser bundlers can tree-shake this path.
   */
  static async open(
    path: string | URL,
    options: OpenFileOptions = {},
  ): Promise<TensogramFile> {
    if (typeof path !== 'string' && !(path instanceof URL)) {
      throw new InvalidArgumentError(
        'TensogramFile.open: path must be a string or file:// URL',
      );
    }
    let readFile: typeof import('node:fs/promises').readFile;
    try {
      ({ readFile } = await import('node:fs/promises'));
    } catch (cause) {
      throw withCause(
        new IoError('TensogramFile.open requires Node; use fromUrl in browsers'),
        cause,
      );
    }

    let bytes: Uint8Array;
    try {
      const readOpts = options.signal ? { signal: options.signal } : undefined;
      const buf = await readFile(path, readOpts);
      bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } catch (err) {
      throw withCause(new IoError(`TensogramFile.open: ${errorMessage(err)}`), err);
    }
    const positions = rethrowTyped(() => scan(bytes));
    return new TensogramFile({
      kind: 'local',
      source: 'local',
      path,
      bytes,
      positions,
    });
  }

  /**
   * Open a `.tgm` file over HTTP(S).
   *
   * Performs a `HEAD` probe to detect Range support:
   *
   * - `Accept-Ranges: bytes` + finite `Content-Length` → lazy backend
   *   (one small Range request per message on first read, payloads
   *   fetched on demand).
   * - Otherwise → eager backend (one GET, entire body downloaded).
   *
   * Non-`200` HEAD responses also fall back to eager GET — some
   * servers reject HEAD but support Range on GET.
   *
   * The behaviour is indistinguishable to callers except in timing
   * and memory use.
   */
  static async fromUrl(
    url: string | URL,
    options: FromUrlOptions = {},
  ): Promise<TensogramFile> {
    const fetchFn = options.fetch ?? globalThis.fetch;
    if (typeof fetchFn !== 'function') {
      throw new InvalidArgumentError(
        'TensogramFile.fromUrl: no fetch implementation is available',
      );
    }

    // ── Probe for Range support ──
    let probe: Response | undefined;
    try {
      const headInit: RequestInit = { method: 'HEAD' };
      if (options.headers !== undefined) headInit.headers = options.headers;
      if (options.signal !== undefined) headInit.signal = options.signal;
      probe = await fetchFn(url, headInit);
    } catch {
      probe = undefined; // HEAD failed — fall back to eager
    }

    if (probe && probe.ok) {
      const acceptRanges = probe.headers.get('accept-ranges');
      const contentLengthStr = probe.headers.get('content-length');
      const contentLength = contentLengthStr === null ? NaN : parseInt(contentLengthStr, 10);
      // Cap at `MAX_SAFE_INTEGER` — our cursor arithmetic uses JS
      // numbers, so anything beyond 2^53 - 1 can't be tracked without
      // loss.  Anything that size belongs on the eager path anyway.
      if (
        acceptRanges?.toLowerCase().includes('bytes') &&
        Number.isFinite(contentLength) &&
        contentLength > 0 &&
        contentLength <= Number.MAX_SAFE_INTEGER
      ) {
        // Try the lazy path.  If lazy scan fails mid-walk (e.g.
        // streaming-mode message or unrecognised magic), fall through
        // to eager.
        const scanResult = await lazyScanMessages(
          url,
          fetchFn,
          options.headers,
          options.signal,
          contentLength,
        );
        if (scanResult !== null) {
          return new TensogramFile({
            kind: 'remote-lazy',
            source: 'remote',
            url,
            fetchFn,
            baseHeaders: options.headers,
            signal: options.signal,
            byteLength: contentLength,
            positions: scanResult,
            cache: new Map(),
            cacheCap: LAZY_CACHE_CAPACITY,
          });
        }
      }
    }

    // ── Eager fallback ──
    return await TensogramFile.#eagerFromUrl(url, options, fetchFn);
  }

  /**
   * Eager GET — the Scope-B behaviour, used when Range is unsupported
   * or the lazy scan bails out.  Kept as a private static so it has
   * access to the private constructor without an `as any` cast.
   */
  static async #eagerFromUrl(
    url: string | URL,
    options: FromUrlOptions,
    fetchFn: typeof globalThis.fetch,
  ): Promise<TensogramFile> {
    const init: RequestInit = {};
    if (options.headers !== undefined) init.headers = options.headers;
    if (options.signal !== undefined) init.signal = options.signal;

    let response: Response;
    try {
      response = await fetchFn(url, init);
    } catch (err) {
      throw withCause(
        new IoError(`TensogramFile.fromUrl: ${errorMessage(err)}`),
        err,
      );
    }
    if (!response.ok) {
      const status = `${response.status} ${response.statusText || ''}`.trim();
      throw new IoError(`TensogramFile.fromUrl: HTTP ${status}`);
    }
    const buf = await response.arrayBuffer();
    const bytes = new Uint8Array(buf);
    const positions = rethrowTyped(() => scan(bytes));
    return new TensogramFile({
      kind: 'remote-eager',
      source: 'remote',
      bytes,
      positions,
    });
  }

  /**
   * Wrap an already-loaded byte buffer.  The input is defensively
   * copied so later mutation by the caller is invisible.
   */
  static fromBytes(bytes: Uint8Array): TensogramFile {
    if (!(bytes instanceof Uint8Array)) {
      throw new InvalidArgumentError('TensogramFile.fromBytes: expected a Uint8Array');
    }
    const copy = new Uint8Array(bytes);
    const positions = rethrowTyped(() => scan(copy));
    return new TensogramFile({
      kind: 'buffer',
      source: 'buffer',
      bytes: copy,
      positions,
    });
  }

  // ── Accessors ──

  /** Number of messages indexed from the file. */
  get messageCount(): number {
    return this.#backend.positions.length;
  }

  /** Total byte length of the backing file. */
  get byteLength(): number {
    switch (this.#backend.kind) {
      case 'buffer':
      case 'remote-eager':
      case 'local':
        return this.#backend.bytes.byteLength;
      case 'remote-lazy':
        return this.#backend.byteLength;
    }
  }

  /** Where these bytes came from. */
  get source(): FileSource {
    return this.#backend.source;
  }

  // ── Data access ──

  /**
   * Raw bytes of a single message.  For the lazy HTTP backend this
   * issues a Range request on first call; subsequent calls hit an
   * internal LRU cache.
   */
  async rawMessage(index: number): Promise<Uint8Array> {
    this.#assertOpen();
    const b = this.#backend;
    if (!Number.isInteger(index) || index < 0 || index >= b.positions.length) {
      throw new ObjectError(
        `message index ${index} out of range (have ${b.positions.length})`,
      );
    }
    const { offset, length } = b.positions[index];

    switch (b.kind) {
      case 'buffer':
      case 'remote-eager':
      case 'local':
        return b.bytes.subarray(offset, offset + length);
      case 'remote-lazy': {
        const hit = b.cache.get(index);
        if (hit) return hit;
        const bytes = await fetchRange(
          b.url,
          b.fetchFn,
          b.baseHeaders,
          b.signal,
          offset,
          offset + length,
        );
        // Simple LRU: delete + set shifts to newest, drop the oldest
        // entry when over capacity.
        b.cache.set(index, bytes);
        if (b.cache.size > b.cacheCap) {
          const first = b.cache.keys().next();
          if (!first.done) b.cache.delete(first.value);
        }
        return bytes;
      }
    }
  }

  /** Decode a single message by index. */
  async message(index: number, options?: DecodeOptions): Promise<DecodedMessage> {
    const slice = await this.rawMessage(index);
    return decode(slice, options);
  }

  /** Decode only the global metadata for a single message. */
  async messageMetadata(index: number): Promise<GlobalMetadata> {
    const slice = await this.rawMessage(index);
    return decodeMetadata(slice);
  }

  /** Async iterate every message in wire order. */
  async *[Symbol.asyncIterator](): AsyncIterator<DecodedMessage> {
    for (let i = 0; i < this.messageCount; i++) {
      yield await this.message(i);
    }
  }

  // ── Write ──

  /**
   * Encode and append a new message to the underlying file.  Only
   * supported when the file was opened via
   * {@link TensogramFile.open} — every other factory throws
   * `InvalidArgumentError`, mirroring the Rust / Python / FFI /
   * C++ `TensogramFile::append` contract.
   *
   * After a successful append, the in-memory position index and
   * byte mirror are refreshed from disk so subsequent reads see
   * the new message.
   */
  async append(
    metadata: GlobalMetadata,
    objects: readonly EncodeInput[],
    options: AppendOptions = {},
  ): Promise<void> {
    this.#assertOpen();
    if (this.#backend.kind !== 'local') {
      throw new InvalidArgumentError(
        'TensogramFile.append: only supported for files opened via TensogramFile.open() — use encode() and concatenate manually for in-memory or remote sources',
      );
    }
    const b = this.#backend;

    const encodeOpts = options.hash === false ? { hash: false as const } : { hash: 'xxh3' as const };
    const bytes = rethrowTyped(() => encode(metadata, objects, encodeOpts));

    let writeFile: typeof import('node:fs/promises').appendFile;
    let readFile: typeof import('node:fs/promises').readFile;
    try {
      ({ appendFile: writeFile, readFile } = await import('node:fs/promises'));
    } catch (cause) {
      throw withCause(
        new IoError('TensogramFile.append requires Node (node:fs/promises)'),
        cause,
      );
    }

    try {
      await writeFile(b.path, bytes);
    } catch (err) {
      throw withCause(new IoError(`TensogramFile.append: ${errorMessage(err)}`), err);
    }

    // Refresh in-memory mirror + rescan positions from disk.
    try {
      const buf = await readFile(b.path);
      b.bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      b.positions = rethrowTyped(() => scan(b.bytes));
    } catch (err) {
      throw withCause(
        new IoError(`TensogramFile.append: refresh failed: ${errorMessage(err)}`),
        err,
      );
    }
  }

  // ── Lifecycle ──

  /** Release the backing buffer.  Already-decoded messages stay alive. */
  close(): void {
    this.#closed = true;
  }

  #assertOpen(): void {
    if (this.#closed) {
      throw new InvalidArgumentError('TensogramFile has been closed');
    }
  }
}

// ── Lazy HTTP helpers ─────────────────────────────────────────────────────

/**
 * Walk the file one preamble at a time, collecting each message's
 * `(offset, length)` from the preamble's `total_length` field.  A
 * single Range request per message (24 bytes each) is issued.
 *
 * Returns `null` when the lazy walk cannot proceed — the caller falls
 * back to eager download.  Signals:
 *
 * - non-`TENSOGRM` magic encountered;
 * - streaming-mode message (`total_length == 0`) — would require
 *   walking frame-by-frame, an optimisation we don't attempt;
 * - advertised length exceeds the file;
 * - any Range response that isn't 206 (including 200 with the whole
 *   body, which some servers return when Range isn't supported).
 */
async function lazyScanMessages(
  url: string | URL,
  fetchFn: typeof globalThis.fetch,
  headers: HeadersInit | undefined,
  signal: AbortSignal | undefined,
  fileLen: number,
): Promise<MessagePosition[] | null> {
  const positions: MessagePosition[] = [];
  let cursor = 0;

  while (cursor + PREAMBLE_BYTES <= fileLen) {
    let preamble: Uint8Array;
    try {
      preamble = await fetchRange(
        url,
        fetchFn,
        headers,
        signal,
        cursor,
        cursor + PREAMBLE_BYTES,
        /* requireRange */ true,
      );
    } catch {
      return null; // eager fallback
    }
    // Verify the magic prefix.  Any mismatch aborts the lazy walk —
    // `TensogramFile.scan` in eager mode tolerates leading garbage but
    // this fast-path relies on preamble-aligned messages.
    for (let i = 0; i < PREAMBLE_MAGIC.length; i++) {
      if (preamble[i] !== PREAMBLE_MAGIC[i]) return null;
    }
    // total_length is big-endian u64 at byte offset 16.  We use a
    // DataView so we don't depend on host endianness.
    const view = new DataView(preamble.buffer, preamble.byteOffset, PREAMBLE_BYTES);
    const hi = view.getUint32(16, /* littleEndian = */ false);
    const lo = view.getUint32(20, /* littleEndian = */ false);
    const totalLength = hi * 0x1_0000_0000 + lo;
    if (totalLength === 0) return null; // streaming-mode → eager
    if (!Number.isFinite(totalLength) || totalLength < MIN_MESSAGE_BYTES) return null;
    if (cursor + totalLength > fileLen) return null;
    positions.push({ offset: cursor, length: totalLength });
    cursor += totalLength;
  }
  return positions;
}

/**
 * Issue a `Range: bytes=start-end-1` request covering `[start, end)`.
 *
 * - `requireRange: true` (used during the lazy scan) throws on any
 *   non-`206` status so the caller can fall back to eager download.
 * - `requireRange: false` (the default, used for payload fetches) also
 *   accepts `200` with the full body: when the server ignores the
 *   Range header we slice the response on the client side.
 */
async function fetchRange(
  url: string | URL,
  fetchFn: typeof globalThis.fetch,
  baseHeaders: HeadersInit | undefined,
  signal: AbortSignal | undefined,
  start: number,
  end: number,
  requireRange = false,
): Promise<Uint8Array> {
  const headers = new Headers(baseHeaders);
  headers.set('Range', `bytes=${start}-${end - 1}`);
  const init: RequestInit = { method: 'GET', headers };
  if (signal !== undefined) init.signal = signal;

  let resp: Response;
  try {
    resp = await fetchFn(url, init);
  } catch (err) {
    throw withCause(new IoError(`Range fetch failed: ${errorMessage(err)}`), err);
  }

  if (resp.status === 206) {
    const buf = await resp.arrayBuffer();
    return new Uint8Array(buf);
  }

  if (requireRange) {
    throw new IoError(`Range request returned HTTP ${resp.status}; falling back to eager`);
  }

  if (resp.status === 200) {
    // Server returned the full body — slice client-side.
    const buf = await resp.arrayBuffer();
    if (end > buf.byteLength) {
      throw new IoError(
        `Server returned ${buf.byteLength} bytes but client asked for range ending at ${end}`,
      );
    }
    return new Uint8Array(buf).subarray(start, end);
  }

  throw new IoError(`Range fetch: HTTP ${resp.status} ${resp.statusText || ''}`.trim());
}

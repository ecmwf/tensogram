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
import { typedArrayFor } from './dtype.js';
import { encode } from './encode.js';
import {
  InvalidArgumentError,
  IoError,
  ObjectError,
  rethrowTyped,
} from './errors.js';
import { getWbg } from './init.js';
import { fetchRange, type FetchRangeContext } from './internal/httpRange.js';
import { errorMessage, withCause } from './internal/io.js';
import {
  type MessageLayout,
  normaliseFrameFooter,
  normaliseFrameIndex,
  normalisePostambleInfo,
  normalisePreambleInfo,
} from './internal/layout.js';
import { createLimiter, DEFAULT_CONCURRENCY } from './internal/pool.js';
import { concatBytes, flattenRangePairs } from './internal/rangePack.js';
import type {
  AppendOptions,
  DataObjectDescriptor,
  DecodedMessage,
  DecodeOptions,
  DecodeRangeOptions,
  DecodeRangeResult,
  EncodeInput,
  FileSource,
  FromUrlOptions,
  GlobalMetadata,
  MessagePosition,
  OpenFileOptions,
  RangePair,
} from './types.js';

// ── Module-level constants ────────────────────────────────────────────────

/** Bytes in the fixed Tensogram preamble. */
const PREAMBLE_BYTES = 24;
/** Bytes in the fixed Tensogram postamble (`[first_footer_offset u64][total_length u64][end_magic 8]`). */
const POSTAMBLE_BYTES = 24;
/** Minimum wire-format message length — preamble + postamble with no frames. */
const MIN_MESSAGE_BYTES = PREAMBLE_BYTES + POSTAMBLE_BYTES;
/** Lazy HTTP backend's per-message LRU cache size. */
const LAZY_CACHE_CAPACITY = 32;
/** Header-chunk size used when populating a header-indexed layout (matches Rust 256 KB). */
const HEADER_CHUNK_BYTES = 256 * 1024;
/** Footer-suffix size used when populating a footer-indexed layout (matches Rust 256 KB). */
const FOOTER_SUFFIX_BYTES = 256 * 1024;
/** Frames up to this size are fetched in full for descriptor extraction (matches Rust 64 KB). */
const DESCRIPTOR_PREFIX_THRESHOLD = 64 * 1024;
/** Initial prefix size tried when parsing descriptor CBOR that precedes payload (matches Rust 8 KB). */
const DESCRIPTOR_PREFIX_INITIAL_BYTES = 8 * 1024;
/** Size of the common frame header in v3 (2 + 2 + 2 + 2 + 8 bytes). */
const DATA_OBJECT_FRAME_HEADER_BYTES = 16;
/** Size of a data-object frame footer in v3 ([cbor_offset 8][hash 8][ENDF 4]). */
const DATA_OBJECT_FRAME_FOOTER_BYTES = 20;
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
  /** Per-message layout cache (preamble, index, metadata) — mirrors Rust's RemoteBackend.state. */
  layouts: Map<number, MessageLayout>;
  /** Tiny LRU of already-fetched message bytes. */
  cache: Map<number, Uint8Array>;
  cacheCap: number;
  /** Bounded-concurrency limiter for fan-out Range reads. */
  limit: <T>(fn: () => Promise<T>) => Promise<T>;
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
        const ctx: FetchRangeContext = {
          url,
          fetchFn,
          ...(options.headers !== undefined ? { baseHeaders: options.headers } : {}),
          ...(options.signal !== undefined ? { signal: options.signal } : {}),
        };
        // Try the lazy path.  If lazy scan fails mid-walk (e.g.
        // streaming-mode message or unrecognised magic), fall through
        // to eager.
        const scanResult = await lazyScanMessages(ctx, contentLength);
        if (scanResult !== null) {
          const concurrency = options.concurrency ?? DEFAULT_CONCURRENCY;
          const backend: LazyHttpBackend = {
            kind: 'remote-lazy',
            source: 'remote',
            url,
            fetchFn,
            byteLength: contentLength,
            positions: scanResult.positions,
            layouts: scanResult.layouts,
            cache: new Map(),
            cacheCap: LAZY_CACHE_CAPACITY,
            limit: createLimiter(concurrency),
          };
          if (options.headers !== undefined) backend.baseHeaders = options.headers;
          if (options.signal !== undefined) backend.signal = options.signal;
          return new TensogramFile(backend);
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
          lazyFetchContext(b),
          offset,
          offset + length,
        );
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

  /**
   * Decode only the global metadata for a single message.
   *
   * For the lazy-HTTP backend, this fetches at most a 256 KB header
   * chunk (when the message is header-indexed) or a 256 KB footer
   * suffix (when footer-indexed) instead of downloading the full
   * message.  Cached for subsequent calls on the same index.
   */
  async messageMetadata(index: number): Promise<GlobalMetadata> {
    this.#assertOpen();
    this.#assertMessageIndex(index);
    const b = this.#backend;
    if (b.kind === 'remote-lazy') {
      const layout = await this.#ensureLayout(b, index);
      if (layout.metadata) return layout.metadata;
    }
    const slice = await this.rawMessage(index);
    const meta = decodeMetadata(slice);
    if (b.kind === 'remote-lazy') {
      const layout = b.layouts.get(index);
      if (layout) layout.metadata = meta;
    }
    return meta;
  }

  /**
   * Decode metadata + all per-object descriptors for a single message,
   * without materialising payload bytes.
   *
   * For the lazy-HTTP backend this reuses the cached layout's index
   * frame and fetches only the descriptor CBOR per object — full
   * frame for frames up to `DESCRIPTOR_PREFIX_THRESHOLD` (64 KB) and
   * header (16 B) + footer (20 B) + CBOR region for larger frames.
   * For local / in-memory / eager-HTTP backends, falls through to the
   * core `decodeDescriptors` helper on the full message bytes.
   */
  async messageDescriptors(
    index: number,
  ): Promise<{ metadata: GlobalMetadata; descriptors: DataObjectDescriptor[] }> {
    this.#assertOpen();
    this.#assertMessageIndex(index);
    const b = this.#backend;
    if (b.kind === 'remote-lazy') {
      const layout = await this.#ensureLayout(b, index);
      if (layout.metadata && layout.descriptors) {
        return { metadata: layout.metadata, descriptors: layout.descriptors };
      }
      if (layout.metadata && layout.index) {
        const descriptors = await this.#fetchDescriptors(b, layout);
        layout.descriptors = descriptors;
        return { metadata: layout.metadata, descriptors };
      }
    }
    const slice = await this.rawMessage(index);
    const metadata = decodeMetadata(slice);
    const positions = rethrowTyped(() => {
      const raw = getWbg().parse_footer_chunk(slice);
      return raw as unknown;
    });
    // Fall back to decoding the full message to extract descriptors,
    // because `parse_footer_chunk` against a full message is not the
    // right shape (it expects a footer-region slice). The eager path
    // uses the standard decode route.
    void positions;
    const decoded = decode(slice);
    const descriptors = decoded.objects.map((o) => o.descriptor);
    decoded.close();
    return { metadata, descriptors };
  }

  /**
   * Decode a single object from a single message.
   *
   * For the lazy-HTTP backend, this issues exactly one Range GET for
   * the target object's frame bytes (after `#ensureLayout` populates
   * the index — at most one extra header-chunk or footer-suffix
   * Range).  Mirrors Rust `RemoteBackend::read_object`.
   *
   * For local / in-memory / eager-HTTP backends, downloads (or
   * subarrays) the message and uses `decodeObject(buf, idx)`.
   */
  async messageObject(
    msgIndex: number,
    objectIndex: number,
    options?: DecodeOptions,
  ): Promise<DecodedMessage> {
    this.#assertOpen();
    this.#assertMessageIndex(msgIndex);
    if (!Number.isInteger(objectIndex) || objectIndex < 0) {
      throw new InvalidArgumentError(
        `objectIndex must be a non-negative integer, got ${String(objectIndex)}`,
      );
    }
    const b = this.#backend;
    if (b.kind === 'remote-lazy') {
      const layout = await this.#ensureLayout(b, msgIndex);
      if (layout.index) {
        return await this.#decodeObjectFromIndex(b, layout, objectIndex, options);
      }
    }
    const slice = await this.rawMessage(msgIndex);
    const { decodeObject } = await import('./decode.js');
    return decodeObject(slice, objectIndex, options);
  }

  /**
   * Decode partial sub-tensor ranges from a single object in a single
   * message.
   *
   * Same Range-economy as `messageObject` for the lazy-HTTP backend:
   * one Range for the object's frame, then `decodeRangeFromFrame` on
   * the WASM side.  Local / in-memory / eager-HTTP backends use the
   * standard `decodeRange(buf, ...)` path.
   */
  async messageObjectRange(
    msgIndex: number,
    objectIndex: number,
    ranges: readonly RangePair[],
    options?: DecodeRangeOptions,
  ): Promise<DecodeRangeResult> {
    this.#assertOpen();
    this.#assertMessageIndex(msgIndex);
    if (!Number.isInteger(objectIndex) || objectIndex < 0) {
      throw new InvalidArgumentError(
        `objectIndex must be a non-negative integer, got ${String(objectIndex)}`,
      );
    }
    const b = this.#backend;
    if (b.kind === 'remote-lazy') {
      const layout = await this.#ensureLayout(b, msgIndex);
      if (layout.index) {
        return await this.#decodeRangeFromIndex(b, layout, objectIndex, ranges, options);
      }
    }
    const slice = await this.rawMessage(msgIndex);
    const { decodeRange } = await import('./range.js');
    return decodeRange(slice, objectIndex, ranges, options);
  }

  /**
   * Decode the same object across many messages in parallel.
   *
   * Each call uses its own bounded-concurrency limiter (default 6,
   * overridable via `options.concurrency`) so the batch never starves
   * the shared backend pool used by inner layout-discovery fetches —
   * the outer and inner pools are independent.  Callers can request
   * hundreds of messages without overwhelming browser per-host limits.
   *
   * For local / in-memory / eager-HTTP backends, falls through to a
   * sequential loop over `messageObject` (zero benefit to fan-out
   * when bytes are already in memory).
   */
  async messageObjectBatch(
    msgIndices: readonly number[],
    objectIndex: number,
    options?: DecodeOptions & { concurrency?: number },
  ): Promise<DecodedMessage[]> {
    this.#assertOpen();
    if (!Array.isArray(msgIndices)) {
      throw new InvalidArgumentError('msgIndices must be an array');
    }
    for (const i of msgIndices) this.#assertMessageIndex(i);
    const b = this.#backend;
    if (b.kind === 'remote-lazy') {
      const limit = createLimiter(options?.concurrency ?? DEFAULT_CONCURRENCY);
      return Promise.all(
        msgIndices.map((i) => limit(() => this.messageObject(i, objectIndex, options))),
      );
    }
    const out: DecodedMessage[] = [];
    for (const i of msgIndices) {
      out.push(await this.messageObject(i, objectIndex, options));
    }
    return out;
  }

  /**
   * Decode partial ranges from the same object across many messages
   * in parallel.  Same concurrency story as `messageObjectBatch`.
   */
  async messageObjectRangeBatch(
    msgIndices: readonly number[],
    objectIndex: number,
    ranges: readonly RangePair[],
    options?: DecodeRangeOptions & { concurrency?: number },
  ): Promise<DecodeRangeResult[]> {
    this.#assertOpen();
    if (!Array.isArray(msgIndices)) {
      throw new InvalidArgumentError('msgIndices must be an array');
    }
    for (const i of msgIndices) this.#assertMessageIndex(i);
    const b = this.#backend;
    if (b.kind === 'remote-lazy') {
      const limit = createLimiter(options?.concurrency ?? DEFAULT_CONCURRENCY);
      return Promise.all(
        msgIndices.map((i) =>
          limit(() => this.messageObjectRange(i, objectIndex, ranges, options)),
        ),
      );
    }
    const out: DecodeRangeResult[] = [];
    for (const i of msgIndices) {
      out.push(await this.messageObjectRange(i, objectIndex, ranges, options));
    }
    return out;
  }

  /**
   * Pre-warm the layout cache for the given messages so subsequent
   * `messageMetadata` / `messageDescriptors` / `messageObject` /
   * `messageObjectRange` calls hit the cache and make zero extra
   * network calls (per message — the per-object frame fetches still
   * happen).  No-op for local / in-memory / eager-HTTP backends.
   */
  async prefetchLayouts(
    msgIndices: readonly number[],
    options?: { concurrency?: number },
  ): Promise<void> {
    this.#assertOpen();
    const b = this.#backend;
    if (b.kind !== 'remote-lazy') return;
    if (!Array.isArray(msgIndices)) {
      throw new InvalidArgumentError('msgIndices must be an array');
    }
    for (const i of msgIndices) this.#assertMessageIndex(i);
    const limit = createLimiter(options?.concurrency ?? DEFAULT_CONCURRENCY);
    await Promise.all(
      msgIndices.map((i) => limit(() => this.#ensureLayout(b, i).then(() => {}))),
    );
  }

  /** Async iterate every message in wire order. */
  async *[Symbol.asyncIterator](): AsyncIterator<DecodedMessage> {
    for (let i = 0; i < this.messageCount; i++) {
      yield await this.message(i);
    }
  }

  // ── Lazy-HTTP internals ──

  /**
   * Bounds-check a message index against the backend's position table.
   * Used by all public per-message accessors so errors are consistent
   * regardless of backend.
   */
  #assertMessageIndex(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.#backend.positions.length) {
      throw new ObjectError(
        `message index ${index} out of range (have ${this.#backend.positions.length})`,
      );
    }
  }

  /**
   * Populate the metadata + index fields of a lazy-HTTP backend's
   * layout entry for one message, reading the appropriate header
   * chunk or footer suffix via HTTP Range.  No-op when the layout is
   * already fully populated.
   */
  async #ensureLayout(b: LazyHttpBackend, index: number): Promise<MessageLayout> {
    const layout = b.layouts.get(index);
    if (!layout) {
      throw new ObjectError(`missing layout entry for message index ${index}`);
    }
    if (layout.metadata && layout.index) return layout;

    const flags = layout.preamble;
    if (flags.hasHeaderMetadata && flags.hasHeaderIndex) {
      await this.#populateFromHeader(b, layout);
    } else if (flags.hasFooterMetadata && flags.hasFooterIndex) {
      await this.#populateFromFooterSuffix(b, layout);
    } else {
      throw new IoError(
        'lazy HTTP access requires header-indexed or footer-indexed messages',
      );
    }
    return layout;
  }

  /**
   * Fetch up to 256 KB from the start of the message, parse header
   * metadata + index frames out of it, and populate the layout
   * entry.  Mirrors Rust's `RemoteBackend::discover_header_layout_locked`.
   */
  async #populateFromHeader(b: LazyHttpBackend, layout: MessageLayout): Promise<void> {
    const chunkSize = Math.min(layout.length, HEADER_CHUNK_BYTES);
    const bytes = await b.limit(() =>
      fetchRange(lazyFetchContext(b), layout.offset, layout.offset + chunkSize),
    );
    const parsed = rethrowTyped(() => getWbg().parse_header_chunk(bytes)) as {
      metadata: GlobalMetadata | null;
      index: { offsets: BigUint64Array; lengths: BigUint64Array } | null;
    };
    if (!parsed.metadata || !parsed.index) {
      throw new IoError(
        'header chunk did not yield metadata + index (chunk too small or footer-indexed)',
      );
    }
    layout.metadata = parsed.metadata;
    layout.index = normaliseFrameIndex(parsed.index);
  }

  /**
   * Fetch up to 256 KB from the end of the message (the postamble
   * plus the footer region), parse metadata + index frames out of
   * it, and populate the layout entry.  Mirrors Rust's
   * `RemoteBackend::discover_footer_layout_from_suffix_locked`.  Falls
   * back to a separate Range for the footer if it lives outside the
   * 256 KB suffix window.
   *
   * Validates the postamble's `end_magic_ok` flag and range bounds
   * (`firstFooterOffset >= PREAMBLE_BYTES`, `totalLength == layout.length`)
   * before trusting any of its fields — otherwise a corrupt or
   * byzantine server could drive range arithmetic into arbitrary
   * offsets.
   */
  async #populateFromFooterSuffix(b: LazyHttpBackend, layout: MessageLayout): Promise<void> {
    const suffixSize = Math.min(layout.length, FOOTER_SUFFIX_BYTES);
    const suffixStart = layout.offset + layout.length - suffixSize;
    const suffix = await b.limit(() =>
      fetchRange(lazyFetchContext(b), suffixStart, layout.offset + layout.length),
    );
    const postamble = normalisePostambleInfo(
      rethrowTyped(() => getWbg().read_postamble_info(suffix)),
    );
    if (!postamble.endMagicOk) {
      throw new IoError('footer suffix: postamble end_magic check failed');
    }
    if (postamble.totalLength !== layout.length) {
      throw new IoError(
        `footer suffix: postamble total_length (${postamble.totalLength}) ` +
          `does not match preamble total_length (${layout.length})`,
      );
    }
    if (postamble.firstFooterOffset < PREAMBLE_BYTES) {
      throw new IoError(
        `footer suffix: first_footer_offset (${postamble.firstFooterOffset}) ` +
          `is inside the preamble (< ${PREAMBLE_BYTES})`,
      );
    }
    if (postamble.firstFooterOffset >= layout.length - POSTAMBLE_BYTES) {
      throw new IoError(
        `footer suffix: first_footer_offset (${postamble.firstFooterOffset}) ` +
          `is at or past the postamble start (${layout.length - POSTAMBLE_BYTES})`,
      );
    }
    const footerAbsStart = layout.offset + postamble.firstFooterOffset;
    const footerAbsEnd = layout.offset + layout.length - POSTAMBLE_BYTES;
    let footerBytes: Uint8Array;
    if (footerAbsStart >= suffixStart) {
      const localStart = footerAbsStart - suffixStart;
      const localEnd = suffix.length - POSTAMBLE_BYTES;
      footerBytes = suffix.subarray(localStart, localEnd);
    } else {
      footerBytes = await b.limit(() =>
        fetchRange(lazyFetchContext(b), footerAbsStart, footerAbsEnd),
      );
    }
    const parsed = rethrowTyped(() => getWbg().parse_footer_chunk(footerBytes)) as {
      metadata: GlobalMetadata | null;
      index: { offsets: BigUint64Array; lengths: BigUint64Array } | null;
    };
    if (!parsed.metadata || !parsed.index) {
      throw new IoError('footer chunk did not yield metadata + index');
    }
    layout.metadata = parsed.metadata;
    layout.index = normaliseFrameIndex(parsed.index);
  }

  /**
   * Fetch descriptors for every object without decoding payloads.
   *
   * For frames below DESCRIPTOR_PREFIX_THRESHOLD we fetch the whole
   * frame (cheaper than multiple tiny Ranges) and parse the CBOR out
   * via the composed `read_data_object_frame_header` +
   * `read_data_object_frame_footer` + `parse_descriptor_cbor` WASM
   * helpers.  For larger frames we fetch only the header + footer +
   * CBOR region, which can be a few KB even for hundred-megabyte
   * frames.  Mirrors Rust `RemoteBackend::read_descriptor_only`.
   */
  async #fetchDescriptors(
    b: LazyHttpBackend,
    layout: MessageLayout,
  ): Promise<DataObjectDescriptor[]> {
    const idx = layout.index;
    if (!idx) throw new IoError('message has no index frame');
    const tasks = idx.offsets.map((off, i) => () =>
      this.#fetchOneDescriptor(b, layout.offset + off, idx.lengths[i]),
    );
    return await Promise.all(tasks.map((t) => b.limit(t)));
  }

  /**
   * Fetch a single descriptor, using the prefix optimisation when
   * the frame is large enough to make the extra round-trip
   * worthwhile.
   */
  async #fetchOneDescriptor(
    b: LazyHttpBackend,
    frameAbsStart: number,
    frameLength: number,
  ): Promise<DataObjectDescriptor> {
    const wbg = getWbg();

    if (frameLength <= DESCRIPTOR_PREFIX_THRESHOLD) {
      const frameBytes = await fetchRange(
        lazyFetchContext(b),
        frameAbsStart,
        frameAbsStart + frameLength,
      );
      return await this.#parseDescriptorFromFullFrame(frameBytes);
    }

    // Large frame: fetch header (16 B) + footer (last 20 B) to learn
    // the CBOR region's offset, then fetch the CBOR region itself.
    const [headerBytes, footerBytes] = await Promise.all([
      fetchRange(lazyFetchContext(b), frameAbsStart, frameAbsStart + DATA_OBJECT_FRAME_HEADER_BYTES),
      fetchRange(
        lazyFetchContext(b),
        frameAbsStart + frameLength - DATA_OBJECT_FRAME_FOOTER_BYTES,
        frameAbsStart + frameLength,
      ),
    ]);

    const header = rethrowTyped(() => wbg.read_data_object_frame_header(headerBytes)) as {
      is_data_object: boolean;
      cbor_after_payload: boolean;
    };
    if (!header.is_data_object) {
      throw new IoError(
        'descriptor fetch: frame at index is not a data-object frame',
      );
    }
    const footer = normaliseFrameFooter(
      rethrowTyped(() => wbg.read_data_object_frame_footer(footerBytes)),
    );
    if (!footer.endMagicOk) {
      throw new IoError('descriptor fetch: frame missing ENDF trailer');
    }
    if (footer.cborOffset < DATA_OBJECT_FRAME_HEADER_BYTES) {
      throw new IoError(
        `descriptor fetch: cbor_offset (${footer.cborOffset}) below frame header size (${DATA_OBJECT_FRAME_HEADER_BYTES})`,
      );
    }

    const cborAbsStart = frameAbsStart + footer.cborOffset;
    const footerAbsStart = frameAbsStart + frameLength - DATA_OBJECT_FRAME_FOOTER_BYTES;
    if (cborAbsStart >= footerAbsStart) {
      throw new IoError('descriptor fetch: cbor_offset points at or past the frame footer');
    }

    if (header.cbor_after_payload) {
      // CBOR lives in [cborAbsStart, footerAbsStart) — fetch it exactly.
      const cborBytes = await fetchRange(lazyFetchContext(b), cborAbsStart, footerAbsStart);
      return rethrowTyped(() => wbg.parse_descriptor_cbor(cborBytes) as DataObjectDescriptor);
    }

    // CBOR precedes payload — fetch an 8 KB prefix and grow on EOF.
    const maxCborLen = footerAbsStart - cborAbsStart;
    let prefixSize = Math.min(maxCborLen, DESCRIPTOR_PREFIX_INITIAL_BYTES);
    // Up to 4 doublings covers 128 KB; that's larger than any realistic
    // single-descriptor CBOR blob and also the cap in Rust's
    // read_descriptor_only (prefix_size doubles to max_cbor_len).
    while (true) {
      const readEnd = Math.min(cborAbsStart + prefixSize, footerAbsStart);
      const prefixBytes = await fetchRange(lazyFetchContext(b), cborAbsStart, readEnd);
      try {
        return rethrowTyped(
          () => getWbg().parse_descriptor_cbor(prefixBytes) as DataObjectDescriptor,
        );
      } catch (err) {
        if (prefixSize >= maxCborLen) throw err;
        prefixSize = Math.min(prefixSize * 2, maxCborLen);
      }
    }
  }

  /** Parse the descriptor from a frame we've already fetched in full. */
  async #parseDescriptorFromFullFrame(frameBytes: Uint8Array): Promise<DataObjectDescriptor> {
    const wbg = getWbg();
    const footer = normaliseFrameFooter(
      rethrowTyped(() =>
        wbg.read_data_object_frame_footer(
          frameBytes.subarray(frameBytes.byteLength - DATA_OBJECT_FRAME_FOOTER_BYTES),
        ),
      ),
    );
    if (!footer.endMagicOk) {
      throw new IoError('descriptor fetch: frame missing ENDF trailer');
    }
    if (footer.cborOffset < DATA_OBJECT_FRAME_HEADER_BYTES) {
      throw new IoError(
        `descriptor fetch: cbor_offset (${footer.cborOffset}) below frame header size`,
      );
    }
    const header = rethrowTyped(() =>
      wbg.read_data_object_frame_header(frameBytes.subarray(0, DATA_OBJECT_FRAME_HEADER_BYTES)),
    ) as { cbor_after_payload: boolean };
    const footerStart = frameBytes.byteLength - DATA_OBJECT_FRAME_FOOTER_BYTES;
    if (footer.cborOffset >= footerStart) {
      throw new IoError('descriptor fetch: cbor_offset past the frame footer');
    }
    if (header.cbor_after_payload) {
      const cbor = frameBytes.subarray(footer.cborOffset, footerStart);
      return rethrowTyped(() => wbg.parse_descriptor_cbor(cbor) as DataObjectDescriptor);
    }
    // CBOR before payload — try the whole pre-footer region; if that
    // can't be decoded, we have no way to know the exact length in
    // isolation (we'd need the bytes the payload consumes).  But the
    // full-frame path is only hit for small frames, so decoding the
    // region up to the footer is fine: CBOR decoders stop at the end
    // of a well-formed value and the rest is the encoded payload,
    // which is binary garbage as CBOR — so we try the full slice and
    // if that fails, shrink back to the 8 KB default.
    const maxCborLen = footerStart - footer.cborOffset;
    const full = frameBytes.subarray(footer.cborOffset, footerStart);
    try {
      return rethrowTyped(() => wbg.parse_descriptor_cbor(full) as DataObjectDescriptor);
    } catch {
      // Fall back: just parse the first prefix — CBOR is self-delimiting.
      const prefix = frameBytes.subarray(
        footer.cborOffset,
        footer.cborOffset + Math.min(maxCborLen, DESCRIPTOR_PREFIX_INITIAL_BYTES),
      );
      return rethrowTyped(() => wbg.parse_descriptor_cbor(prefix) as DataObjectDescriptor);
    }
  }

  /**
   * Decode one object given a populated layout index — issues exactly
   * one Range GET for the target frame.  Returns a DecodedMessage whose
   * `metadata` is the message's real cached metadata (from
   * `#ensureLayout`), not a default-constructed placeholder.
   */
  async #decodeObjectFromIndex(
    b: LazyHttpBackend,
    layout: MessageLayout,
    objectIndex: number,
    options?: DecodeOptions,
  ): Promise<DecodedMessage> {
    const idx = layout.index;
    if (!idx) throw new IoError('message has no index frame');
    if (objectIndex >= idx.offsets.length) {
      throw new ObjectError(
        `object index ${objectIndex} out of range (have ${idx.offsets.length})`,
      );
    }
    if (!layout.metadata) {
      throw new IoError('message metadata not populated after ensureLayout');
    }
    const frameStart = layout.offset + idx.offsets[objectIndex];
    const frameEnd = frameStart + idx.lengths[objectIndex];
    const frameBytes = await b.limit(() => fetchRange(lazyFetchContext(b), frameStart, frameEnd));
    return await this.#wrapWasmDecodedObject(frameBytes, layout.metadata, options);
  }

  /**
   * Decode partial ranges from one object given a populated layout
   * index — one Range GET for the frame, then decode_range_from_frame
   * on the WASM side.
   */
  async #decodeRangeFromIndex(
    b: LazyHttpBackend,
    layout: MessageLayout,
    objectIndex: number,
    ranges: readonly RangePair[],
    options?: DecodeRangeOptions,
  ): Promise<DecodeRangeResult> {
    const idx = layout.index;
    if (!idx) throw new IoError('message has no index frame');
    if (objectIndex >= idx.offsets.length) {
      throw new ObjectError(
        `object index ${objectIndex} out of range (have ${idx.offsets.length})`,
      );
    }
    const frameStart = layout.offset + idx.offsets[objectIndex];
    const frameEnd = frameStart + idx.lengths[objectIndex];
    const frameBytes = await b.limit(() => fetchRange(lazyFetchContext(b), frameStart, frameEnd));
    const flat = flattenRangePairs(ranges);
    const wbg = getWbg();
    const result = rethrowTyped(
      () =>
        wbg.decode_range_from_frame(frameBytes, flat, options?.verifyHash ?? false) as {
          descriptor: DataObjectDescriptor;
          parts: Uint8Array[];
        },
    );
    if (options?.join) {
      const joined = concatBytes(result.parts);
      return {
        descriptor: result.descriptor,
        parts: [typedArrayFor(result.descriptor.dtype, joined, /* copy */ false)],
      };
    }
    return {
      descriptor: result.descriptor,
      parts: result.parts.map((p) =>
        typedArrayFor(result.descriptor.dtype, p, /* copy */ false),
      ),
    };
  }

  /**
   * Wrap WASM `decode_object_from_frame` output in the same
   * `DecodedMessage` shape that `decode()` produces, so callers can
   * use object.data() / object.dataView() on the single object.
   *
   * `metadata` is the message's real cached metadata (not the
   * default-constructed placeholder the WASM helper returns), so
   * consumers of messageObject / messageObjectBatch see the actual
   * GlobalMetadata.
   */
  async #wrapWasmDecodedObject(
    frameBytes: Uint8Array,
    metadata: GlobalMetadata,
    options?: DecodeOptions,
  ): Promise<DecodedMessage> {
    const { wrapWbgDecodedMessage } = await import('./internal/wbgWrap.js');
    const wbg = getWbg();
    const handle = rethrowTyped(() =>
      wbg.decode_object_from_frame(
        frameBytes,
        options?.verifyHash ?? false,
        options?.restoreNonFinite ?? true,
      ),
    );
    const wrapped = wrapWbgDecodedMessage(handle);
    return { ...wrapped, metadata };
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

/** Build a FetchRangeContext from a LazyHttpBackend. */
function lazyFetchContext(b: LazyHttpBackend): FetchRangeContext {
  const ctx: FetchRangeContext = { url: b.url, fetchFn: b.fetchFn };
  if (b.baseHeaders !== undefined) ctx.baseHeaders = b.baseHeaders;
  if (b.signal !== undefined) ctx.signal = b.signal;
  return ctx;
}

/** Shape returned by `lazyScanMessages` — positions plus preamble-only layouts. */
interface LazyScanResult {
  positions: MessagePosition[];
  layouts: Map<number, MessageLayout>;
}

/**
 * Walk the file one preamble at a time (24-byte Range per message),
 * collecting each message's `(offset, length)` from the preamble's
 * `total_length` field and building a preamble-only `MessageLayout`
 * cache entry per message.  Metadata and index frames remain
 * unpopulated until the caller asks for them via `messageMetadata`
 * / `messageDescriptors` / per-object accessors; at that point the
 * backend issues a single header-chunk or footer-suffix Range read
 * per message.
 *
 * **Not a 256 KB forward-chunk walk.**  The plan initially proposed
 * reading 256 KB per message and parsing header metadata + index
 * inline so header-indexed messages would be ready without a second
 * round trip; this shipped version is the simpler preamble-only
 * walk.  The fused-chunk variant is tracked as a follow-up in
 * `plans/TODO.md` — it needs a benchmark to confirm the round-trip
 * saving outweighs the larger per-message fetches.
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
  ctx: FetchRangeContext,
  fileLen: number,
): Promise<LazyScanResult | null> {
  const positions: MessagePosition[] = [];
  const layouts = new Map<number, MessageLayout>();
  let cursor = 0;

  while (cursor + PREAMBLE_BYTES <= fileLen) {
    let preambleBytes: Uint8Array;
    try {
      preambleBytes = await fetchRange(
        ctx,
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
      if (preambleBytes[i] !== PREAMBLE_MAGIC[i]) return null;
    }

    // Parse structured preamble via WASM and normalise bigints to
    // safe-integer numbers (throws if any field exceeds 2^53 - 1).
    let preamble;
    try {
      const raw = getWbg().read_preamble_info(preambleBytes);
      preamble = normalisePreambleInfo(raw);
    } catch {
      return null;
    }

    if (preamble.totalLength === 0) return null; // streaming-mode → eager
    if (preamble.totalLength < MIN_MESSAGE_BYTES) return null;
    if (cursor + preamble.totalLength > fileLen) return null;

    const msgIdx = positions.length;
    positions.push({ offset: cursor, length: preamble.totalLength });
    layouts.set(msgIdx, {
      offset: cursor,
      length: preamble.totalLength,
      preamble,
    });
    cursor += preamble.totalLength;
  }
  return { positions, layouts };
}

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
  type FrameIndex,
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
    const effectiveBidirectional = options.bidirectional ?? true;
    if (effectiveBidirectional && options.concurrency === 1) {
      throw new InvalidArgumentError(
        'TensogramFile.fromUrl: bidirectional scan requires concurrency >= 2; ' +
          'concurrency: 1 would serialise the paired Range fetch and defeat the purpose',
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
        const concurrency = options.concurrency ?? DEFAULT_CONCURRENCY;
        const limit = createLimiter(concurrency);
        // Try the lazy path.  If lazy scan fails mid-walk (e.g.
        // streaming-mode message or unrecognised magic), fall through
        // to eager.
        const scanResult = await lazyScanMessages(ctx, contentLength, {
          bidirectional: effectiveBidirectional,
          debug: options.debug ?? false,
          limit,
        });
        if (scanResult !== null) {
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
            limit,
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

  /**
   * Per-message `(offset, length)` layouts in file order.
   *
   * Populated by the open-time scan; this accessor is passive — it
   * does not trigger additional fetches.  Returns a fresh array of
   * fresh objects each call so callers cannot mutate the backend's
   * internal positions.  Forward-only and bidirectional opens
   * produce identical layouts on well-formed files; the parity
   * harness uses this to assert walker-mode equivalence.  Mirrors
   * the Rust `TensogramFile::message_layouts` and Python
   * `PyTensogramFile.message_layouts` accessors.
   */
  get messageLayouts(): readonly { offset: number; length: number }[] {
    return this.#backend.positions.map((p) => ({ offset: p.offset, length: p.length }));
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
    // Eager fallback: in-memory / local-file / eager-HTTP backend.
    // `parse_footer_chunk` is *not* called here — it expects a footer-
    // region slice and would error on stray "FR" sequences inside
    // arbitrary payload bytes.  The full-message decode below is the
    // source of truth and is cheap when bytes are already in memory.
    const slice = await this.rawMessage(index);
    const metadata = decodeMetadata(slice);
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
   *
   * Concurrency: every leaf `fetchRange` call inside
   * `#fetchOneDescriptor` is routed through `b.limit`, so the
   * configured cap (default 6) bounds total in-flight HTTP requests
   * — including the parallel header+footer fetches and the CBOR
   * prefix-doubling loop.  We deliberately do *not* wrap the outer
   * descriptor task in `b.limit` as well: doing so would deadlock at
   * low concurrency caps, because each task holds an outer slot
   * while waiting for inner-fetch slots that no other task can
   * release.
   */
  async #fetchDescriptors(
    b: LazyHttpBackend,
    layout: MessageLayout,
  ): Promise<DataObjectDescriptor[]> {
    const idx = layout.index;
    if (!idx) throw new IoError('message has no index frame');
    const tasks = idx.offsets.map((off, i) =>
      this.#fetchOneDescriptor(b, layout.offset + off, idx.lengths[i]),
    );
    return await Promise.all(tasks);
  }

  /**
   * Fetch a single descriptor, using the prefix optimisation when
   * the frame is large enough to make the extra round-trip
   * worthwhile.  Every `fetchRange` call inside this method goes
   * through `b.limit` so the configured per-host concurrency cap is
   * respected even when many descriptor tasks are in flight.
   */
  async #fetchOneDescriptor(
    b: LazyHttpBackend,
    frameAbsStart: number,
    frameLength: number,
  ): Promise<DataObjectDescriptor> {
    const wbg = getWbg();

    if (frameLength <= DESCRIPTOR_PREFIX_THRESHOLD) {
      const frameBytes = await b.limit(() =>
        fetchRange(lazyFetchContext(b), frameAbsStart, frameAbsStart + frameLength),
      );
      return await this.#parseDescriptorFromFullFrame(frameBytes);
    }

    // Large frame: fetch header (16 B) + footer (last 20 B) to learn
    // the CBOR region's offset, then fetch the CBOR region itself.
    // Each leaf fetch is independently throttled by `b.limit` — under
    // a busy pool this naturally serialises inside one descriptor
    // task instead of bypassing the cap.
    const [headerBytes, footerBytes] = await Promise.all([
      b.limit(() =>
        fetchRange(
          lazyFetchContext(b),
          frameAbsStart,
          frameAbsStart + DATA_OBJECT_FRAME_HEADER_BYTES,
        ),
      ),
      b.limit(() =>
        fetchRange(
          lazyFetchContext(b),
          frameAbsStart + frameLength - DATA_OBJECT_FRAME_FOOTER_BYTES,
          frameAbsStart + frameLength,
        ),
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
      const cborBytes = await b.limit(() =>
        fetchRange(lazyFetchContext(b), cborAbsStart, footerAbsStart),
      );
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
      const prefixBytes = await b.limit(() =>
        fetchRange(lazyFetchContext(b), cborAbsStart, readEnd),
      );
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
   * Passes the message's real cached metadata as the
   * `metadataOverride` so consumers of messageObject /
   * messageObjectBatch see the actual `GlobalMetadata` rather than
   * the default-constructed placeholder the WASM single-frame helper
   * returns.  The override is wired in at construction time inside
   * `wrapWbgDecodedMessage` so the returned object's identity stays
   * stable — vital for the wrapper's `FinalizationRegistry` to
   * observe GC and free the handle if `close()` is forgotten.
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
    return wrapWbgDecodedMessage(handle, metadata);
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

type Limiter = <T>(fn: () => Promise<T>) => Promise<T>;

interface LazyScanOptions {
  bidirectional: boolean;
  debug: boolean;
  limit: Limiter;
}

/**
 * Discriminated outcome shapes mirroring the Rust
 * `tensogram::remote_scan_parse::{ForwardOutcome, BackwardOutcome,
 * BackwardCommit}` enums.  The WASM exports serialise each enum
 * via `#[serde(tag = "kind", rename_all_fields = "camelCase")]` so
 * the discriminant is the `kind` field and field names are camelCase.
 */
type ForwardOutcome =
  | { kind: 'Hit'; offset: bigint; length: bigint; msgEnd: bigint }
  | { kind: 'ExceedsBound'; offset: bigint; length: bigint; msgEnd: bigint }
  | { kind: 'Streaming'; remaining: bigint }
  | { kind: 'Terminate'; reason: string };

type BackwardOutcome =
  | { kind: 'Format'; reason: string }
  | { kind: 'Streaming' }
  | {
      kind: 'NeedPreambleValidation';
      msgStart: bigint;
      length: bigint;
      firstFooterOffset: bigint;
    };

type BackwardCommit =
  | { kind: 'Format'; reason: string }
  | { kind: 'Layout'; offset: bigint; length: bigint };

interface ScanLayoutEntry {
  offset: number;
  length: number;
  preamble: PreambleInfo;
  metadata?: GlobalMetadata;
  index?: FrameIndex;
}

interface ScanState {
  next: number;
  prev: number;
  layouts: ScanLayoutEntry[];
  suffixRev: ScanLayoutEntry[];
  bwdActive: boolean;
  fwdTerminated: boolean;
  gapClosed: boolean;
}

type PreambleInfo = MessageLayout['preamble'];

/**
 * Drive the lazy open-time scan.
 *
 * State-machine dispatcher mirroring Rust's `scan_step_locked`:
 * each iteration either issues a paired bidirectional round (when
 * `bwdActive && !fwdTerminated`) or a forward-only step.  Backward
 * yields (format error, streaming preamble, gap-below-min, overlap,
 * exceeds-bound) transition the loop into forward-only continuation
 * from `state.next` — never bail-to-eager, never restart from offset
 * 0.
 *
 * Returns `null` only on unrecoverable errors (transport failure on
 * a Range fetch, magic mismatch on the forward step's preamble) so
 * the caller can fall back to eager download.
 */
async function lazyScanMessages(
  ctx: FetchRangeContext,
  fileLen: number,
  options: LazyScanOptions,
): Promise<LazyScanResult | null> {
  if (options.debug) {
    console.debug(
      'tensogram:scan:mode',
      options.bidirectional ? 'bidirectional' : 'forward-only',
    );
  }
  const state: ScanState = {
    next: 0,
    prev: fileLen,
    layouts: [],
    suffixRev: [],
    bwdActive: options.bidirectional,
    fwdTerminated: false,
    gapClosed: false,
  };

  if (state.bwdActive && !state.fwdTerminated) {
    const ok = await runPipelinedBidirectional(ctx, fileLen, state, options);
    if (!ok) return null;
  }
  while (!scanComplete(state)) {
    const ok = await tryForwardStep(ctx, fileLen, state, options);
    if (!ok) return null;
  }
  if (state.layouts.length === 0 && state.suffixRev.length === 0 && !state.gapClosed) {
    return null;
  }
  return finaliseScan(state);
}

function scanComplete(s: ScanState): boolean {
  return s.gapClosed || (s.fwdTerminated && s.suffixRev.length === 0);
}

/**
 * Pipelined bidirectional walk: each iteration fetches the next forward
 * preamble, the next backward postamble, AND the previous iteration's
 * candidate-preamble validation in one parallel `Promise.allSettled`.
 *
 * That overlap collapses the per-round critical path from 2 RTTs to 1
 * RTT, mirroring the Rust `scan_pipelined_async` shape so wall-clock
 * for full-discovery operations is within constant factor of the Rust
 * remote backend.
 *
 * Bails out (returns `true` after disabling backward) on any anomaly:
 * gap-below-min, format error, ExceedsBound, streaming, or backward
 * overlap.  Locally-accumulated forward layouts commit to
 * `state.layouts` so the caller's subsequent forward-only walk
 * continues from the latest discovered position; the backward suffix
 * is discarded by `disableBackward` (mirrors Rust's `disable_backward`)
 * and any unvalidated pending candidate is dropped.  Returns `false`
 * only on a transport-layer rejection that the caller treats as
 * unrecoverable.
 */
async function runPipelinedBidirectional(
  ctx: FetchRangeContext,
  fileLen: number,
  state: ScanState,
  options: LazyScanOptions,
): Promise<boolean> {
  const minSize = MIN_MESSAGE_BYTES;
  const wbg = getWbg();

  type Pending = { msgStart: number; length: number; firstFooterOffset: number };

  const localFwd: ScanLayoutEntry[] = [];
  const localBwd: ScanLayoutEntry[] = [];
  let fwdCursor = state.next;
  let bwdCursor = state.prev;
  let pending: Pending | undefined;
  let bailReason: string | undefined;

  // Returns `false` if the caller's top-level signal aborted during
  // the footer fetch (caller must surface the abort by returning
  // `false` from the scan).  Returns `true` otherwise — including
  // the `bailReason`-set fallback paths, which the caller detects
  // via the shared `bailReason` after the call.
  const validatePending = async (
    p: Pending,
    valBytes: Uint8Array,
    overrideSignal?: AbortSignal,
  ): Promise<boolean> => {
    let candidateFooterBytes: Uint8Array | undefined;
    if (
      footerRegionPresent(p.firstFooterOffset, p.length) &&
      preambleHasFooterIndex(valBytes)
    ) {
      const footerStart = p.msgStart + p.firstFooterOffset;
      const footerEnd = p.msgStart + p.length - POSTAMBLE_BYTES;
      if (footerStart < footerEnd) {
        try {
          candidateFooterBytes = await options.limit(() =>
            fetchRange(ctx, footerStart, footerEnd, true, overrideSignal),
          );
        } catch {
          if (ctx.signal?.aborted) return false;
          candidateFooterBytes = undefined;
        }
      }
    }
    const validation = wbg.validate_backward_preamble_outcome(
      valBytes,
      BigInt(p.msgStart),
      BigInt(p.length),
    ) as BackwardCommit;
    if (validation.kind === 'Format') {
      bailReason = validation.reason;
      return true;
    }
    const layout = buildLayoutFromPreamble(valBytes, p.msgStart, p.length);
    if (layout === null) {
      bailReason = 'preamble-parse-error-bwd';
      return true;
    }
    if (candidateFooterBytes !== undefined) {
      tryApplyEagerFooter(layout, candidateFooterBytes, options);
    }
    localBwd.push(layout);
    return true;
  };

  while (true) {
    if (fwdCursor === bwdCursor) {
      break;
    }
    if (bwdCursor < fwdCursor + minSize) {
      bailReason = 'gap-below-min-message-size';
      break;
    }
    if (ctx.signal?.aborted) return false;

    // Per-iteration child controller: aborts on first sibling failure
    // so a hung Range GET can't stall the round; also forwards the
    // caller's top-level abort so user cancellation still propagates.
    const childAc = new AbortController();
    let parentAbortHandler: (() => void) | undefined;
    if (ctx.signal !== undefined) {
      if (ctx.signal.aborted) {
        childAc.abort();
      } else {
        parentAbortHandler = (): void => childAc.abort();
        ctx.signal.addEventListener('abort', parentAbortHandler, { once: true });
      }
    }
    const overrideSignal = childAc.signal;

    try {
      const fwdPromise = options.limit(() =>
        fetchRange(ctx, fwdCursor, fwdCursor + PREAMBLE_BYTES, true, overrideSignal),
      );
      const bwdPromise = options.limit(() =>
        fetchRange(ctx, bwdCursor - POSTAMBLE_BYTES, bwdCursor, true, overrideSignal),
      );
      const pendingForFetch = pending;
      const valPromise =
        pendingForFetch !== undefined
          ? options.limit(() =>
              fetchRange(
                ctx,
                pendingForFetch.msgStart,
                pendingForFetch.msgStart + PREAMBLE_BYTES,
                true,
                overrideSignal,
              ),
            )
          : undefined;

      const promises: Promise<Uint8Array>[] =
        valPromise !== undefined
          ? [fwdPromise, bwdPromise, valPromise]
          : [fwdPromise, bwdPromise];
      for (const p of promises) {
        p.then(undefined, () => childAc.abort());
      }

      const settled = await Promise.allSettled(promises);
      if (settled.some((r) => r.status === 'rejected')) return false;

      const fwdBytes = (settled[0] as PromiseFulfilledResult<Uint8Array>).value;
      const bwdBytes = (settled[1] as PromiseFulfilledResult<Uint8Array>).value;
      const valBytes =
        valPromise !== undefined
          ? (settled[2] as PromiseFulfilledResult<Uint8Array>).value
          : undefined;

      if (valBytes !== undefined && pending !== undefined) {
        const p = pending;
        pending = undefined;
        const ok = await validatePending(p, valBytes, overrideSignal);
        if (!ok) return false;
        if (bailReason !== undefined) break;
      }

      const preIterFwdCursor = fwdCursor;
      const fwdOutcome = wbg.parse_forward_preamble_outcome(
        fwdBytes,
        BigInt(fwdCursor),
        BigInt(fileLen),
        BigInt(bwdCursor),
      ) as ForwardOutcome;
      if (fwdOutcome.kind === 'Hit') {
        const layout = buildLayoutFromPreamble(
          fwdBytes,
          Number(fwdOutcome.offset),
          Number(fwdOutcome.length),
        );
        if (layout === null) {
          bailReason = 'preamble-parse-error-fwd';
          break;
        }
        localFwd.push(layout);
        fwdCursor = Number(fwdOutcome.msgEnd);
      } else if (fwdOutcome.kind === 'ExceedsBound') {
        bailReason = 'forward-exceeds-backward-bound';
        break;
      } else if (fwdOutcome.kind === 'Streaming') {
        bailReason = 'streaming-fwd-non-tail';
        break;
      } else {
        bailReason = `terminate-${fwdOutcome.reason}`;
        break;
      }

      const bwdOutcome = wbg.parse_backward_postamble_outcome(
        bwdBytes,
        BigInt(preIterFwdCursor),
        BigInt(bwdCursor),
      ) as BackwardOutcome;
      if (bwdOutcome.kind === 'NeedPreambleValidation') {
        const candidateStart = Number(bwdOutcome.msgStart);
        const candidateLength = Number(bwdOutcome.length);
        // Same-message meet: forward just committed the same message that
        // backward postamble points at (3-msg files, odd-count middle).
        // Yield backward silently — don't pending-queue a duplicate of an
        // already-forward-committed layout, and keep `bwdCursor` where it
        // is so the next iteration's `cursorsMet` test fires correctly.
        if (
          fwdOutcome.kind === 'Hit' &&
          Number(fwdOutcome.offset) === candidateStart &&
          Number(fwdOutcome.length) === candidateLength
        ) {
          // bwdCursor stays at its pre-iter value — fwd has already
          // claimed [candidateStart, fwdCursor), and they meet there.
        } else {
          pending = {
            msgStart: candidateStart,
            length: candidateLength,
            firstFooterOffset: Number(bwdOutcome.firstFooterOffset),
          };
          bwdCursor = candidateStart;
        }
      } else if (bwdOutcome.kind === 'Format') {
        bailReason = bwdOutcome.reason;
        break;
      } else {
        bailReason = 'streaming-zero-bwd';
        break;
      }
    } finally {
      if (parentAbortHandler !== undefined && ctx.signal !== undefined) {
        ctx.signal.removeEventListener('abort', parentAbortHandler);
      }
    }
  }

  if (pending !== undefined && bailReason === undefined) {
    const p = pending;
    pending = undefined;
    let valBytes: Uint8Array;
    try {
      valBytes = await options.limit(() =>
        fetchRange(ctx, p.msgStart, p.msgStart + PREAMBLE_BYTES, true),
      );
    } catch {
      return false;
    }
    const ok = await validatePending(p, valBytes);
    if (!ok) return false;
  }

  for (const layout of localFwd) {
    recordForwardHop(state, layout, options);
  }
  for (const layout of localBwd) {
    recordBackwardHop(state, layout, options);
  }
  if (state.bwdActive && state.next === state.prev) {
    closeGap(state, options);
  } else if (bailReason !== undefined && state.bwdActive) {
    disableBackward(state, bailReason, options);
  }
  return true;
}



function footerRegionPresent(firstFooterOffset: number, length: number): boolean {
  if (length < POSTAMBLE_BYTES) return false;
  const footerMax = length - POSTAMBLE_BYTES;
  return firstFooterOffset >= PREAMBLE_BYTES && firstFooterOffset < footerMax;
}

// Preamble flag bits per wire-format §3 (u16 BE at offset 10):
//   FOOTER_METADATA = 1 << 1, FOOTER_INDEX = 1 << 3.
const FLAG_FOOTER_METADATA = 1 << 1;
const FLAG_FOOTER_INDEX = 1 << 3;

function preambleHasFooterIndex(preambleBytes: Uint8Array): boolean {
  if (preambleBytes.length < 12) return false;
  const flags = (preambleBytes[10] << 8) | preambleBytes[11];
  return (flags & FLAG_FOOTER_METADATA) !== 0 && (flags & FLAG_FOOTER_INDEX) !== 0;
}

async function tryForwardStep(
  ctx: FetchRangeContext,
  fileLen: number,
  state: ScanState,
  options: LazyScanOptions,
): Promise<boolean> {
  if (state.next + MIN_MESSAGE_BYTES > fileLen) {
    terminateForward(state, 'eof', options);
    return true;
  }
  if (ctx.signal?.aborted) return false;

  let preambleBytes: Uint8Array;
  try {
    preambleBytes = await options.limit(() =>
      fetchRange(ctx, state.next, state.next + PREAMBLE_BYTES, true),
    );
  } catch {
    return false;
  }

  const wbg = getWbg();
  const outcome = wbg.parse_forward_preamble_outcome(
    preambleBytes,
    BigInt(state.next),
    BigInt(fileLen),
    BigInt(fileLen),
  ) as ForwardOutcome;

  switch (outcome.kind) {
    case 'Hit': {
      const layout = buildLayoutFromPreamble(
        preambleBytes,
        Number(outcome.offset),
        Number(outcome.length),
      );
      if (layout === null) return false;
      recordForwardHop(state, layout, options);
      return true;
    }
    case 'Streaming':
      return false;
    case 'Terminate': {
      if (state.layouts.length === 0 && state.suffixRev.length === 0) {
        return false;
      }
      terminateForward(state, outcome.reason, options);
      return true;
    }
    case 'ExceedsBound':
      return false;
  }
}

/**
 * Populate `layout.metadata` and `layout.index` from a pre-fetched
 * footer chunk when (and only when) the just-validated preamble's
 * flags carry FOOTER_METADATA + FOOTER_INDEX.  Best-effort: any parse
 * failure is silently swallowed and the lazy `#ensureLayout` path
 * picks up the layout later.
 *
 * Header-indexed messages with a footer hash frame may have a
 * non-empty footer region whose bytes the dispatcher fetched
 * speculatively; for those, the FOOTER_INDEX flag check below
 * short-circuits and the bytes are discarded harmlessly.
 */
function tryApplyEagerFooter(
  layout: ScanLayoutEntry,
  footerBytes: Uint8Array,
  options: LazyScanOptions,
): void {
  const flags = layout.preamble;
  if (!(flags.hasFooterMetadata && flags.hasFooterIndex)) {
    return;
  }
  let parsed: {
    metadata: GlobalMetadata | null;
    index: { offsets: BigUint64Array; lengths: BigUint64Array } | null;
  };
  try {
    parsed = getWbg().parse_footer_chunk(footerBytes) as typeof parsed;
  } catch {
    return;
  }
  if (parsed.metadata && parsed.index) {
    layout.metadata = parsed.metadata;
    layout.index = normaliseFrameIndex(parsed.index);
    if (options.debug) {
      console.debug('tensogram:scan:footer-eager', {
        offset: layout.offset,
        footerBytes: footerBytes.length,
      });
    }
  }
}

function buildLayoutFromPreamble(
  preambleBytes: Uint8Array,
  offset: number,
  length: number,
): ScanLayoutEntry | null {
  try {
    const raw = getWbg().read_preamble_info(preambleBytes);
    const preamble = normalisePreambleInfo(raw);
    return { offset, length, preamble };
  } catch {
    return null;
  }
}

function recordForwardHop(
  state: ScanState,
  layout: ScanLayoutEntry,
  options: LazyScanOptions,
): void {
  state.layouts.push(layout);
  state.next = layout.offset + layout.length;
  if (options.debug) {
    console.debug('tensogram:scan:hop', {
      direction: 'fwd',
      offset: layout.offset,
      length: layout.length,
    });
  }
}

function recordBackwardHop(
  state: ScanState,
  layout: ScanLayoutEntry,
  options: LazyScanOptions,
): void {
  state.suffixRev.push(layout);
  state.prev = layout.offset;
  if (options.debug) {
    console.debug('tensogram:scan:hop', {
      direction: 'bwd',
      offset: layout.offset,
      length: layout.length,
    });
  }
}

function disableBackward(state: ScanState, reason: string, options: LazyScanOptions): void {
  state.bwdActive = false;
  state.suffixRev.length = 0;
  if (options.debug) console.debug('tensogram:scan:fallback', reason);
}

function terminateForward(state: ScanState, reason: string, options: LazyScanOptions): void {
  state.fwdTerminated = true;
  state.bwdActive = false;
  state.suffixRev.length = 0;
  if (options.debug) console.debug('tensogram:scan:fwd-terminated', reason);
}

function closeGap(state: ScanState, options: LazyScanOptions): void {
  state.gapClosed = true;
  if (options.debug) console.debug('tensogram:scan:gap-closed');
}

function finaliseScan(state: ScanState): LazyScanResult {
  const reversed = state.suffixRev.slice().reverse();
  const positions: MessagePosition[] = [
    ...state.layouts.map((l) => ({ offset: l.offset, length: l.length })),
    ...reversed.map((l) => ({ offset: l.offset, length: l.length })),
  ];
  const layouts = new Map<number, MessageLayout>();
  const toMessageLayout = (l: ScanLayoutEntry): MessageLayout => ({
    offset: l.offset,
    length: l.length,
    preamble: l.preamble,
    metadata: l.metadata,
    index: l.index,
  });
  state.layouts.forEach((l, i) => layouts.set(i, toMessageLayout(l)));
  reversed.forEach((l, j) => layouts.set(state.layouts.length + j, toMessageLayout(l)));
  return { positions, layouts };
}

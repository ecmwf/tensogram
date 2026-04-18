// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `StreamingEncoder` — frame-at-a-time message construction.
 *
 * Mirrors Rust `StreamingEncoder<Vec<u8>>` and Python `StreamingEncoder`.
 * The backing sink is an in-memory `Vec<u8>` on the WASM side; callers
 * retrieve the complete message via {@link StreamingEncoder.finish} when
 * done.
 *
 * The class is **single-use**: after `finish()` (or `close()` without
 * finishing) the instance is dead and every method throws
 * `InvalidArgumentError`.  Keep the construction cheap and pair it with
 * `try { ... } finally { enc.close(); }` whenever early failure is
 * possible.
 *
 * Example:
 *
 * ```ts
 * const enc = new StreamingEncoder({ version: 2 });
 * enc.writePreceder({ mars: { param: '2t' } });
 * enc.writeObject(descriptor, new Float32Array([1, 2, 3]));
 * enc.writeObject(otherDescriptor, new Float64Array([4, 5, 6]));
 * const bytes = enc.finish();     // Uint8Array
 * ```
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';
import {
  assertValidDescriptor,
  assertValidMetadata,
  toUint8View,
} from './internal/validation.js';
import type {
  BaseEntry,
  DataObjectDescriptor,
  GlobalMetadata,
  StreamingEncoderOptions,
} from './types.js';

// Structural view of the wasm-bindgen-generated class.  Keeping the
// binding narrow prevents accidental reliance on internal surface.
interface WbgStreamingEncoder {
  free(): void;
  write_object(descriptor_js: unknown, data: unknown): void;
  write_object_pre_encoded(descriptor_js: unknown, data: unknown): void;
  write_preceder(metadata_js: unknown): void;
  object_count(): number;
  bytes_written(): number;
  finish(): Uint8Array;
}

const registry = new FinalizationRegistry<WbgStreamingEncoder>((h) => {
  try {
    h.free();
  } catch {
    /* best effort */
  }
});

/**
 * Frame-at-a-time Tensogram encoder.  Two modes, selected at
 * construction by the presence of `opts.onBytes`:
 *
 * - **Buffered (default).**  Frames accumulate in an internal buffer;
 *   {@link finish} returns the complete wire-format message.
 * - **Streaming.**  Frames are forwarded to `opts.onBytes` as they're
 *   produced; {@link finish} returns an empty `Uint8Array` after
 *   flushing the footer.
 *
 * Writing is synchronous on the JS side because the WASM calls are
 * synchronous.  The callback in streaming mode must therefore also
 * complete synchronously — `Promise` returns are silently discarded.
 */
export class StreamingEncoder {
  readonly #handle: WbgStreamingEncoder;
  readonly #streaming: boolean;
  #closed = false;

  /**
   * Construct a new encoder and write the preamble + header metadata
   * frame.  In buffered mode the bytes accumulate internally; in
   * streaming mode they flow to `opts.onBytes` immediately.
   *
   * @param metadata - Global metadata (`version: 2` required).
   * @param opts     - Hash selection + optional `onBytes` sink.
   * @throws {InvalidArgumentError} when `metadata` is malformed or
   *   when `opts.onBytes` is supplied but is not a function.
   */
  constructor(metadata: GlobalMetadata, opts?: StreamingEncoderOptions) {
    assertValidMetadata(metadata);
    if (opts?.onBytes !== undefined && typeof opts.onBytes !== 'function') {
      throw new InvalidArgumentError(
        'StreamingEncoder: opts.onBytes must be a function',
      );
    }
    const wbg = getWbg();
    const hash = opts?.hash !== false;
    const sink = opts?.onBytes;
    this.#streaming = sink !== undefined;
    this.#handle = rethrowTyped(
      () =>
        new wbg.StreamingEncoder(metadata, hash, sink) as unknown as WbgStreamingEncoder,
    );
    registry.register(this, this.#handle, this);
  }

  /**
   * Write a PrecederMetadata frame for the next data object.  Must be
   * followed by exactly one {@link writeObject} or
   * {@link writeObjectPreEncoded} before another preceder or
   * {@link finish}.
   */
  writePreceder(entry: BaseEntry): void {
    this.#assertOpen();
    if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new InvalidArgumentError(
        'writePreceder: entry must be a plain object of per-object metadata',
      );
    }
    if ('_reserved_' in entry) {
      throw new InvalidArgumentError(
        'writePreceder: entry must not contain `_reserved_` — the library manages it',
      );
    }
    rethrowTyped(() => this.#handle.write_preceder(entry));
  }

  /**
   * Encode and write a single data object.
   *
   * @param descriptor - Descriptor of the object.
   * @param data       - Raw native-endian payload (any `ArrayBufferView`).
   */
  writeObject(descriptor: DataObjectDescriptor, data: ArrayBufferView): void {
    this.#assertOpen();
    assertValidDescriptor(descriptor);
    if (!ArrayBuffer.isView(data)) {
      throw new InvalidArgumentError(
        'writeObject: data must be an ArrayBufferView (TypedArray, DataView, …)',
      );
    }
    rethrowTyped(() =>
      this.#handle.write_object(descriptor as unknown, toUint8View(data)),
    );
  }

  /**
   * Write a pre-encoded data object (bypass the encoding pipeline).
   *
   * The descriptor must accurately describe the encoding that was
   * already applied, including any codec-specific params (e.g.
   * `szip_block_offsets`).
   */
  writeObjectPreEncoded(descriptor: DataObjectDescriptor, data: Uint8Array): void {
    this.#assertOpen();
    assertValidDescriptor(descriptor);
    if (!(data instanceof Uint8Array)) {
      throw new InvalidArgumentError(
        'writeObjectPreEncoded: data must be a Uint8Array of pre-encoded bytes',
      );
    }
    rethrowTyped(() =>
      this.#handle.write_object_pre_encoded(descriptor as unknown, data),
    );
  }

  /** Number of data objects written so far. */
  get objectCount(): number {
    this.#assertOpen();
    return this.#handle.object_count();
  }

  /**
   * Total bytes written to the internal buffer so far (preamble +
   * header frames + all completed data-object frames).  Crosses the
   * WASM boundary as `f64`, which is lossless up to
   * `Number.MAX_SAFE_INTEGER` (≈ 9 PiB) — well beyond any realistic
   * Tensogram message.
   */
  get bytesWritten(): number {
    this.#assertOpen();
    return this.#handle.bytes_written();
  }

  /**
   * Finalise the message: write footer frames + postamble.  In
   * buffered mode returns the complete wire bytes; in streaming mode
   * the footer flows through the `onBytes` callback and this returns
   * an empty `Uint8Array` (zero-length marker, not a failure — every
   * byte has already been delivered).
   *
   * The encoder is closed after this call; any further method raises
   * `InvalidArgumentError`.
   */
  finish(): Uint8Array {
    this.#assertOpen();
    const bytes = rethrowTyped(() => this.#handle.finish());
    this.#markClosed();
    return bytes;
  }

  /**
   * `true` when the encoder was constructed with an `onBytes`
   * callback — bytes flow through the callback rather than
   * accumulating in an internal buffer.  Useful for code that needs
   * to branch on mode (e.g. deciding whether to expect bytes back
   * from `finish()`).
   */
  get streaming(): boolean {
    return this.#streaming;
  }

  /**
   * Release the underlying WASM handle without finalising.  Safe to
   * call multiple times.  Any partial buffer is discarded.
   */
  close(): void {
    if (this.#closed) return;
    this.#markClosed();
    try {
      this.#handle.free();
    } catch {
      /* best effort */
    }
  }

  #markClosed(): void {
    this.#closed = true;
    registry.unregister(this);
  }

  #assertOpen(): void {
    if (this.#closed) {
      throw new InvalidArgumentError('StreamingEncoder has been finished or closed');
    }
  }
}



// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Progressive decoding via the Web Streams API.
 *
 * `decodeStream` wraps the WASM-side `StreamingDecoder` in an async
 * generator: feed a `ReadableStream<Uint8Array>` (e.g. `fetch().body`)
 * and receive `DecodedFrame` objects as complete messages arrive.
 *
 * The underlying decoder tolerates corruption — a single bad message
 * is skipped and the iterator continues. Use the `onError` callback
 * to observe skips without interrupting the stream.
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';
import { typedArrayFor } from './dtype.js';
import type {
  BaseEntry,
  DataObjectDescriptor,
  DecodedFrame,
  DecodeStreamOptions,
  TypedArray,
} from './types.js';

/**
 * Structural view of the wasm-bindgen-generated `StreamingDecoder`.
 * Kept narrow so we never accidentally rely on the full surface.
 */
interface WbgStreamingDecoder {
  free(): void;
  feed(chunk: Uint8Array): void;
  next_frame(): WbgDecodedFrame | undefined;
  last_error(): string | undefined;
  skipped_count(): number;
  set_max_buffer(max_bytes: number): void;
}

interface WbgDecodedFrame {
  free(): void;
  descriptor(): unknown;
  base_entry(): unknown;
  data_u8(): Uint8Array;
  byte_length(): number;
}

const frameRegistry = new FinalizationRegistry<WbgDecodedFrame>((handle) => {
  try {
    handle.free();
  } catch {
    /* best effort */
  }
});

/**
 * Progressively decode a `ReadableStream<Uint8Array>` into a sequence
 * of `DecodedFrame` objects. Each frame corresponds to one data object
 * within a Tensogram message.
 *
 * ```ts
 * const res = await fetch('/forecast.tgm');
 * for await (const frame of decodeStream(res.body!)) {
 *   render(frame.descriptor.shape, frame.data());
 *   frame.close();
 * }
 * ```
 *
 * The generator cleans up on early `break`, thrown exceptions, and
 * `AbortSignal` firing. Corrupt messages are silently skipped — pass
 * `onError` to observe them.
 */
export async function* decodeStream(
  stream: ReadableStream<Uint8Array>,
  options: DecodeStreamOptions = {},
): AsyncGenerator<DecodedFrame, void, void> {
  if (!(stream instanceof ReadableStream)) {
    throw new InvalidArgumentError(
      'decodeStream: expected a ReadableStream<Uint8Array>',
      'decodeStream: expected a ReadableStream<Uint8Array>',
    );
  }

  const wbg = getWbg();
  const decoder = new wbg.StreamingDecoder() as unknown as WbgStreamingDecoder;

  if (options.maxBufferBytes !== undefined) {
    decoder.set_max_buffer(options.maxBufferBytes);
  }

  const reader = stream.getReader();
  let lastReportedSkipCount = 0;

  // Cancel the reader if the external signal fires.
  const onAbort = (): void => {
    reader.cancel(options.signal?.reason).catch(() => undefined);
  };
  options.signal?.addEventListener('abort', onAbort, { once: true });

  try {
    while (true) {
      options.signal?.throwIfAborted();

      const { done, value } = await reader.read();
      if (done) break;
      if (value === undefined) continue;

      rethrowTyped(() => decoder.feed(value));
      reportErrors(decoder, options.onError, lastReportedSkipCount);
      lastReportedSkipCount = decoder.skipped_count();

      for (;;) {
        const handle = rethrowTyped(() => decoder.next_frame());
        if (handle === undefined) break;
        yield buildFrame(handle);
      }
    }

    // Drain any frames that completed on the final chunk.
    for (;;) {
      const handle = rethrowTyped(() => decoder.next_frame());
      if (handle === undefined) break;
      yield buildFrame(handle);
    }

    reportErrors(decoder, options.onError, lastReportedSkipCount);
  } finally {
    options.signal?.removeEventListener('abort', onAbort);
    try {
      reader.releaseLock();
    } catch {
      /* reader may already be in a locked-cancel state */
    }
    try {
      decoder.free();
    } catch {
      /* best effort */
    }
  }
}

function reportErrors(
  decoder: WbgStreamingDecoder,
  onError: DecodeStreamOptions['onError'],
  previousCount: number,
): void {
  if (!onError) return;
  const skipped = decoder.skipped_count();
  if (skipped <= previousCount) return;
  const message = decoder.last_error() ?? 'unknown streaming error';
  onError({ message, skippedCount: skipped });
}

function buildFrame(handle: WbgDecodedFrame): DecodedFrame {
  const descriptor = rethrowTyped(() => handle.descriptor() as DataObjectDescriptor);
  const baseEntry = rethrowTyped(() => handle.base_entry() as BaseEntry | null);
  const byteLength = handle.byte_length();
  let closed = false;

  const frame: DecodedFrame = {
    descriptor,
    baseEntry,
    byteLength,
    data(): TypedArray {
      assertOpen(closed);
      const bytes = rethrowTyped(() => handle.data_u8());
      return typedArrayFor(descriptor.dtype, bytes, /* copy */ true);
    },
    dataView(): TypedArray {
      assertOpen(closed);
      const bytes = rethrowTyped(() => handle.data_u8());
      return typedArrayFor(descriptor.dtype, bytes, /* copy */ false);
    },
    close(): void {
      if (closed) return;
      closed = true;
      frameRegistry.unregister(frame);
      try {
        handle.free();
      } catch {
        /* best effort */
      }
    },
  };

  frameRegistry.register(frame, handle, frame);
  return frame;
}

function assertOpen(closed: boolean): void {
  if (closed) {
    throw new InvalidArgumentError(
      'decoded frame has been closed — payload access is no longer valid',
      'decoded frame has been closed — payload access is no longer valid',
    );
  }
}

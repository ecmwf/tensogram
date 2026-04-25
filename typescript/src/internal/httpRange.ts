// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * HTTP Range-request helpers used by the lazy TensogramFile backend.
 *
 * `@internal` — not re-exported from `index.ts`.
 */

import { IoError } from '../errors.js';
import { errorMessage, withCause } from './io.js';

export interface FetchRangeContext {
  url: string | URL;
  fetchFn: typeof globalThis.fetch;
  baseHeaders?: HeadersInit;
  signal?: AbortSignal;
}

/**
 * Issue a `Range: bytes=start-end-1` request covering `[start, end)`.
 *
 * - `requireRange: true` (used during the lazy scan) throws on any
 *   non-`206` status so the caller can fall back to eager download.
 * - `requireRange: false` (the default, used for payload fetches) also
 *   accepts `200` with the full body: when the server ignores the
 *   Range header we slice the response on the client side.
 *
 * `overrideSignal`, when supplied, takes precedence over
 * `ctx.signal` for the underlying `fetch`.  Used by the bidirectional
 * scan loop to drive a per-iteration child {@link AbortController}
 * that cancels the sibling Range fetch when one side fails, without
 * leaking that cancellation into post-open Range requests that still
 * honour the user's top-level signal.
 */
export async function fetchRange(
  ctx: FetchRangeContext,
  start: number,
  end: number,
  requireRange = false,
  overrideSignal?: AbortSignal,
): Promise<Uint8Array> {
  const headers = new Headers(ctx.baseHeaders);
  headers.set('Range', `bytes=${start}-${end - 1}`);
  const init: RequestInit = { method: 'GET', headers };
  const signal = overrideSignal ?? ctx.signal;
  if (signal !== undefined) init.signal = signal;

  let resp: Response;
  try {
    resp = await ctx.fetchFn(ctx.url, init);
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

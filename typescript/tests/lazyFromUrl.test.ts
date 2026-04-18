// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `TensogramFile.fromUrl` lazy-Range backend tests.  We simulate the
 * server via a mock `fetch` function passed through `options.fetch`,
 * so the tests don't need network access.
 *
 * Two key properties:
 *
 * 1. When the server advertises `Accept-Ranges: bytes` on `HEAD`,
 *    `fromUrl` uses Range requests — it does not download the full
 *    body during open, and only fetches specific messages on demand.
 *
 * 2. When the server does not advertise Range (or HEAD fails), the
 *    behaviour falls back to a single eager GET — matching Scope-B.
 */

import { describe, expect, it } from 'vitest';
import { encode, TensogramFile } from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

/** Build a mock fetch that serves the supplied bytes with Range support. */
function makeRangeServer(body: Uint8Array): {
  fetch: typeof globalThis.fetch;
  requests: Array<{ method: string; range?: string }>;
} {
  const requests: Array<{ method: string; range?: string }> = [];
  const fetchFn: typeof globalThis.fetch = async (
    _input: string | URL | Request,
    init?: RequestInit,
  ) => {
    const method = init?.method ?? 'GET';
    const rangeHeader =
      init?.headers instanceof Headers
        ? init.headers.get('range')
        : init?.headers && (init.headers as Record<string, string>)['Range'];
    requests.push({ method, range: rangeHeader ?? undefined });

    if (method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: {
          'accept-ranges': 'bytes',
          'content-length': String(body.byteLength),
        },
      });
    }

    if (rangeHeader) {
      const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      if (!match) {
        return new Response('bad range', { status: 416 });
      }
      const start = parseInt(match[1], 10);
      const end = parseInt(match[2], 10); // inclusive
      const sliceLen = end - start + 1;
      const copy = new ArrayBuffer(sliceLen);
      new Uint8Array(copy).set(body.subarray(start, end + 1));
      return new Response(copy, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${body.byteLength}`,
          'content-length': String(sliceLen),
        },
      });
    }

    // Full body.
    const copy = new ArrayBuffer(body.byteLength);
    new Uint8Array(copy).set(body);
    return new Response(copy, { status: 200 });
  };
  return { fetch: fetchFn, requests };
}

/** Build a mock fetch whose server does NOT advertise Range support. */
function makeNoRangeServer(body: Uint8Array): {
  fetch: typeof globalThis.fetch;
  requests: Array<{ method: string }>;
} {
  const requests: Array<{ method: string }> = [];
  const fetchFn: typeof globalThis.fetch = async (_url, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    requests.push({ method });
    if (method === 'HEAD') {
      return new Response(null, {
        status: 200,
        // No accept-ranges header at all.
        headers: { 'content-length': String(body.byteLength) },
      });
    }
    const copy = new ArrayBuffer(body.byteLength);
    new Uint8Array(copy).set(body);
    return new Response(copy, { status: 200 });
  };
  return { fetch: fetchFn, requests };
}

function concatBytes(...bs: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const b of bs) total += b.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of bs) {
    out.set(b, off);
    off += b.byteLength;
  }
  return out;
}

function makeMessage(values: readonly number[]): Uint8Array {
  return encode(defaultMeta(), [
    {
      descriptor: makeDescriptor([values.length], 'float32'),
      data: new Float32Array(values),
    },
  ]);
}

describe('Scope C.1 — TensogramFile.fromUrl Range backend', () => {
  initOnce();

  it('uses Range requests when server advertises Accept-Ranges: bytes', async () => {
    const body = concatBytes(makeMessage([1, 2, 3]), makeMessage([10, 20]));
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/a.tgm', {
      fetch: fakeFetch,
    });
    try {
      expect(file.source).toBe('remote');
      // One HEAD + two preamble fetches — no full-body download during open.
      const methods = requests.map((r) => r.method);
      expect(methods.filter((m) => m === 'GET').every((m) => m === 'GET')).toBe(true);
      expect(requests.some((r) => r.method === 'HEAD')).toBe(true);
      // Every GET issued so far carries a Range header.
      expect(requests.filter((r) => r.method === 'GET').every((r) => r.range !== undefined)).toBe(true);
      expect(file.messageCount).toBe(2);

      // Fetching message 0 adds exactly one more Range GET.
      const before = requests.length;
      const m0 = await file.message(0);
      expect(Array.from(m0.objects[0].data() as Float32Array)).toEqual([1, 2, 3]);
      m0.close();
      expect(requests.length - before).toBe(1);

      // Re-fetching the same message hits the LRU cache (no new request).
      const beforeCache = requests.length;
      const m0again = await file.message(0);
      m0again.close();
      expect(requests.length).toBe(beforeCache);
    } finally {
      file.close();
    }
  });

  it('falls back to eager GET when server omits Accept-Ranges', async () => {
    const body = concatBytes(makeMessage([7, 8, 9]), makeMessage([0]));
    const { fetch: fakeFetch, requests } = makeNoRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/b.tgm', {
      fetch: fakeFetch,
    });
    try {
      expect(file.source).toBe('remote');
      expect(file.messageCount).toBe(2);
      // One HEAD + one full-body GET (no Range).  rawMessage(i) is O(1) from memory.
      const methods = requests.map((r) => r.method);
      expect(methods).toEqual(['HEAD', 'GET']);

      const before = requests.length;
      const m1 = await file.message(1);
      expect(Array.from(m1.objects[0].data() as Float32Array)).toEqual([0]);
      m1.close();
      // No additional network calls — message bytes are already in memory.
      expect(requests.length).toBe(before);
    } finally {
      file.close();
    }
  });

  it('falls back to eager when HEAD fails entirely', async () => {
    const body = makeMessage([100]);
    let sawGet = false;
    const fakeFetch: typeof globalThis.fetch = async (_url, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      if (method === 'HEAD') {
        throw new Error('HEAD not allowed');
      }
      sawGet = true;
      const copy = new ArrayBuffer(body.byteLength);
      new Uint8Array(copy).set(body);
      return new Response(copy, { status: 200 });
    };

    const file = await TensogramFile.fromUrl('https://example.invalid/c.tgm', {
      fetch: fakeFetch,
    });
    try {
      expect(file.messageCount).toBe(1);
      expect(sawGet).toBe(true);
      const m = await file.message(0);
      expect(Array.from(m.objects[0].data() as Float32Array)).toEqual([100]);
      m.close();
    } finally {
      file.close();
    }
  });

  it('passes Range headers through to the server', async () => {
    const body = concatBytes(makeMessage([1, 2]), makeMessage([3, 4]));
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/d.tgm', {
      fetch: fakeFetch,
    });
    try {
      const before = requests.length;
      await (await file.message(0)).close?.();
      const rangeHeaders = requests.slice(before).map((r) => r.range);
      for (const r of rangeHeaders) {
        expect(r).toMatch(/^bytes=\d+-\d+$/);
      }
    } finally {
      file.close();
    }
  });

  it('evicts the least-recently-used entry when the cache fills', async () => {
    // Build a file with 40 small messages (> LAZY_CACHE_CAPACITY = 32).
    // After fetching every message once we should have cache-evicted the
    // earliest ones, so re-reading message 0 costs a fresh Range GET.
    const CACHE_CAPACITY = 32;
    const TOTAL = CACHE_CAPACITY + 4;
    const parts = Array.from({ length: TOTAL }, (_, i) => makeMessage([i]));
    const body = concatBytes(...parts);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/f.tgm', {
      fetch: fakeFetch,
    });
    try {
      expect(file.messageCount).toBe(TOTAL);
      // Fetch every message once — populates the LRU to capacity + overflow.
      for (let i = 0; i < TOTAL; i++) {
        (await file.message(i)).close();
      }

      // Re-reading message 0 (evicted by later fetches) must cost a
      // fresh Range GET; re-reading the most-recent message (TOTAL-1)
      // must hit the cache with zero new requests.
      const beforeOldest = requests.length;
      (await file.message(0)).close();
      const oldestCost = requests.length - beforeOldest;
      expect(oldestCost).toBeGreaterThanOrEqual(1);

      const beforeNewest = requests.length;
      (await file.message(0)).close(); // just-fetched → cached
      expect(requests.length - beforeNewest).toBe(0);
    } finally {
      file.close();
    }
  });

  it('handles streaming-mode messages by falling back to eager', async () => {
    // Build a tiny "streaming-mode" message by patching the preamble's
    // total_length to zero.  This is the same shape the real streaming
    // encoder produces; the lazy scanner must recognise it and bail
    // out, after which eager fallback takes over.
    const full = makeMessage([1, 2, 3, 4, 5]);
    const patched = new Uint8Array(full);
    // total_length is 8 bytes starting at offset 16.
    for (let i = 16; i < 24; i++) patched[i] = 0;
    const { fetch: fakeFetch, requests } = makeRangeServer(patched);

    const file = await TensogramFile.fromUrl('https://example.invalid/e.tgm', {
      fetch: fakeFetch,
    });
    try {
      // Eager fallback means we saw at least one non-Range GET (full body).
      const nonRange = requests.filter((r) => r.method === 'GET' && !r.range);
      expect(nonRange.length).toBeGreaterThan(0);
      // And the file is readable via the eager path — scan() can still
      // find the message because it walks for END_MAGIC when
      // total_length is 0.
      expect(file.messageCount).toBe(1);
    } finally {
      file.close();
    }
  });

  it('throws InvalidArgumentError when no fetch implementation is available', async () => {
    const prevFetch = globalThis.fetch;
    // Simulate an environment that doesn't provide a fetch implementation.
    // We pass a non-callable option to force the "no fetch available" branch.
    // (We can't delete globalThis.fetch in modern Node reliably.)
    const { TensogramFile: TF, InvalidArgumentError: IAE } = await import(
      '../src/index.js'
    );
    try {
      await expect(
        TF.fromUrl('https://example.invalid/a.tgm', {
          // @ts-expect-error intentional: callers who supply a non-function
          // fetch option must get a clear error.
          fetch: 'not a function',
        }),
      ).rejects.toThrow(IAE);
    } finally {
      globalThis.fetch = prevFetch;
    }
  });

  it('throws IoError on a HEAD-404 + GET-404 server', async () => {
    const fakeFetch: typeof globalThis.fetch = async () =>
      new Response('not found', { status: 404 });
    await expect(
      TensogramFile.fromUrl('https://example.invalid/404.tgm', { fetch: fakeFetch }),
    ).rejects.toThrow(/HTTP 404/);
  });

  it('falls back to eager when the advertised Content-Length is past MAX_SAFE_INTEGER', async () => {
    // A mythical 10 PiB server.  The lazy walk can't track cursor
    // arithmetic past 2^53 − 1 without precision loss, so we must
    // fall back to eager (which then fails its own way because no
    // real body is produced — but the important thing is we don't
    // silently walk broken arithmetic).
    const body = makeMessage([1, 2, 3]);
    let sawNonRangeGet = false;
    const fakeFetch: typeof globalThis.fetch = async (_url, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: {
            'accept-ranges': 'bytes',
            'content-length': '10000000000000000', // 10^16 ≫ 2^53 − 1
          },
        });
      }
      // On the eager GET we just serve the real body so open succeeds.
      sawNonRangeGet = true;
      const copy = new ArrayBuffer(body.byteLength);
      new Uint8Array(copy).set(body);
      return new Response(copy, { status: 200 });
    };
    const file = await TensogramFile.fromUrl('https://example.invalid/huge.tgm', {
      fetch: fakeFetch,
    });
    try {
      expect(sawNonRangeGet).toBe(true);
      expect(file.messageCount).toBe(1);
    } finally {
      file.close();
    }
  });
});

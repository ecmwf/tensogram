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
  requests: Array<{ method: string; range?: string; bytes?: number }>;
} {
  const requests: Array<{ method: string; range?: string; bytes?: number }> = [];
  const fetchFn: typeof globalThis.fetch = async (
    _input: string | URL | Request,
    init?: RequestInit,
  ) => {
    const method = init?.method ?? 'GET';
    const rangeHeader =
      init?.headers instanceof Headers
        ? init.headers.get('range')
        : init?.headers && (init.headers as Record<string, string>)['Range'];
    let bytes: number | undefined;
    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      if (m) bytes = parseInt(m[2], 10) - parseInt(m[1], 10) + 1;
    }
    requests.push({ method, range: rangeHeader ?? undefined, bytes });

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

/**
 * Range-capable fetch that also exposes a peak-in-flight metric.
 * Uses an artificial delay so concurrent requests overlap in time
 * and the limiter cap can be observed.
 */
function makeMetricRangeServer(body: Uint8Array): {
  fetch: typeof globalThis.fetch;
  peakInFlight: () => number;
  totalRequests: () => number;
} {
  let inFlight = 0;
  let peak = 0;
  let total = 0;
  const fetchFn: typeof globalThis.fetch = async (
    _input: string | URL | Request,
    init?: RequestInit,
  ) => {
    inFlight++;
    peak = Math.max(peak, inFlight);
    total++;
    try {
      // Tiny delay so the await actually overlaps requests.
      await new Promise((r) => setTimeout(r, 5));
      const method = init?.method ?? 'GET';
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: {
            'accept-ranges': 'bytes',
            'content-length': String(body.byteLength),
          },
        });
      }
      const range =
        init?.headers instanceof Headers ? init.headers.get('range') : undefined;
      if (range) {
        const m = /^bytes=(\d+)-(\d+)$/.exec(range);
        if (!m) return new Response('bad range', { status: 416 });
        const start = parseInt(m[1], 10);
        const end = parseInt(m[2], 10);
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
      const copy = new ArrayBuffer(body.byteLength);
      new Uint8Array(copy).set(body);
      return new Response(copy, { status: 200 });
    } finally {
      inFlight--;
    }
  };
  return {
    fetch: fetchFn,
    peakInFlight: () => peak,
    totalRequests: () => total,
  };
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

  it('messageMetadata fetches header chunk only (not the full message)', async () => {
    // Build a message > 256 KB so the header-chunk cap actually
    // kicks in.  100k float32s = 400 KB raw payload + ~100 B header
    // overhead — comfortably larger than HEADER_CHUNK_BYTES (256 KB).
    const big = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([100_000], 'float32'),
        data: new Float32Array(100_000),
      },
    ]);
    expect(big.byteLength).toBeGreaterThan(256 * 1024);
    const { fetch: fakeFetch, requests } = makeRangeServer(big);

    const file = await TensogramFile.fromUrl('https://example.invalid/big.tgm', {
      fetch: fakeFetch,
    });
    try {
      const before = requests.length;
      const meta = await file.messageMetadata(0);
      // `version` on decoded metadata is synthesised from the
      // preamble — always equal to the wire-format `WIRE_VERSION`
      // (currently 3), regardless of what the caller passed at encode
      // time.  See `plans/WIRE_FORMAT.md` §6.1 ("CBOR metadata is
      // free-form; the version lives in the preamble").
      expect(meta.version).toBe(3);
      const newRequests = requests.slice(before);
      expect(newRequests.length).toBeLessThanOrEqual(1);
      const totalNewBytes = newRequests.reduce((sum, r) => sum + (r.bytes ?? 0), 0);
      // Critically, the bytes fetched must be strictly less than the
      // full message size — otherwise the optimisation has regressed.
      expect(totalNewBytes).toBeLessThan(big.byteLength);
      expect(totalNewBytes).toBeLessThanOrEqual(256 * 1024);

      // Second call hits the layout cache: zero new requests.
      const beforeCached = requests.length;
      await file.messageMetadata(0);
      expect(requests.length).toBe(beforeCached);
    } finally {
      file.close();
    }
  });

  it('messageObject fetches only the target frame (not the full message)', async () => {
    // Two-object message; messageObject(0, 1) should fetch the second
    // frame's bytes only.
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([100_000], 'float32'),
        data: new Float32Array(100_000),
      },
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    const { fetch: fakeFetch, requests } = makeRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/two.tgm', {
      fetch: fakeFetch,
    });
    try {
      const before = requests.length;
      const decoded = await file.messageObject(0, 1);
      try {
        expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
      } finally {
        decoded.close();
      }
      const newRequests = requests.slice(before);
      const totalNewBytes = newRequests.reduce((sum, r) => sum + (r.bytes ?? 0), 0);
      // Tight object frame is ~32 bytes payload + framing; together
      // with the layout-discovery fetch, the total is well under
      // 256 KB and far less than the full message.
      expect(totalNewBytes).toBeLessThan(msg.byteLength);
    } finally {
      file.close();
    }
  });

  it('messageObject returns the real cached metadata (not a default placeholder)', async () => {
    // Regression: an earlier draft had messageObject returning a
    // DecodedMessage whose `metadata` was GlobalMetadata::default()
    // from the WASM side, not the message's real metadata.  This
    // test guards against that.
    const mars = { param: '2t', step: 0, date: '20260401' };
    const meta = {
      version: 3,
      base: [{ mars }, { mars }],
    };
    const msg = encode(meta, [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([5, 6, 7, 8]),
      },
    ]);
    const { fetch: fakeFetch } = makeRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/meta.tgm', {
      fetch: fakeFetch,
    });
    try {
      const decoded = await file.messageObject(0, 1);
      try {
        expect(decoded.metadata.version).toBe(3);
        expect(decoded.metadata.base).toBeDefined();
        expect(decoded.metadata.base?.length).toBe(2);
        const base0 = decoded.metadata.base?.[0] as Record<string, unknown>;
        expect((base0?.mars as Record<string, unknown>)?.param).toBe('2t');
      } finally {
        decoded.close();
      }
    } finally {
      file.close();
    }
  });

  it('messageObjectRange decodes partial ranges from one frame only', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([1_000_000], 'float32'),
        data: new Float32Array(1_000_000).map((_, i) => i),
      },
    ]);
    const { fetch: fakeFetch } = makeRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/range.tgm', {
      fetch: fakeFetch,
    });
    try {
      const result = await file.messageObjectRange(0, 0, [
        [10, 5],
        [100, 3],
      ]);
      expect(result.parts).toHaveLength(2);
      expect(Array.from(result.parts[0] as Float32Array)).toEqual([10, 11, 12, 13, 14]);
      expect(Array.from(result.parts[1] as Float32Array)).toEqual([100, 101, 102]);
    } finally {
      file.close();
    }
  });

  it('messageObjectRange supports join: true', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([100], 'float32'),
        data: new Float32Array(100).map((_, i) => i),
      },
    ]);
    const { fetch: fakeFetch } = makeRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/join.tgm', {
      fetch: fakeFetch,
    });
    try {
      const result = await file.messageObjectRange(
        0,
        0,
        [
          [0, 3],
          [10, 2],
        ],
        { join: true },
      );
      expect(result.parts).toHaveLength(1);
      expect(Array.from(result.parts[0] as Float32Array)).toEqual([0, 1, 2, 10, 11]);
    } finally {
      file.close();
    }
  });

  it('messageObjectBatch fans out with bounded concurrency', async () => {
    // Build a 6-message file; each message holds one tiny object.
    const msgs = Array.from({ length: 6 }, (_, i) =>
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([2], 'float32'),
          data: new Float32Array([i, i + 0.5]),
        },
      ]),
    );
    const body = concatBytes(...msgs);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/six.tgm', {
      fetch: fakeFetch,
      concurrency: 2,
    });
    try {
      const results = await file.messageObjectBatch([0, 1, 2, 3, 4, 5], 0);
      expect(results).toHaveLength(6);
      for (let i = 0; i < 6; i++) {
        expect(Array.from(results[i].objects[0].data() as Float32Array)).toEqual([i, i + 0.5]);
        results[i].close();
      }
      // Sanity: the batch made network activity (we're not just hitting cache).
      expect(requests.length).toBeGreaterThan(1);
    } finally {
      file.close();
    }
  });

  it('prefetchLayouts warms cache so messageMetadata is free afterwards', async () => {
    const msgs = Array.from({ length: 4 }, (_, i) =>
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([3], 'float32'),
          data: new Float32Array([i, i, i]),
        },
      ]),
    );
    const body = concatBytes(...msgs);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/prefetch.tgm', {
      fetch: fakeFetch,
    });
    try {
      const beforePrefetch = requests.length;
      await file.prefetchLayouts([0, 1, 2, 3]);
      const afterPrefetch = requests.length;
      expect(afterPrefetch).toBeGreaterThan(beforePrefetch);

      // Now messageMetadata for any of those indices must be free.
      const beforeMeta = requests.length;
      for (let i = 0; i < 4; i++) {
        const meta = await file.messageMetadata(i);
        // Synthetic `version` is sourced from the preamble — see the
        // longer note on the `messageMetadata fetches header chunk only`
        // test above.
        expect(meta.version).toBe(3);
      }
      expect(requests.length).toBe(beforeMeta);
    } finally {
      file.close();
    }
  });

  it('messageObjectBatch with concurrency = 1 still progresses (no nested-pool deadlock)', async () => {
    // Regression: an earlier draft routed both batch slots and inner
    // layout-discovery slots through the same shared backend limiter.
    // With concurrency = 1 (or any cap matching the batch size), all
    // slots got held by outer messageObject calls waiting for inner
    // layout fetches that could never enter the queue, deadlocking.
    // The current design uses an independent batch limiter so inner
    // calls always have the shared pool free.
    const msgs = Array.from({ length: 3 }, (_, i) =>
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([1], 'float32'),
          data: new Float32Array([i + 1]),
        },
      ]),
    );
    const body = concatBytes(...msgs);
    const { fetch: fakeFetch } = makeRangeServer(body);
    const file = await TensogramFile.fromUrl('https://example.invalid/three.tgm', {
      fetch: fakeFetch,
    });
    try {
      const results = await file.messageObjectBatch([0, 1, 2], 0, { concurrency: 1 });
      expect(results.map((r) => (r.objects[0].data() as Float32Array)[0])).toEqual([1, 2, 3]);
      results.forEach((r) => r.close());
    } finally {
      file.close();
    }
  });

  it('prefetchLayouts is a no-op on the eager backend', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    const { fetch: fakeFetch, requests } = makeNoRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/eager.tgm', {
      fetch: fakeFetch,
    });
    try {
      const before = requests.length;
      await file.prefetchLayouts([0]);
      // Eager backend already has all bytes in memory; prefetchLayouts
      // makes zero new HTTP calls.
      expect(requests.length).toBe(before);
    } finally {
      file.close();
    }
  });

  it('bails to eager when preamble advertises total_length below preamble+postamble', async () => {
    // A v3 preamble(24) + v3 postamble(24) = 48 bytes minimum. Any
    // lower `total_length` is impossible on the wire, so the lazy
    // scanner must refuse to walk.  We pick 40 specifically because it
    // matches the old (v2-era) minimum — this test would silently pass
    // under the stale POSTAMBLE_BYTES = 16 constant.
    const full = makeMessage([1, 2, 3, 4, 5]);
    const patched = new Uint8Array(full);
    // Rewrite total_length at offset 16 to 40 (big-endian u64).
    for (let i = 16; i < 24; i++) patched[i] = 0;
    patched[23] = 40;
    const { fetch: fakeFetch, requests } = makeRangeServer(patched);

    const file = await TensogramFile.fromUrl('https://example.invalid/tiny.tgm', {
      fetch: fakeFetch,
    });
    try {
      const nonRangeGets = requests.filter((r) => r.method === 'GET' && !r.range);
      expect(nonRangeGets.length).toBeGreaterThan(0);
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

  it('messageDescriptors returns one descriptor per object on the lazy backend', async () => {
    // Three objects of mixed shape — verifies the index-frame walk
    // and per-frame CBOR descriptor parse round-trip cleanly without
    // decoding payloads.  All three frames are below the prefix
    // threshold so the small-frame (full-frame fetch) path runs.
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
      {
        descriptor: makeDescriptor([2, 3], 'float32'),
        data: new Float32Array([1, 2, 3, 4, 5, 6]),
      },
      {
        descriptor: makeDescriptor([5], 'int32'),
        data: new Int32Array([10, 20, 30, 40, 50]),
      },
    ]);
    const { fetch: fakeFetch } = makeRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/desc.tgm', {
      fetch: fakeFetch,
    });
    try {
      const { metadata, descriptors } = await file.messageDescriptors(0);
      expect(metadata.version).toBe(3);
      expect(descriptors).toHaveLength(3);
      expect(descriptors[0].shape).toEqual([4]);
      expect(descriptors[0].dtype).toBe('float32');
      expect(descriptors[1].shape).toEqual([2, 3]);
      expect(descriptors[1].dtype).toBe('float32');
      expect(descriptors[2].shape).toEqual([5]);
      expect(descriptors[2].dtype).toBe('int32');

      // Cached on the layout: a second call makes zero descriptor-CBOR
      // round trips.  We can't directly observe the request count cheaply
      // here without a metrics fixture, but a re-call must succeed and
      // return shape-equal descriptors.
      const second = await file.messageDescriptors(0);
      expect(second.descriptors.map((d) => d.shape)).toEqual(
        descriptors.map((d) => d.shape),
      );
    } finally {
      file.close();
    }
  });

  it('messageDescriptors works on the eager backend without parse_footer_chunk error', async () => {
    // Regression: an earlier draft of messageDescriptors called
    // `parse_footer_chunk(slice)` on the *full* message bytes and
    // discarded the result.  That call could throw on valid messages
    // when stray "FR" sequences appeared inside compressed payload
    // bytes (e.g., szip / zstd output).  We force the eager backend
    // by supplying a server that omits Accept-Ranges, then verify
    // descriptors decode successfully even when the raw payload byte
    // pattern does contain "FR" markers — which we manufacture by
    // including 0x46 0x52 ("FR") in a Uint8 payload.
    const trickyPayload = new Uint8Array([
      0x46, 0x52, // "FR" — would have tripped the old eager-path bug
      0x00, 0x09, // type 9 (NTensorFrame) → looks frame-shaped to scanner
      0x00, 0x00, // version
      0x00, 0x00, // flags
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, // length 16 bytes — within bounds
      0xff, 0xff, // garbage to fill out
    ]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([trickyPayload.byteLength], 'uint8'), data: trickyPayload },
    ]);
    const { fetch: fakeFetch } = makeNoRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/eager-desc.tgm', {
      fetch: fakeFetch,
    });
    try {
      const { metadata, descriptors } = await file.messageDescriptors(0);
      expect(metadata.version).toBe(3);
      expect(descriptors).toHaveLength(1);
      expect(descriptors[0].dtype).toBe('uint8');
      expect(descriptors[0].shape).toEqual([trickyPayload.byteLength]);
    } finally {
      file.close();
    }
  });

  it('messageObjectRangeBatch decodes the same range across many messages', async () => {
    // Six messages, each with one float32 object of the same shape.
    // Batch-decode a partial range of each in parallel under a small
    // concurrency cap so the limiter actually serialises some calls.
    const N = 6;
    const msgs = Array.from({ length: N }, (_, i) =>
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([10], 'float32'),
          data: new Float32Array(10).map((_, k) => i * 100 + k),
        },
      ]),
    );
    const body = concatBytes(...msgs);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/range-batch.tgm', {
      fetch: fakeFetch,
      concurrency: 2,
    });
    try {
      const results = await file.messageObjectRangeBatch(
        Array.from({ length: N }, (_, i) => i),
        0,
        [
          [0, 2],
          [5, 3],
        ],
      );
      expect(results).toHaveLength(N);
      for (let i = 0; i < N; i++) {
        expect(results[i].parts).toHaveLength(2);
        const head = Array.from(results[i].parts[0] as Float32Array);
        expect(head).toEqual([i * 100 + 0, i * 100 + 1]);
        const middle = Array.from(results[i].parts[1] as Float32Array);
        expect(middle).toEqual([i * 100 + 5, i * 100 + 6, i * 100 + 7]);
      }
      // Sanity: the batch made at least one network call.
      expect(requests.length).toBeGreaterThan(1);
    } finally {
      file.close();
    }
  });

  it('messageDescriptors descriptor fan-out respects the concurrency cap', async () => {
    // Regression for the inner-fetchRange bypass: build a multi-object
    // message whose object frames each exceed DESCRIPTOR_PREFIX_THRESHOLD
    // (64 KB) so #fetchOneDescriptor takes the large-frame branch and
    // issues 3 raw fetches per descriptor (header + footer + CBOR).
    // Without routing those leaf fetches through `b.limit`, peak
    // in-flight would balloon to 6+ even at concurrency = 2.
    const N_OBJECTS = 4;
    const ELEMS = 20_000; // 20k float32 = 80 KB > 64 KB threshold
    const objects = Array.from({ length: N_OBJECTS }, (_, i) => ({
      descriptor: makeDescriptor([ELEMS], 'float32'),
      data: new Float32Array(ELEMS).map((_, k) => i * 100 + k),
    }));
    const msg = encode(defaultMeta(), objects);
    const server = makeMetricRangeServer(msg);
    const file = await TensogramFile.fromUrl('https://example.invalid/big-desc.tgm', {
      fetch: server.fetch,
      concurrency: 2,
    });
    try {
      const { descriptors } = await file.messageDescriptors(0);
      expect(descriptors).toHaveLength(N_OBJECTS);
      // Cap is 2: peak in-flight HTTP requests for the entire open
      // + descriptor fan-out must never exceed 2.  Without the
      // limiter wiring around the per-descriptor leaf fetches this
      // would hit 6+ (two outer descriptor tasks each firing
      // header + footer + CBOR in parallel).
      expect(server.peakInFlight()).toBeLessThanOrEqual(2);
    } finally {
      file.close();
    }
  });

  it('messageObjectRangeBatch supports join: true and per-call concurrency', async () => {
    // Smaller scenario but with the join option set, mirroring the
    // single-call `messageObjectRange supports join: true` test.
    const msgs = Array.from({ length: 3 }, (_, i) =>
      encode(defaultMeta(), [
        {
          descriptor: makeDescriptor([8], 'float32'),
          data: new Float32Array(8).map((_, k) => i * 10 + k),
        },
      ]),
    );
    const body = concatBytes(...msgs);
    const { fetch: fakeFetch } = makeRangeServer(body);
    const file = await TensogramFile.fromUrl('https://example.invalid/range-join.tgm', {
      fetch: fakeFetch,
    });
    try {
      const results = await file.messageObjectRangeBatch(
        [0, 1, 2],
        0,
        [
          [0, 2],
          [4, 2],
        ],
        { join: true, concurrency: 1 },
      );
      expect(results).toHaveLength(3);
      for (let i = 0; i < 3; i++) {
        expect(results[i].parts).toHaveLength(1);
        expect(Array.from(results[i].parts[0] as Float32Array)).toEqual([
          i * 10 + 0,
          i * 10 + 1,
          i * 10 + 4,
          i * 10 + 5,
        ]);
      }
    } finally {
      file.close();
    }
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

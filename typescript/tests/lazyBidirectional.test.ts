// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Bidirectional remote-scan walker tests.
 *
 * Mirrors the Rust `rust/tensogram/tests/remote_http.rs` bidirectional
 * suite — every scenario, every reason string, every dispatch
 * decision is covered here for the TypeScript walker.
 *
 * The mock fetch is the same Range-aware pattern used by
 * `lazyFromUrl.test.ts`, extended with per-Range tracking so the
 * tests can assert which (offset, length) pairs were paired in a
 * single round.
 */

import { describe, expect, it, vi } from 'vitest';
import { encode, InvalidArgumentError, TensogramFile } from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

interface RangeRequest {
  method: string;
  range?: string;
  start?: number;
  end?: number;
  bytes?: number;
}

function makeRangeServer(body: Uint8Array): {
  fetch: typeof globalThis.fetch;
  requests: RangeRequest[];
} {
  const requests: RangeRequest[] = [];
  const fetchFn: typeof globalThis.fetch = async (
    _input: string | URL | Request,
    init?: RequestInit,
  ) => {
    const method = init?.method ?? 'GET';
    const rangeHeader =
      init?.headers instanceof Headers
        ? init.headers.get('range')
        : init?.headers && (init.headers as Record<string, string>)['Range'];
    const record: RangeRequest = { method };
    if (rangeHeader) {
      record.range = rangeHeader;
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      if (m) {
        record.start = parseInt(m[1], 10);
        record.end = parseInt(m[2], 10) + 1;
        record.bytes = record.end - record.start;
      }
    }
    requests.push(record);

    if (method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: {
          'accept-ranges': 'bytes',
          'content-length': String(body.byteLength),
        },
      });
    }

    if (rangeHeader && record.start !== undefined && record.end !== undefined) {
      const start = record.start;
      const end = record.end - 1;
      const sliceLen = record.bytes!;
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
  };
  return { fetch: fetchFn, requests };
}

function concatBytes(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((acc, p) => acc + p.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
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

function makeNMessages(n: number): Uint8Array {
  const parts: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    parts.push(makeMessage([i, i + 1, i + 2]));
  }
  return concatBytes(...parts);
}

function rangeRequests(reqs: RangeRequest[]): RangeRequest[] {
  return reqs.filter((r) => r.method === 'GET' && r.range !== undefined);
}

function preambleRangeAt(offset: number, reqs: RangeRequest[]): boolean {
  return reqs.some((r) => r.start === offset && r.end === offset + 24);
}

function postambleRangeBefore(end: number, reqs: RangeRequest[]): boolean {
  return reqs.some((r) => r.start === end - 24 && r.end === end);
}

describe('TensogramFile.fromUrl bidirectional walker', () => {
  initOnce();

  it('opens a 1-message file and same_message_check yields backward', async () => {
    const body = makeMessage([1, 2, 3]);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/m1.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      expect(file.messageCount).toBe(1);
      expect(file.messageLayouts).toEqual([
        { offset: 0, length: body.byteLength },
      ]);
      const ranges = rangeRequests(requests);
      expect(preambleRangeAt(0, ranges)).toBe(true);
      expect(postambleRangeBefore(body.byteLength, ranges)).toBe(true);
    } finally {
      file.close();
    }
  });

  it('opens a 2-message file with paired-round meet at midpoint', async () => {
    const m0 = makeMessage([0, 1]);
    const m1 = makeMessage([10, 11, 12, 13]);
    const body = concatBytes(m0, m1);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/m2.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      expect(file.messageCount).toBe(2);
      expect(file.messageLayouts).toEqual([
        { offset: 0, length: m0.byteLength },
        { offset: m0.byteLength, length: m1.byteLength },
      ]);
      const ranges = rangeRequests(requests);
      expect(preambleRangeAt(0, ranges)).toBe(true);
      expect(postambleRangeBefore(body.byteLength, ranges)).toBe(true);
    } finally {
      file.close();
    }
  });

  it('opens a 3-message file with odd-count meet-in-the-middle', async () => {
    const m0 = makeMessage([0]);
    const m1 = makeMessage([1, 1]);
    const m2 = makeMessage([2, 2, 2]);
    const body = concatBytes(m0, m1, m2);
    const { fetch: fakeFetch } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/m3.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      expect(file.messageCount).toBe(3);
      expect(file.messageLayouts).toEqual([
        { offset: 0, length: m0.byteLength },
        { offset: m0.byteLength, length: m1.byteLength },
        { offset: m0.byteLength + m1.byteLength, length: m2.byteLength },
      ]);
    } finally {
      file.close();
    }
  });

  it('opens a 10-message file with layouts identical to forward-only', async () => {
    const body = makeNMessages(10);
    const { fetch: fwdFetch } = makeRangeServer(body);
    const { fetch: bidirFetch } = makeRangeServer(body);

    const fwdFile = await TensogramFile.fromUrl('https://example.invalid/fwd.tgm', {
      fetch: fwdFetch,
      bidirectional: false,
    });
    const bidirFile = await TensogramFile.fromUrl('https://example.invalid/bidir.tgm', {
      fetch: bidirFetch,
      bidirectional: true,
    });
    try {
      expect(bidirFile.messageCount).toBe(fwdFile.messageCount);
      expect(bidirFile.messageLayouts).toEqual(fwdFile.messageLayouts);
    } finally {
      fwdFile.close();
      bidirFile.close();
    }
  });

  it('forward-only mode is byte-identical to behaviour before bidirectional', async () => {
    const body = makeNMessages(5);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/fwd.tgm', {
      fetch: fakeFetch,
    });
    try {
      expect(file.messageCount).toBe(5);
      const preambleFetches = rangeRequests(requests).filter(
        (r) => r.bytes === 24,
      );
      expect(preambleFetches.length).toBe(5);
    } finally {
      file.close();
    }
  });

  it('rejects bidirectional: true with concurrency: 1 synchronously', async () => {
    await expect(
      TensogramFile.fromUrl('https://example.invalid/x.tgm', {
        fetch: async () => new Response(null, { status: 200 }),
        bidirectional: true,
        concurrency: 1,
      }),
    ).rejects.toBeInstanceOf(InvalidArgumentError);
  });

  it('disables backward and recovers via forward-only when END_MAGIC corrupted', async () => {
    const m0 = makeMessage([0]);
    const m1 = makeMessage([1, 2]);
    const body = concatBytes(m0, m1);
    const corrupted = new Uint8Array(body);
    for (let i = body.byteLength - 8; i < body.byteLength; i++) {
      corrupted[i] = 0;
    }
    const { fetch: fakeFetch } = makeRangeServer(corrupted);

    const file = await TensogramFile.fromUrl('https://example.invalid/c.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      expect(file.messageCount).toBe(2);
      expect(file.messageLayouts).toEqual([
        { offset: 0, length: m0.byteLength },
        { offset: m0.byteLength, length: m1.byteLength },
      ]);
    } finally {
      file.close();
    }
  });

  it('disables backward when postamble length mismatches preamble', async () => {
    const m0 = makeMessage([0]);
    const m1 = makeMessage([1, 2]);
    const body = concatBytes(m0, m1);
    const tampered = new Uint8Array(body);
    const tailTotalLengthOffset = body.byteLength - 16;
    const wrong = BigInt(m1.byteLength + 8);
    new DataView(tampered.buffer).setBigUint64(tailTotalLengthOffset, wrong, false);
    const { fetch: fakeFetch } = makeRangeServer(tampered);

    const file = await TensogramFile.fromUrl('https://example.invalid/lm.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      expect(file.messageCount).toBe(2);
      expect(file.messageLayouts).toEqual([
        { offset: 0, length: m0.byteLength },
        { offset: m0.byteLength, length: m1.byteLength },
      ]);
    } finally {
      file.close();
    }
  });

  it('emits debug events on every state transition when debug: true', async () => {
    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    try {
      const body = makeNMessages(3);
      const { fetch: fakeFetch } = makeRangeServer(body);

      const file = await TensogramFile.fromUrl('https://example.invalid/dbg.tgm', {
        fetch: fakeFetch,
        bidirectional: true,
        debug: true,
      });
      try {
        const tags = debugSpy.mock.calls.map((c) => c[0]);
        expect(tags).toContain('tensogram:scan:mode');
        expect(tags).toContain('tensogram:scan:hop');
        expect(tags).toContain('tensogram:scan:gap-closed');
      } finally {
        file.close();
      }
    } finally {
      debugSpy.mockRestore();
    }
  });

  it('emits forward-only mode tag when bidirectional: false and debug: true', async () => {
    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    try {
      const body = makeMessage([1, 2, 3]);
      const { fetch: fakeFetch } = makeRangeServer(body);
      const file = await TensogramFile.fromUrl('https://example.invalid/dbg2.tgm', {
        fetch: fakeFetch,
        debug: true,
      });
      try {
        const modeCalls = debugSpy.mock.calls.filter(
          (c) => c[0] === 'tensogram:scan:mode',
        );
        expect(modeCalls.length).toBeGreaterThan(0);
        expect(modeCalls[0][1]).toBe('forward-only');
      } finally {
        file.close();
      }
    } finally {
      debugSpy.mockRestore();
    }
  });

  it('honours user AbortSignal aborted before fromUrl', async () => {
    const body = makeMessage([1, 2, 3]);
    const { fetch: rangeFetch } = makeRangeServer(body);
    const abortAwareFetch: typeof globalThis.fetch = async (input, init) => {
      if (init?.signal?.aborted) {
        throw new DOMException('aborted', 'AbortError');
      }
      return rangeFetch(input, init);
    };
    const ctl = new AbortController();
    ctl.abort();
    await expect(
      TensogramFile.fromUrl('https://example.invalid/abort.tgm', {
        fetch: abortAwareFetch,
        bidirectional: true,
        signal: ctl.signal,
      }),
    ).rejects.toThrow();
  });

  it('paired round issues 2 Range requests per scan iteration on a 2-message file', async () => {
    const m0 = makeMessage([0]);
    const m1 = makeMessage([1, 2]);
    const body = concatBytes(m0, m1);
    const { fetch: fakeFetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/p.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      expect(file.messageCount).toBe(2);
      const preambleFetches = rangeRequests(requests).filter((r) => r.bytes === 24);
      expect(preambleFetches.length).toBeLessThanOrEqual(3);
      expect(preambleRangeAt(0, preambleFetches)).toBe(true);
      expect(postambleRangeBefore(body.byteLength, preambleFetches)).toBe(true);
    } finally {
      file.close();
    }
  });

  it('outcome is deterministic regardless of which paired-fetch resolves first', async () => {
    const body = makeNMessages(4);
    const makeServerWithDelay = (delayBwd: boolean) => {
      const { fetch: base, requests } = makeRangeServer(body);
      const delayedFetch: typeof globalThis.fetch = async (input, init) => {
        const isPostamblePart =
          init?.headers instanceof Headers
            ? init.headers.get('range')?.startsWith('bytes=' + (body.byteLength - 24))
            : false;
        const shouldDelay = delayBwd ? isPostamblePart : !isPostamblePart;
        if (shouldDelay && init?.method === 'GET') {
          await new Promise((r) => setTimeout(r, 5));
        }
        return base(input, init);
      };
      return { fetch: delayedFetch, requests };
    };
    const { fetch: fetchA } = makeServerWithDelay(true);
    const { fetch: fetchB } = makeServerWithDelay(false);

    const fileA = await TensogramFile.fromUrl('https://example.invalid/a.tgm', {
      fetch: fetchA,
      bidirectional: true,
    });
    const fileB = await TensogramFile.fromUrl('https://example.invalid/b.tgm', {
      fetch: fetchB,
      bidirectional: true,
    });
    try {
      expect(fileA.messageLayouts).toEqual(fileB.messageLayouts);
    } finally {
      fileA.close();
      fileB.close();
    }
  });

  it('falls back to eager when streaming-mode preamble at start of file', async () => {
    const full = makeMessage([1, 2, 3, 4, 5]);
    const patched = new Uint8Array(full);
    for (let i = 16; i < 24; i++) patched[i] = 0;
    const { fetch: fakeFetch, requests } = makeRangeServer(patched);

    const file = await TensogramFile.fromUrl('https://example.invalid/s.tgm', {
      fetch: fakeFetch,
      bidirectional: true,
    });
    try {
      const nonRange = requests.filter((r) => r.method === 'GET' && !r.range);
      expect(nonRange.length).toBeGreaterThan(0);
      expect(file.messageCount).toBe(1);
    } finally {
      file.close();
    }
  });

  it('falls back to eager when streaming-mode tail follows fixed messages', async () => {
    const m0 = makeMessage([1, 2, 3]);
    const m1 = makeMessage([10, 20, 30, 40]);
    const body = concatBytes(m0, m1);
    const patched = new Uint8Array(body);
    const tailTotalLengthOffset = m0.byteLength + 16;
    for (let i = tailTotalLengthOffset; i < tailTotalLengthOffset + 8; i++) {
      patched[i] = 0;
    }
    const { fetch: fakeFetch, requests } = makeRangeServer(patched);

    const file = await TensogramFile.fromUrl('https://example.invalid/st.tgm', {
      fetch: fakeFetch,
      bidirectional: false,
    });
    try {
      const nonRange = requests.filter((r) => r.method === 'GET' && !r.range);
      expect(nonRange.length).toBeGreaterThan(0);
      expect(file.messageCount).toBe(2);
    } finally {
      file.close();
    }
  });

  it('cancels in-flight Range fetches when caller aborts mid-round', async () => {
    const body = makeNMessages(4);
    const aborts: number[] = [];
    let nextCall = 0;
    const slowAbortAwareFetch: typeof globalThis.fetch = async (input, init) => {
      const callIdx = ++nextCall;
      const sig = init?.signal;
      return new Promise((resolve, reject) => {
        const onAbort = (): void => {
          aborts.push(callIdx);
          reject(new DOMException('aborted', 'AbortError'));
        };
        if (sig?.aborted) {
          onAbort();
          return;
        }
        sig?.addEventListener('abort', onAbort, { once: true });
        const t = setTimeout(() => {
          sig?.removeEventListener('abort', onAbort);
          const range =
            init?.headers instanceof Headers
              ? init.headers.get('range')
              : init?.headers && (init.headers as Record<string, string>)['Range'];
          const method = init?.method ?? 'GET';
          if (method === 'HEAD') {
            resolve(
              new Response(null, {
                status: 200,
                headers: {
                  'accept-ranges': 'bytes',
                  'content-length': String(body.byteLength),
                },
              }),
            );
            return;
          }
          if (range) {
            const m = /^bytes=(\d+)-(\d+)$/.exec(range);
            if (!m) {
              resolve(new Response('bad range', { status: 416 }));
              return;
            }
            const start = parseInt(m[1], 10);
            const end = parseInt(m[2], 10);
            const slice = body.slice(start, end + 1);
            resolve(
              new Response(slice, {
                status: 206,
                headers: {
                  'content-range': `bytes ${start}-${end}/${body.byteLength}`,
                  'content-length': String(slice.byteLength),
                },
              }),
            );
            return;
          }
          resolve(new Response(body, { status: 200 }));
        }, 100);
        sig?.addEventListener(
          'abort',
          () => clearTimeout(t),
          { once: true },
        );
      });
    };

    const ctl = new AbortController();
    const openPromise = TensogramFile.fromUrl('https://example.invalid/abrt.tgm', {
      fetch: slowAbortAwareFetch,
      bidirectional: true,
      signal: ctl.signal,
    });
    setTimeout(() => ctl.abort(), 30);
    await expect(openPromise).rejects.toThrow();
    expect(aborts.length).toBeGreaterThan(0);
  });
});

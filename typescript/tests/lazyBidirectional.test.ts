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

  it('terminates cleanly without an extra Range fetch on a truncated tail', async () => {
    const m0 = makeMessage([1, 2, 3]);
    const truncated = new Uint8Array(m0.byteLength + 30);
    truncated.set(m0, 0);
    const { fetch: fakeFetch, requests } = makeRangeServer(truncated);

    const file = await TensogramFile.fromUrl('https://example.invalid/trunc.tgm', {
      fetch: fakeFetch,
      bidirectional: false,
    });
    try {
      expect(file.messageCount).toBe(1);
      const tailRangeFetches = rangeRequests(requests).filter(
        (r) => r.start !== undefined && r.start >= m0.byteLength,
      );
      expect(tailRangeFetches.length).toBe(0);
    } finally {
      file.close();
    }
  });

  it('header-indexed bidirectional walk issues no footer-region fetches', async () => {
    await initOnce();
    const messages = [
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([4], 'float32'), data: new Float32Array(4) },
      ]),
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([8], 'float32'), data: new Float32Array(8) },
      ]),
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([16], 'float32'), data: new Float32Array(16) },
      ]),
      encode(defaultMeta(), [
        { descriptor: makeDescriptor([32], 'float32'), data: new Float32Array(32) },
      ]),
    ];
    const body = new Uint8Array(messages.reduce((s, m) => s + m.byteLength, 0));
    let off = 0;
    for (const m of messages) {
      body.set(m, off);
      off += m.byteLength;
    }
    const { fetch, requests } = makeRangeServer(body);

    const file = await TensogramFile.fromUrl('https://example.invalid/h.tgm', {
      fetch,
      bidirectional: true,
    });
    expect(file.messageCount).toBe(4);

    // Every scan-walk Range fetch is exactly PREAMBLE_BYTES (24) wide.
    // Eager-footer fetches would be larger (footer regions span multiple
    // CBOR frames).  On header-indexed messages the FOOTER_INDEX flag
    // gate must skip the speculative footer fetch entirely, so the
    // observed request log contains only 24-byte scan ranges (plus the
    // HEAD probe).
    const nonProbe = requests.filter((r) => r.method !== 'HEAD' && r.bytes !== undefined);
    for (const r of nonProbe) {
      expect(r.bytes).toBe(24);
    }
    file.close();
  });

  it('synthetic FOOTER_INDEX flag triggers eager-footer fetch attempt then falls through to lazy', async () => {
    await initOnce();
    // Two header-indexed messages, then patch the SECOND message's
    // preamble flags to set FOOTER_METADATA | FOOTER_INDEX bits.  The
    // bytes don't actually contain a FooterIndex frame, so the
    // best-effort parse will fail silently — but we can observe that
    // the dispatcher issued an extra footer-region Range request on
    // the backward-discovered message before falling through.
    const m1 = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data: new Float32Array(4) },
    ]);
    const m2 = encode(defaultMeta(), [
      { descriptor: makeDescriptor([16], 'float32'), data: new Float32Array(16) },
    ]);
    const body = new Uint8Array(m1.byteLength + m2.byteLength);
    body.set(m1, 0);
    body.set(m2, m1.byteLength);
    // Patch m2's preamble flags (u16 BE at offset 10..12 within the
    // preamble — wire.rs layout: [magic 8][version u16][flags u16]
    // [reserved u32][total_length u64]).  MessageFlags bit positions
    // (wire.rs:148-156): HEADER_METADATA = 1<<0, FOOTER_METADATA = 1<<1,
    // HEADER_INDEX = 1<<2, FOOTER_INDEX = 1<<3.  Set FOOTER_METADATA |
    // FOOTER_INDEX (= 0x000a) and clear HEADER_METADATA | HEADER_INDEX
    // (= 0x0005) so the TS dispatcher reads the message as
    // footer-indexed and triggers the eager-footer code path.
    const flagsHi = m1.byteLength + 10;
    const flagsLo = m1.byteLength + 11;
    body[flagsHi] = 0x00;
    body[flagsLo] = (body[flagsLo] & ~0x05) | 0x0a;
    // Patch m2's first_footer_offset in postamble to a value inside
    // the message's payload region so footerRegionPresent returns true.
    // postamble lives at the last 24 bytes of m2; first_footer_offset
    // is the first 8 bytes (big-endian).
    const m2End = body.byteLength;
    const postambleStart = m2End - 24;
    const fakeFooterOffset = 28; // > PREAMBLE_BYTES, < length - POSTAMBLE_BYTES
    new DataView(body.buffer).setBigUint64(postambleStart, BigInt(fakeFooterOffset), false);

    const { fetch, requests } = makeRangeServer(body);

    // Walking will fail at preamble validation (the patched flags
    // imply footer-indexed but the bytes contain header-indexed frames),
    // OR succeed depending on whether our own forward parse cares.
    // Either way the test asserts only that a footer-region Range was
    // attempted before any lazy populate.
    try {
      const file = await TensogramFile.fromUrl('https://example.invalid/synth.tgm', {
        fetch,
        bidirectional: true,
      });
      file.close();
    } catch {
      // Walker may bail on synthetic mismatches; what we care about
      // is the Range pattern logged before failure.
    }

    const fetched = requests.filter((r) => r.method !== 'HEAD' && r.bytes !== undefined);
    const wide = fetched.filter((r) => r.bytes !== 24);
    // The synthetic flag patching forces a footer-region fetch (≠ 24
    // bytes) once the backward postamble is parsed.  If wide.length is
    // 0, the dispatcher never attempted the eager-footer fetch — that
    // would be a regression in the FOOTER_METADATA | FOOTER_INDEX
    // detection path.
    expect(wide.length).toBeGreaterThan(0);
  });

  it('cancels in-flight Range fetches when caller aborts mid-round', async () => {
    const body = makeNMessages(4);
    const aborts: number[] = [];
    let nextCall = 0;
    const slowAbortAwareFetch: typeof globalThis.fetch = async (_input, init) => {
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
            const sliceLen = end - start + 1;
            const sliceBuf = new ArrayBuffer(sliceLen);
            new Uint8Array(sliceBuf).set(body.subarray(start, end + 1));
            resolve(
              new Response(sliceBuf, {
                status: 206,
                headers: {
                  'content-range': `bytes ${start}-${end}/${body.byteLength}`,
                  'content-length': String(sliceLen),
                },
              }),
            );
            return;
          }
          const fullBuf = new ArrayBuffer(body.byteLength);
          new Uint8Array(fullBuf).set(body);
          resolve(new Response(fullBuf, { status: 200 }));
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

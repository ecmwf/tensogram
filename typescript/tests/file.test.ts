// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  decode,
  encode,
  init,
  InvalidArgumentError,
  IoError,
  ObjectError,
  TensogramFile,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

/**
 * Build a `Response` whose body is the given bytes. We copy into a
 * fresh ArrayBuffer to satisfy the newer `BodyInit` typing, which
 * expects a concrete `BufferSource` rather than `Uint8Array<ArrayBufferLike>`.
 */
function responseFrom(bytes: Uint8Array, init?: ResponseInit): Response {
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  return new Response(copy, init);
}

function makeMessage(values: readonly number[]): Uint8Array {
  return encode(defaultMeta(), [
    {
      descriptor: makeDescriptor([values.length], 'float32'),
      data: new Float32Array(values),
    },
  ]);
}

function concatMessages(...msgs: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const m of msgs) total += m.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const m of msgs) {
    out.set(m, off);
    off += m.byteLength;
  }
  return out;
}

describe('Phase 4 — TensogramFile', () => {
  describe('fromBytes', () => {
    it('indexes a single-message buffer', async () => {
      await init();
      const msg = makeMessage([1, 2, 3, 4]);
      const file = TensogramFile.fromBytes(msg);
      expect(file.messageCount).toBe(1);
      expect(file.source).toBe('buffer');
      expect(file.byteLength).toBe(msg.byteLength);
      file.close();
    });

    it('indexes a multi-message buffer', async () => {
      await init();
      const combined = concatMessages(
        makeMessage([1, 2, 3]),
        makeMessage([4, 5]),
        makeMessage([6, 7, 8, 9]),
      );
      const file = TensogramFile.fromBytes(combined);
      expect(file.messageCount).toBe(3);
    });

    it('decodes each message independently', async () => {
      await init();
      const combined = concatMessages(makeMessage([1, 2]), makeMessage([10, 20, 30]));
      const file = TensogramFile.fromBytes(combined);

      const m0 = await file.message(0);
      expect(Array.from(m0.objects[0].data() as Float32Array)).toEqual([1, 2]);
      m0.close();

      const m1 = await file.message(1);
      expect(Array.from(m1.objects[0].data() as Float32Array)).toEqual([10, 20, 30]);
      m1.close();
    });

    it('throws ObjectError for out-of-range indices', async () => {
      await init();
      const file = TensogramFile.fromBytes(makeMessage([1]));
      // rawMessage is async now (Scope C) — sync throws become rejections.
      await expect(file.rawMessage(5)).rejects.toThrow(ObjectError);
      await expect(file.message(-1)).rejects.toThrow(ObjectError);
      await expect(file.message(1.5)).rejects.toThrow(ObjectError);
    });

    it('rawMessage returns bytes that re-decode to the same message', async () => {
      await init();
      const msg = makeMessage([42, 43, 44]);
      const file = TensogramFile.fromBytes(concatMessages(msg, makeMessage([99])));

      const raw = await file.rawMessage(0);
      const decoded = decode(new Uint8Array(raw));
      expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([42, 43, 44]);
      decoded.close();
    });

    it('rejects non-Uint8Array inputs', () => {
      expect(() =>
        TensogramFile.fromBytes(
          // @ts-expect-error intentional bad input
          [1, 2, 3],
        ),
      ).toThrow(InvalidArgumentError);
    });

    it('messageMetadata returns metadata without touching payload', async () => {
      await init();
      const file = TensogramFile.fromBytes(makeMessage([1, 2, 3]));
      const meta = await file.messageMetadata(0);
      // Wire version is carried in the preamble; v3 is fixed at 3.
      expect(meta.version).toBe(3);
    });

    it('async iteration yields every message', async () => {
      await init();
      const file = TensogramFile.fromBytes(
        concatMessages(
          makeMessage([1]),
          makeMessage([2, 3]),
          makeMessage([4, 5, 6]),
          makeMessage([7, 8, 9, 10]),
        ),
      );

      const allDecoded: number[][] = [];
      for await (const msg of file) {
        allDecoded.push(Array.from(msg.objects[0].data() as Float32Array));
        msg.close();
      }
      expect(allDecoded).toEqual([[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]]);
    });

    it('close() blocks further access', async () => {
      await init();
      const file = TensogramFile.fromBytes(makeMessage([1]));
      file.close();
      await expect(file.rawMessage(0)).rejects.toThrow(InvalidArgumentError);
      await expect(file.message(0)).rejects.toThrow(InvalidArgumentError);
    });

    it('close() is idempotent', async () => {
      await init();
      const file = TensogramFile.fromBytes(makeMessage([1]));
      expect(() => file.close()).not.toThrow();
      expect(() => file.close()).not.toThrow();
    });

    it('defensively copies the input so caller mutation is invisible', async () => {
      await init();
      const msg = makeMessage([1, 2, 3]);
      const buf = new Uint8Array(msg);
      const file = TensogramFile.fromBytes(buf);

      // Clobber the caller's buffer completely.
      for (let i = 0; i < buf.byteLength; i++) buf[i] = 0;

      const m = await file.message(0);
      expect(Array.from(m.objects[0].data() as Float32Array)).toEqual([1, 2, 3]);
      m.close();
    });
  });

  describe('open (Node filesystem)', () => {
    let tmp: string;
    beforeEach(() => {
      tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-'));
    });
    afterEach(() => {
      rmSync(tmp, { recursive: true, force: true });
    });

    it('opens and reads a temp file', async () => {
      await init();
      const path = join(tmp, 'sample.tgm');
      const msg = makeMessage([100, 200, 300]);
      writeFileSync(path, msg);

      const file = await TensogramFile.open(path);
      try {
        expect(file.source).toBe('local');
        expect(file.messageCount).toBe(1);
        const m = await file.message(0);
        expect(Array.from(m.objects[0].data() as Float32Array)).toEqual([100, 200, 300]);
        m.close();
      } finally {
        file.close();
      }
    });

    it('opens a multi-message file and iterates', async () => {
      await init();
      const path = join(tmp, 'multi.tgm');
      const combined = concatMessages(
        makeMessage([1, 2]),
        makeMessage([3, 4]),
        makeMessage([5, 6]),
      );
      writeFileSync(path, combined);

      const file = await TensogramFile.open(path);
      try {
        const seen: number[][] = [];
        for await (const m of file) {
          seen.push(Array.from(m.objects[0].data() as Float32Array));
          m.close();
        }
        expect(seen).toEqual([[1, 2], [3, 4], [5, 6]]);
      } finally {
        file.close();
      }
    });

    it('rejects non-string / non-URL paths', async () => {
      await expect(
        // @ts-expect-error intentional
        TensogramFile.open(42),
      ).rejects.toThrow(InvalidArgumentError);
    });

    it('throws IoError on missing file', async () => {
      await init();
      await expect(
        TensogramFile.open(join(tmp, 'does-not-exist.tgm')),
      ).rejects.toThrow(IoError);
    });
  });

  describe('fromUrl (fetch-based)', () => {
    it('uses a custom fetch and reads the returned bytes', async () => {
      await init();
      const msg = makeMessage([1, 2, 3, 4]);
      const fakeFetch: typeof globalThis.fetch = async (
        _url: string | URL | Request,
        _init?: RequestInit,
      ) => {
        return responseFrom(msg, { status: 200, statusText: 'OK' });
      };
      const file = await TensogramFile.fromUrl('https://example.invalid/f.tgm', {
        fetch: fakeFetch,
      });
      try {
        expect(file.source).toBe('remote');
        expect(file.messageCount).toBe(1);
        const m = await file.message(0);
        expect(Array.from(m.objects[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
        m.close();
      } finally {
        file.close();
      }
    });

    it('forwards headers and signal to fetch', async () => {
      await init();
      const msg = makeMessage([1]);
      let sawUrl: string | undefined;
      let sawHeaders: HeadersInit | undefined;
      let sawSignal: AbortSignal | undefined;
      const fakeFetch: typeof globalThis.fetch = async (url, init) => {
        sawUrl = String(url);
        sawHeaders = init?.headers;
        sawSignal = init?.signal ?? undefined;
        return responseFrom(msg, { status: 200 });
      };
      const controller = new AbortController();
      await TensogramFile.fromUrl('https://example.invalid/a', {
        fetch: fakeFetch,
        headers: { 'x-auth': 'demo' },
        signal: controller.signal,
      });
      expect(sawUrl).toContain('example.invalid');
      expect((sawHeaders as Record<string, string> | undefined)?.['x-auth']).toBe('demo');
      expect(sawSignal).toBeDefined();
    });

    it('throws IoError on non-2xx HTTP status', async () => {
      await init();
      const fakeFetch: typeof globalThis.fetch = async () =>
        new Response('not found' as string, { status: 404, statusText: 'Not Found' });
      await expect(
        TensogramFile.fromUrl('https://example.invalid/missing', { fetch: fakeFetch }),
      ).rejects.toThrow(IoError);
    });

    it('wraps fetch errors in IoError', async () => {
      await init();
      const fakeFetch: typeof globalThis.fetch = async () => {
        throw new Error('TCP reset');
      };
      await expect(
        TensogramFile.fromUrl('https://example.invalid/a', { fetch: fakeFetch }),
      ).rejects.toThrow(IoError);
    });

    it('throws InvalidArgumentError when no fetch is available', async () => {
      await init();
      // Simulate a fetch-less runtime by temporarily hiding globalThis.fetch.
      const savedFetch = globalThis.fetch;
      (globalThis as { fetch?: typeof fetch }).fetch = undefined;
      try {
        await expect(
          TensogramFile.fromUrl('https://example.invalid/a'),
        ).rejects.toThrow(InvalidArgumentError);
      } finally {
        globalThis.fetch = savedFetch;
      }
    });
  });
});

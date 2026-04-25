// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Smoke test for the bidirectional remote-scan walker behind the Vite
 * dev-server CORS proxy.
 *
 * The proxy at `tensoscope/vite.config.ts:20-72` forwards method +
 * `Range` header pass-through and `await`s each upstream `fetch`
 * independently — there is no serialising queue.  This test pins
 * that contract by simulating the proxy's forwarding semantics
 * inline (a fetch that just delegates to a Range-aware mock) and
 * verifies that {@link TensogramFile.fromUrl} with `bidirectional:
 * true` succeeds end-to-end.
 *
 * If the proxy is ever changed in a way that serialises paired Range
 * requests (or strips the Range header), a real integration test
 * would catch it; this lightweight guard catches the wrapper-level
 * regressions.
 */

import { describe, expect, it } from 'vitest';
import { encode, init, TensogramFile } from '@ecmwf.int/tensogram';

function makeRangeAwareFetch(body: Uint8Array): typeof globalThis.fetch {
  return async (_input, init) => {
    const method = init?.method ?? 'GET';
    const range =
      init?.headers instanceof Headers
        ? init.headers.get('range')
        : init?.headers && (init.headers as Record<string, string>)['Range'];
    if (method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: {
          'accept-ranges': 'bytes',
          'content-length': String(body.byteLength),
        },
      });
    }
    if (range) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(range);
      if (!m) return new Response('bad range', { status: 416 });
      const start = parseInt(m[1], 10);
      const end = parseInt(m[2], 10);
      const slice = body.slice(start, end + 1);
      return new Response(slice, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${body.byteLength}`,
          'content-length': String(slice.byteLength),
        },
      });
    }
    return new Response(body, { status: 200 });
  };
}

describe('Tensoscope bidirectional smoke test', () => {
  it('opens a 3-message file via a proxy-shape fetch with bidirectional: true', async () => {
    await init();
    const m0 = encode({}, [
      {
        descriptor: {
          type: 'ntensor',
          ndim: 1,
          shape: [3],
          strides: [1],
          dtype: 'float32',
          byte_order: 'little',
          encoding: 'none',
          filter: 'none',
          compression: 'none',
        },
        data: new Float32Array([1, 2, 3]),
      },
    ]);
    const m1 = encode({}, [
      {
        descriptor: {
          type: 'ntensor',
          ndim: 1,
          shape: [4],
          strides: [1],
          dtype: 'float32',
          byte_order: 'little',
          encoding: 'none',
          filter: 'none',
          compression: 'none',
        },
        data: new Float32Array([10, 20, 30, 40]),
      },
    ]);
    const m2 = encode({}, [
      {
        descriptor: {
          type: 'ntensor',
          ndim: 1,
          shape: [2],
          strides: [1],
          dtype: 'float32',
          byte_order: 'little',
          encoding: 'none',
          filter: 'none',
          compression: 'none',
        },
        data: new Float32Array([100, 200]),
      },
    ]);
    const total = m0.byteLength + m1.byteLength + m2.byteLength;
    const body = new Uint8Array(total);
    body.set(m0, 0);
    body.set(m1, m0.byteLength);
    body.set(m2, m0.byteLength + m1.byteLength);

    const proxyShapeFetch = makeRangeAwareFetch(body);
    const file = await TensogramFile.fromUrl(
      'http://localhost:5173/api/proxy?url=https://example.invalid/data.tgm',
      { fetch: proxyShapeFetch, bidirectional: true },
    );
    try {
      expect(file.messageCount).toBe(3);
      expect(file.messageLayouts.length).toBe(3);
      expect(file.messageLayouts[0].offset).toBe(0);
      expect(file.messageLayouts[1].offset).toBe(m0.byteLength);
      expect(file.messageLayouts[2].offset).toBe(m0.byteLength + m1.byteLength);
    } finally {
      file.close();
    }
  });
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 13 — Lazy `TensogramFile.fromUrl` over HTTP Range (TypeScript)
 *
 * When the server advertises `Accept-Ranges: bytes` and a finite
 * `Content-Length` on `HEAD`, `TensogramFile.fromUrl` uses Range
 * requests: message payloads are fetched on demand, not during open.
 * The example uses a mock `fetch` so it runs anywhere — swap it for
 * `globalThis.fetch` to drive a real server.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 13_range_access.ts
 */

import {
  encode,
  init,
  TensogramFile,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

function describe(shape: number[], dtype: DataObjectDescriptor['dtype']): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype,
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
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

function makeRangeServer(body: Uint8Array): {
  fetch: typeof globalThis.fetch;
  requestCount: { value: number };
} {
  const requestCount = { value: 0 };
  const fetchFn: typeof globalThis.fetch = async (_url, init?: RequestInit) => {
    requestCount.value++;
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

    const rangeHeader =
      init?.headers instanceof Headers
        ? init.headers.get('range')
        : (init?.headers as Record<string, string> | undefined)?.['Range'];
    if (rangeHeader) {
      const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      if (!match) return new Response('bad range', { status: 416 });
      const start = parseInt(match[1], 10);
      const end = parseInt(match[2], 10);
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
    // Full body fallback (not reached when the client uses Range).
    const copy = new ArrayBuffer(body.byteLength);
    new Uint8Array(copy).set(body);
    return new Response(copy, { status: 200 });
  };
  return { fetch: fetchFn, requestCount };
}

async function main(): Promise<void> {
  await init();

  // Build a "server body" of three messages concatenated.
  const messages = [
    encode({ version: 3 }, [{ descriptor: describe([3], 'float32'), data: new Float32Array([1, 2, 3]) }]),
    encode({ version: 3 }, [{ descriptor: describe([2], 'float64'), data: new Float64Array([10, 20]) }]),
    encode({ version: 3 }, [{ descriptor: describe([1], 'int32'), data: new Int32Array([42]) }]),
  ];
  const body = concatBytes(...messages);
  console.log(`server body: ${body.byteLength} bytes across ${messages.length} messages`);

  const { fetch: fakeFetch, requestCount } = makeRangeServer(body);

  // Open the file — a HEAD probe + Range reads per message preamble
  // during the lazy scan, no payload downloads yet.
  const file = await TensogramFile.fromUrl('https://example.invalid/demo.tgm', {
    fetch: fakeFetch,
  });
  try {
    console.log(`after open: ${requestCount.value} requests (1 HEAD + ${messages.length} preamble reads)`);
    console.log(`messageCount=${file.messageCount}  byteLength=${file.byteLength}`);

    // Fetch just the middle message — one more Range request.
    const before = requestCount.value;
    const m1 = await file.message(1);
    try {
      console.log(
        `fetched message 1 in ${requestCount.value - before} more request(s); ` +
          `values=${Array.from(m1.objects[0].data() as Float64Array)}`,
      );
    } finally {
      m1.close();
    }

    // Fetching the same message again hits the LRU cache — zero new
    // network calls.
    const cachedBefore = requestCount.value;
    const m1Cached = await file.message(1);
    m1Cached.close();
    console.log(`cached message 1 in ${requestCount.value - cachedBefore} more request(s)`);
  } finally {
    file.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

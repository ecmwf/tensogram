// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 16 — Remote batch + prefetch (TypeScript)
 *
 * Shows the bounded-concurrency parallel-fetch APIs:
 *
 *   - prefetchLayouts(msgIndices)        — pre-warm the layout cache
 *   - messageObjectBatch(idx, j)         — fan-out object decode
 *
 * The mock server records peak in-flight requests so we can verify
 * the per-host concurrency cap is respected.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 16_remote_batch.ts
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  encode,
  init,
  TensogramFile,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

function describe(shape: number[]): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype: 'float32',
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

function concatBytes(...parts: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

interface ServerHandle {
  url: string;
  close: () => Promise<void>;
  peakInFlight: () => number;
}

/**
 * Range-capable HTTP server that records the peak number of
 * concurrent in-flight requests, so the example can demonstrate
 * the bounded-concurrency limiter actually capping fan-out.
 */
function serveWithMetrics(body: Uint8Array): Promise<ServerHandle> {
  return new Promise((resolve) => {
    let inFlight = 0;
    let peak = 0;

    const handler = async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      // Tiny artificial delay so concurrent requests can overlap on the wire.
      await new Promise((r) => setTimeout(r, 5));

      try {
        if (req.method === 'HEAD') {
          res.writeHead(200, {
            'Content-Length': body.byteLength,
            'Accept-Ranges': 'bytes',
          });
          res.end();
          return;
        }
        const range = req.headers['range'];
        if (typeof range === 'string' && range.startsWith('bytes=')) {
          const m = /^bytes=(\d+)-(\d+)?$/.exec(range);
          if (!m) {
            res.writeHead(416);
            res.end();
            return;
          }
          const start = parseInt(m[1], 10);
          const end = m[2] !== undefined ? parseInt(m[2], 10) : body.byteLength - 1;
          const slice = Buffer.from(body.buffer, body.byteOffset + start, end - start + 1);
          res.writeHead(206, {
            'Content-Range': `bytes ${start}-${end}/${body.byteLength}`,
            'Content-Length': slice.byteLength,
          });
          res.end(slice);
          return;
        }
        res.writeHead(200, { 'Content-Length': body.byteLength });
        res.end(Buffer.from(body));
      } finally {
        inFlight--;
      }
    };

    const server = createServer((req, res) => {
      void handler(req, res);
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      if (!addr || typeof addr === 'string') {
        throw new Error('server.address() returned an unexpected shape');
      }
      const url = `http://127.0.0.1:${addr.port}/data.tgm`;
      resolve({
        url,
        close: () => new Promise((res) => server.close(() => res())),
        peakInFlight: () => peak,
      });
    });
  });
}

async function main(): Promise<void> {
  await init();

  // Build 8 messages, each with one tensor; concatenate.
  const N = 8;
  const messages = Array.from({ length: N }, (_, i) =>
    encode({ version: 3 }, [
      {
        descriptor: describe([10]),
        data: new Float32Array(10).map((_, k) => i * 100 + k),
      },
    ]),
  );
  const body = concatBytes(...messages);
  console.log(`Built ${N}-message file: ${body.byteLength} bytes`);

  const server = await serveWithMetrics(body);
  try {
    console.log(`Serving at ${server.url}`);

    // Open with concurrency = 3 — the prefetch + batch fan-out must
    // never exceed 3 concurrent server requests.
    const file = await TensogramFile.fromUrl(server.url, { concurrency: 3 });
    try {
      console.log(`messageCount = ${file.messageCount}`);

      console.log('\nprefetching all layouts...');
      await file.prefetchLayouts(Array.from({ length: N }, (_, i) => i), {
        concurrency: 3,
      });
      console.log(`peak in-flight after prefetch: ${server.peakInFlight()}`);

      console.log('\nbatch decode object 0 across all messages with concurrency 3...');
      const results = await file.messageObjectBatch(
        Array.from({ length: N }, (_, i) => i),
        0,
        { concurrency: 3 },
      );
      try {
        results.forEach((r, i) => {
          const arr = r.objects[0].data() as Float32Array;
          console.log(`  message ${i}: head=[${arr[0]}, ${arr[1]}, ${arr[2]}]`);
        });
      } finally {
        results.forEach((r) => r.close());
      }
      console.log(`\npeak in-flight overall: ${server.peakInFlight()} (cap: 3)`);
      if (server.peakInFlight() > 3) {
        throw new Error(
          `Peak in-flight (${server.peakInFlight()}) exceeded the configured concurrency cap of 3`,
        );
      }
    } finally {
      file.close();
    }
  } finally {
    await server.close();
  }
  console.log('\nRemote batch + prefetch example complete.');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 15 — Remote access (TypeScript)
 *
 * Mirrors `examples/python/14_remote_access.py` and
 * `examples/rust/src/bin/14_remote_access.rs`: stand up a tiny
 * Range-capable HTTP server in-process, then open the file with
 * `TensogramFile.fromUrl` and demonstrate the layout-aware accessors:
 *
 *   - messageMetadata(i)      — fetches header chunk only
 *   - messageDescriptors(i)   — descriptors via index frame
 *   - messageObject(i, j)     — exactly one Range GET for one frame
 *   - messageObjectRange(...) — partial sub-tensor decode
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 15_remote_access.ts
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

/**
 * Serve `body` over HTTP with full Range support.  Mirrors the
 * Python example's _make_range_handler exactly, in Node.
 */
function serveRangeBody(body: Uint8Array): { url: string; close: () => Promise<void> } {
  const handler = (req: IncomingMessage, res: ServerResponse): void => {
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
  };

  const server = createServer(handler);
  return new Promise<{ url: string; close: () => Promise<void> }>((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      if (!addr || typeof addr === 'string') {
        throw new Error('server.address() returned an unexpected shape');
      }
      const url = `http://127.0.0.1:${addr.port}/forecast.tgm`;
      resolve({
        url,
        close: () =>
          new Promise<void>((res) =>
            server.close(() => res()),
          ),
      });
    });
  }) as unknown as { url: string; close: () => Promise<void> };
}

async function main(): Promise<void> {
  await init();

  // ── Encode a self-contained .tgm with two objects ─────────────────────
  const meta = {
    version: 3,
    base: [
      { mars: { param: '2t', step: 0, date: '20260401' } },
      { mars: { param: 'msl', step: 0, date: '20260401' } },
    ],
  };
  const temperature = new Float32Array(72 * 144);
  const pressure = new Float32Array(72 * 144);
  for (let i = 0; i < temperature.length; i++) {
    temperature[i] = 273.15 + (i % 100) * 0.1;
    pressure[i] = 101325 + (i % 50) * 50;
  }
  const tgm = encode(meta, [
    { descriptor: describe([72, 144]), data: temperature },
    { descriptor: describe([72, 144]), data: pressure },
  ]);
  console.log(`Encoded ${tgm.byteLength} bytes with 2 objects`);

  // ── Stand up the local Range-capable server ───────────────────────────
  const server = await Promise.resolve(serveRangeBody(tgm));
  try {
    console.log(`Serving at ${server.url}`);

    // ── Open via lazy HTTP backend ───────────────────────────────────────
    const file = await TensogramFile.fromUrl(server.url);
    try {
      console.log(`\nOpened remote: source=${file.source}`);
      console.log(`  messageCount = ${file.messageCount}`);
      console.log(`  byteLength   = ${file.byteLength}`);

      // ── Metadata only (header chunk) ──────────────────────────────────
      const m = await file.messageMetadata(0);
      console.log(`\nMetadata: version=${m.version}`);
      console.log(`  base[0] = ${JSON.stringify(m.base?.[0])}`);

      // ── Descriptors only (CBOR per frame) ─────────────────────────────
      const { descriptors } = await file.messageDescriptors(0);
      console.log(`\nDescriptors: ${descriptors.length} objects`);
      descriptors.forEach((d, i) => {
        console.log(`  [${i}] shape=${JSON.stringify(d.shape)}  dtype=${d.dtype}`);
      });

      // ── Per-object decode (one Range per object) ──────────────────────
      const t = await file.messageObject(0, 0);
      try {
        const arr = t.objects[0].data() as Float32Array;
        console.log(`\nObject 0: shape=${JSON.stringify(t.objects[0].descriptor.shape)} bytes=${arr.byteLength}`);
      } finally {
        t.close();
      }
      const p = await file.messageObject(0, 1);
      try {
        const arr = p.objects[0].data() as Float32Array;
        console.log(`Object 1: shape=${JSON.stringify(p.objects[0].descriptor.shape)} bytes=${arr.byteLength}`);
      } finally {
        p.close();
      }

      // ── Range decode (a few elements from one frame) ──────────────────
      const r = await file.messageObjectRange(0, 0, [
        [0, 4],
        [10000, 3],
      ]);
      console.log(`\nRange decode (object 0): parts=${r.parts.length}`);
      r.parts.forEach((part, i) => {
        const head = Array.from(part as Float32Array).slice(0, 4);
        console.log(`  parts[${i}] length=${part.length} head=${JSON.stringify(head)}`);
      });
    } finally {
      file.close();
    }
  } finally {
    await server.close();
  }
  console.log('\nRemote access example complete.');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

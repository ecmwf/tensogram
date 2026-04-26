// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 18 — Remote-scan tracing (TypeScript)
 *
 * Mirrors `examples/rust/src/bin/18_remote_scan_trace.rs`: stand up a
 * tiny Range-capable Node HTTP server, encode a multi-message `.tgm`,
 * then open it with `{ bidirectional: true, debug: true }` and watch
 * the dispatcher's `tensogram:scan:*` events.
 *
 * The dispatcher emits these events via `console.debug`.  The example
 * intercepts that channel so each event prints with a prefix; in real
 * code you'd forward them to whatever logging library you use.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript && npm install
 *   npx tsx 18_remote_scan_trace.ts
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { encode, init, TensogramFile, type DataObjectDescriptor } from '@ecmwf.int/tensogram';

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

interface ServerHandle {
  url: string;
  close: () => Promise<void>;
}

function serveRangeBody(body: Uint8Array): Promise<ServerHandle> {
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
      const m = /^bytes=(\d+)?-(\d+)?$/.exec(range);
      if (!m) {
        res.writeHead(416);
        res.end();
        return;
      }
      const total = body.byteLength;
      let start: number;
      let end: number;
      if (m[1] === undefined && m[2] !== undefined) {
        const suffix = parseInt(m[2], 10);
        start = Math.max(0, total - suffix);
        end = total - 1;
      } else {
        start = parseInt(m[1] ?? '0', 10);
        end = m[2] !== undefined ? parseInt(m[2], 10) : total - 1;
      }
      const slice = Buffer.from(body.buffer, body.byteOffset + start, end - start + 1);
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${total}`,
        'Content-Length': slice.byteLength,
      });
      res.end(slice);
      return;
    }
    res.writeHead(200, { 'Content-Length': body.byteLength });
    res.end(Buffer.from(body));
  };
  return new Promise<ServerHandle>((resolve) => {
    const server = createServer(handler);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      if (!addr || typeof addr === 'string') {
        throw new Error('server.address() returned an unexpected shape');
      }
      const url = `http://127.0.0.1:${addr.port}/forecast.tgm`;
      resolve({
        url,
        close: () => new Promise<void>((res) => server.close(() => res())),
      });
    });
  });
}

async function main(): Promise<void> {
  await init();

  const N = 5;
  const messages: Uint8Array[] = [];
  for (let i = 0; i < N; i++) {
    const data = new Float32Array(16).map((_, j) => i * 100 + j);
    messages.push(
      encode(
        { version: 3, base: [{ msg_index: i }] },
        [{ descriptor: describe([16]), data }],
      ),
    );
  }
  const total = messages.reduce((acc, m) => acc + m.byteLength, 0);
  const file = new Uint8Array(total);
  let offset = 0;
  for (const m of messages) {
    file.set(m, offset);
    offset += m.byteLength;
  }
  console.log(`Encoded ${N} messages, ${file.byteLength} bytes total`);

  const server = await serveRangeBody(file);
  const originalDebug = console.debug.bind(console);
  console.debug = (label: unknown, payload?: unknown) => {
    if (typeof label === 'string' && label.startsWith('tensogram:')) {
      originalDebug(`  [trace] ${label}`, payload ?? '');
    } else {
      originalDebug(label as string, payload as object);
    }
  };

  try {
    console.log(`\nServing at ${server.url}`);
    console.log('\n── Forward-only walker (default) ──');
    const fwd = await TensogramFile.fromUrl(server.url, { debug: true });
    try {
      console.log(`Opened: ${fwd.messageCount} messages`);
    } finally {
      fwd.close();
    }

    console.log('\n── Bidirectional walker (opt-in) ──');
    const bid = await TensogramFile.fromUrl(server.url, {
      bidirectional: true,
      debug: true,
    });
    try {
      console.log(`Opened: ${bid.messageCount} messages`);
    } finally {
      bid.close();
    }

    console.log('\nDispatcher events streamed via console.debug above.');
    console.log('Replace the interceptor with your own logger to capture them.');
  } finally {
    console.debug = originalDebug;
    await server.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

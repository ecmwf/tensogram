// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 17 — S3 / S3-compatible read via createAwsSigV4Fetch (TypeScript)
 *
 * Stand up a mock S3-style HTTP server that rejects requests missing
 * a SigV4 `Authorization` header, then open the file via
 * `TensogramFile.fromUrl({ fetch: signedFetch })`.
 *
 * Real-world usage substitutes this mock for `globalThis.fetch`
 * against the actual S3 endpoint:
 *
 *     const signedFetch = createAwsSigV4Fetch({
 *       accessKeyId:     process.env.AWS_ACCESS_KEY_ID!,
 *       secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
 *       region:          "eu-west-1",
 *     });
 *     const file = await TensogramFile.fromUrl(
 *       "https://my-bucket.s3.eu-west-1.amazonaws.com/data.tgm",
 *       { fetch: signedFetch },
 *     );
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 17_remote_s3_signed_fetch.ts
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  createAwsSigV4Fetch,
  encode,
  init,
  TensogramFile,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

const CREDS = {
  accessKeyId: 'AKIDEXAMPLE',
  secretAccessKey: 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY',
  region: 'us-east-1',
  service: 's3',
};

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

interface MockS3 {
  url: string;
  close: () => Promise<void>;
  unauthedRequests: () => number;
}

/**
 * S3-style mock: returns 401 unless the incoming request carries
 * `Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/...`.
 * Does NOT verify the signature itself — the byte-for-byte signer
 * is covered by tests/signAwsV4.test.ts.
 */
function mockS3(body: Uint8Array): Promise<MockS3> {
  let unauthed = 0;
  const handler = (req: IncomingMessage, res: ServerResponse): void => {
    const auth = req.headers['authorization'];
    if (
      typeof auth !== 'string' ||
      !auth.startsWith('AWS4-HMAC-SHA256 ') ||
      !auth.includes(`Credential=${CREDS.accessKeyId}/`)
    ) {
      unauthed++;
      res.writeHead(401);
      res.end('Unauthorized');
      return;
    }
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

  return new Promise((resolve) => {
    const server = createServer(handler);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      if (!addr || typeof addr === 'string') {
        throw new Error('server.address() returned an unexpected shape');
      }
      const url = `http://127.0.0.1:${addr.port}/bucket/data.tgm`;
      resolve({
        url,
        close: () => new Promise((res) => server.close(() => res())),
        unauthedRequests: () => unauthed,
      });
    });
  });
}

async function main(): Promise<void> {
  await init();

  const data = new Float32Array([3.14, 2.72, 1.61, 1.41]);
  const tgm = encode({ version: 3 }, [
    { descriptor: describe([4]), data },
  ]);
  const server = await mockS3(tgm);
  try {
    console.log(`Serving mock S3 at ${server.url}`);

    // ── Without signing: rejected ─────────────────────────────────────
    try {
      await TensogramFile.fromUrl(server.url);
      throw new Error('expected unsigned request to fail');
    } catch (err) {
      console.log(`Unsigned request rejected as expected: ${(err as Error).message}`);
    }
    console.log(`unauthed requests so far: ${server.unauthedRequests()}`);

    // ── With SigV4 signing: accepted ──────────────────────────────────
    const signedFetch = createAwsSigV4Fetch(CREDS);
    const file = await TensogramFile.fromUrl(server.url, { fetch: signedFetch });
    try {
      console.log(`\nSigned request accepted; messageCount = ${file.messageCount}`);
      const decoded = await file.message(0);
      try {
        const arr = decoded.objects[0].data() as Float32Array;
        console.log(`Decoded: [${Array.from(arr).join(', ')}]`);
      } finally {
        decoded.close();
      }
    } finally {
      file.close();
    }
  } finally {
    await server.close();
  }
  console.log('\nS3 SigV4 example complete.');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});

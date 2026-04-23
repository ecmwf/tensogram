// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Integration tests for `createAwsSigV4Fetch`.  We mock S3 with an
 * in-process fetch that rejects requests missing a SigV4
 * `Authorization` header, and verify TensogramFile.fromUrl works
 * end-to-end through the signed wrapper.
 */

import { describe, expect, it } from 'vitest';
import { createAwsSigV4Fetch } from '../src/auth/awsSigV4Fetch.js';
import { encode, TensogramFile } from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

const CREDS = {
  accessKeyId: 'AKIDEXAMPLE',
  secretAccessKey: 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY',
  region: 'us-east-1',
  service: 's3',
};

/**
 * Build a mock S3 fetch that returns 401 unless the request carries
 * a SigV4 Authorization header for the configured access key.  Does
 * NOT verify the signature itself — the byte-for-byte signer test
 * lives in signAwsV4.test.ts.
 */
function makeStrictAuthServer(body: Uint8Array): {
  fetch: typeof globalThis.fetch;
  authedRequests: number;
  unauthedRequests: number;
  reset: () => void;
} {
  const state = { authedRequests: 0, unauthedRequests: 0 };
  const fetchFn: typeof globalThis.fetch = async (
    _input: string | URL | Request,
    init?: RequestInit,
  ) => {
    const auth =
      init?.headers instanceof Headers
        ? init.headers.get('authorization')
        : (init?.headers as Record<string, string> | undefined)?.['Authorization'];

    if (!auth || !auth.startsWith('AWS4-HMAC-SHA256 ')) {
      state.unauthedRequests++;
      return new Response('Unauthorized', { status: 401 });
    }
    if (!auth.includes(`Credential=${CREDS.accessKeyId}/`)) {
      state.unauthedRequests++;
      return new Response('Forbidden', { status: 403 });
    }
    state.authedRequests++;
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
      init?.headers instanceof Headers ? init.headers.get('range') : undefined;
    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      if (!m) return new Response('bad range', { status: 416 });
      const start = parseInt(m[1], 10);
      const end = parseInt(m[2], 10);
      const slice = body.subarray(start, end + 1);
      const buf = new ArrayBuffer(slice.byteLength);
      new Uint8Array(buf).set(slice);
      return new Response(buf, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${body.byteLength}`,
          'content-length': String(slice.byteLength),
        },
      });
    }
    const buf = new ArrayBuffer(body.byteLength);
    new Uint8Array(buf).set(body);
    return new Response(buf, { status: 200 });
  };
  return {
    fetch: fetchFn,
    get authedRequests() {
      return state.authedRequests;
    },
    get unauthedRequests() {
      return state.unauthedRequests;
    },
    reset(): void {
      state.authedRequests = 0;
      state.unauthedRequests = 0;
    },
  };
}

describe('createAwsSigV4Fetch — integration', () => {
  initOnce();

  it('signs requests so the strict-auth mock S3 accepts them', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([4], 'float32'),
        data: new Float32Array([1, 2, 3, 4]),
      },
    ]);
    const server = makeStrictAuthServer(msg);
    const signedFetch = createAwsSigV4Fetch(CREDS, { fetchImpl: server.fetch });

    const file = await TensogramFile.fromUrl(
      'https://my-bucket.s3.us-east-1.amazonaws.com/data.tgm',
      { fetch: signedFetch },
    );
    try {
      expect(file.messageCount).toBe(1);
      const decoded = await file.message(0);
      expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
      decoded.close();
      expect(server.authedRequests).toBeGreaterThan(0);
      expect(server.unauthedRequests).toBe(0);
    } finally {
      file.close();
    }
  });

  it('a plain fetch (no signing) is rejected by the strict mock', async () => {
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([2], 'float32'),
        data: new Float32Array([1, 2]),
      },
    ]);
    const server = makeStrictAuthServer(msg);
    await expect(
      TensogramFile.fromUrl('https://my-bucket.s3.us-east-1.amazonaws.com/data.tgm', {
        fetch: server.fetch,
      }),
    ).rejects.toThrow();
    expect(server.unauthedRequests).toBeGreaterThan(0);
  });

  it('rejects construction when no fetch implementation is available', () => {
    expect(() =>
      createAwsSigV4Fetch(CREDS, {
        // @ts-expect-error intentional: forcing the no-fetch branch
        fetchImpl: 'not a function',
      }),
    ).toThrow();
  });
});

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `fetch`-compatible wrapper that automatically applies AWS Signature
 * Version 4 to outgoing read-only requests, suitable as the
 * `FromUrlOptions.fetch` hook on `TensogramFile.fromUrl` against S3
 * or any S3-compatible HTTPS endpoint.
 *
 * Body signing always uses the SHA-256 of the empty string (or the
 * caller-supplied body's hash) — write paths are out of scope; for
 * uploads use a presigned URL.
 */

import { signAwsV4Request, type SigV4Credentials } from './signAwsV4.js';

/** Options accepted by [`createAwsSigV4Fetch`]. */
export interface AwsSigV4FetchOptions {
  /** Underlying fetch to invoke after signing.  Defaults to `globalThis.fetch`. */
  fetchImpl?: typeof globalThis.fetch;
}

/**
 * Build a `fetch`-compatible function that signs every outgoing
 * request with AWS SigV4 before delegating to the underlying fetch.
 *
 * Usage:
 *
 *     const signedFetch = createAwsSigV4Fetch({
 *       accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
 *       secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
 *       region: 'eu-west-1',
 *     });
 *     const file = await TensogramFile.fromUrl(
 *       'https://my-bucket.s3.eu-west-1.amazonaws.com/data.tgm',
 *       { fetch: signedFetch },
 *     );
 */
export function createAwsSigV4Fetch(
  creds: SigV4Credentials,
  options: AwsSigV4FetchOptions = {},
): typeof globalThis.fetch {
  const baseFetch = options.fetchImpl ?? globalThis.fetch;
  if (typeof baseFetch !== 'function') {
    throw new Error(
      'createAwsSigV4Fetch: no fetch implementation available; ' +
        'pass options.fetchImpl explicitly',
    );
  }

  return async function signedFetch(
    input: string | URL | Request,
    init: RequestInit = {},
  ): Promise<Response> {
    const request = await buildSignedRequest(input, init, creds);
    return baseFetch(request.url, request.init);
  };
}

/**
 * Sign one outgoing request and return the URL + RequestInit ready
 * to pass to the underlying fetch.  Uses
 * `'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'`
 * (SHA-256 of empty string) as the payload hash for read-only
 * requests; the caller is expected to use presigned URLs for write
 * paths.
 */
async function buildSignedRequest(
  input: string | URL | Request,
  init: RequestInit,
  creds: SigV4Credentials,
): Promise<{ url: string | URL; init: RequestInit }> {
  let url: string | URL;
  let method: string;
  let headers: Headers;
  if (input instanceof Request) {
    url = input.url;
    method = init.method ?? input.method;
    headers = new Headers(input.headers);
  } else {
    url = input;
    method = init.method ?? 'GET';
    headers = new Headers(init.headers);
  }

  const parsedUrl = url instanceof URL ? url : new URL(String(url));
  const result = await signAwsV4Request(
    {
      method,
      url: parsedUrl,
      headers,
      payloadHash: EMPTY_BODY_SHA256,
      timestamp: new Date(),
    },
    creds,
  );

  // Merge signed headers into the outgoing init.  `result.headers`
  // already carries host / x-amz-date / x-amz-content-sha256 / any
  // session token plus any caller-supplied headers (e.g. Range).
  result.headers.set('Authorization', result.authorization);

  const newInit: RequestInit = { ...init, method, headers: result.headers };
  return { url, init: newInit };
}

const EMPTY_BODY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';

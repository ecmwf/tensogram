// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Pure AWS Signature Version 4 signer.
 *
 * Implements the subset of AWS SigV4 needed to sign read-only HTTPS
 * requests against S3-compatible endpoints (GET / HEAD with optional
 * `Range` header).  Operates against the standard
 * [AWS sig-v4-test-suite](https://docs.aws.amazon.com/general/latest/gr/signature-v4-test-suite.html)
 * known-answer vectors — see `tests/signAwsV4.test.ts`.
 *
 * **Security note**: this signer is for client-side use against
 * read-only object stores.  Body signing always uses
 * `UNSIGNED-PAYLOAD`, so it is unsuitable for write paths where the
 * caller wants the server to verify the request body.  For write
 * paths use a presigned URL instead.
 *
 * Uses the Web Crypto API (`globalThis.crypto.subtle`), which is
 * available in browsers and in Node.js 18+.  No transitive
 * dependencies.
 */

/** AWS credentials carried with each signature. */
export interface SigV4Credentials {
  accessKeyId: string;
  secretAccessKey: string;
  /** Optional STS session token; emitted as `x-amz-security-token`. */
  sessionToken?: string;
  /** AWS region, e.g. `eu-west-1`. */
  region: string;
  /** AWS service identifier, defaults to `s3`. */
  service?: string;
}

/** Inputs to a single signing operation. */
export interface SigV4Input {
  /** HTTP method, e.g. `GET` / `HEAD`. */
  method: string;
  /** Full request URL (host + path + query). */
  url: URL;
  /** Request headers; the signer adds `host`, `x-amz-date`, and (optionally) `x-amz-content-sha256`. */
  headers: Headers;
  /**
   * Hex SHA-256 of the request body.  Use `'UNSIGNED-PAYLOAD'` for
   * GET / HEAD with a Range header (S3 accepts this when the body is
   * empty), or `'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'`
   * (the SHA-256 of an empty string) when the server requires a real
   * payload hash even on empty bodies.
   */
  payloadHash: string;
  /** Reference timestamp; defaults to "now" in the wrapper. */
  timestamp: Date;
  /**
   * When `true` (S3 default), include `x-amz-content-sha256` among
   * the signed headers — required by S3 even for GET / HEAD.  When
   * `false`, omit the header entirely; matches the generic-service
   * AWS sig-v4-test-suite vectors.  Defaults to `true`.
   */
  includeContentSha256?: boolean;
}

/** Result of signing — caller adds `Authorization` to the outgoing request. */
export interface SigV4Result {
  /** Value to set as the `Authorization` header. */
  authorization: string;
  /** Headers to include with the request (includes signed headers + payload-hash + date). */
  headers: Headers;
}

/**
 * Sign one HTTPS request with AWS SigV4.
 *
 * Returns the `Authorization` header value plus the full set of
 * headers to send (including `host`, `x-amz-date`,
 * `x-amz-content-sha256`, and any session token).  The caller is
 * responsible for issuing the actual HTTP request.
 */
export async function signAwsV4Request(
  input: SigV4Input,
  creds: SigV4Credentials,
): Promise<SigV4Result> {
  const service = creds.service ?? 's3';
  const region = creds.region;
  const amzDate = formatAmzDate(input.timestamp);
  const dateStamp = amzDate.slice(0, 8);

  const signedHeaders = buildSignedHeaders(input, amzDate, creds);

  const canonicalRequest = buildCanonicalRequest(
    input.method,
    input.url,
    signedHeaders,
    input.payloadHash,
  );

  const credentialScope = `${dateStamp}/${region}/${service}/aws4_request`;
  const stringToSign = [
    'AWS4-HMAC-SHA256',
    amzDate,
    credentialScope,
    await sha256Hex(canonicalRequest),
  ].join('\n');

  const signingKey = await deriveSigningKey(
    creds.secretAccessKey,
    dateStamp,
    region,
    service,
  );
  const signature = bytesToHex(await hmacSha256(signingKey, stringToSign));

  const signedHeaderNames = Object.keys(signedHeaders).sort().join(';');
  const authorization =
    `AWS4-HMAC-SHA256 Credential=${creds.accessKeyId}/${credentialScope},` +
    ` SignedHeaders=${signedHeaderNames},` +
    ` Signature=${signature}`;

  return { authorization, headers: toHeadersObject(signedHeaders) };
}

// ── Canonical request construction ─────────────────────────────────────────

/**
 * Build the lowercase {name: value} map of headers to sign.  Always
 * includes `host`, `x-amz-date`, and `x-amz-content-sha256`; adds
 * `x-amz-security-token` when an STS session token is supplied; and
 * preserves any caller-supplied headers (notably `range`).
 *
 * Header *values* are trimmed and inner-whitespace-collapsed per the
 * SigV4 spec (`get-header-value-trim` test vector).
 */
function buildSignedHeaders(
  input: SigV4Input,
  amzDate: string,
  creds: SigV4Credentials,
): Record<string, string> {
  const out: Record<string, string> = {};

  // Caller-supplied headers come first; we may override their values.
  input.headers.forEach((value, name) => {
    out[name.toLowerCase()] = canonicaliseHeaderValue(value);
  });

  out.host = input.url.host;
  out['x-amz-date'] = amzDate;
  if (input.includeContentSha256 ?? true) {
    out['x-amz-content-sha256'] = input.payloadHash;
  }
  if (creds.sessionToken) {
    out['x-amz-security-token'] = creds.sessionToken;
  }

  return out;
}

/** Trim and collapse runs of inner whitespace in unquoted header values. */
function canonicaliseHeaderValue(value: string): string {
  return value.trim().replace(/\s+/g, ' ');
}

/** Convert an internal header map to a `Headers` object for outgoing requests. */
function toHeadersObject(map: Record<string, string>): Headers {
  const out = new Headers();
  for (const [k, v] of Object.entries(map)) out.set(k, v);
  return out;
}

/**
 * Construct the canonical request string per SigV4 §3:
 *   METHOD\n CanonicalURI\n CanonicalQueryString\n
 *   CanonicalHeaders\n SignedHeaders\n PayloadHash
 */
function buildCanonicalRequest(
  method: string,
  url: URL,
  signedHeaders: Record<string, string>,
  payloadHash: string,
): string {
  const canonicalUri = canonicaliseUri(url.pathname);
  const canonicalQuery = canonicaliseQuery(url.searchParams);
  const headerNames = Object.keys(signedHeaders).sort();
  const canonicalHeaders = headerNames.map((n) => `${n}:${signedHeaders[n]}\n`).join('');
  const signedHeaderNames = headerNames.join(';');
  return [
    method.toUpperCase(),
    canonicalUri,
    canonicalQuery,
    canonicalHeaders,
    signedHeaderNames,
    payloadHash,
  ].join('\n');
}

/**
 * Canonicalise a URI path segment-by-segment — each segment is
 * URI-encoded except for unreserved characters and forward slashes.
 * Empty path becomes `/`.  S3 special-cases double-encoding for
 * non-`s3` services; we always single-encode (matches the
 * `get-vanilla` test vector).
 */
function canonicaliseUri(path: string): string {
  if (path === '') return '/';
  return path
    .split('/')
    .map((segment) => uriEncode(segment, /* encodeSlash */ false))
    .join('/');
}

/** Canonicalise the query string — sorted by name, with each name and value URI-encoded. */
function canonicaliseQuery(params: URLSearchParams): string {
  const pairs: Array<[string, string]> = [];
  params.forEach((value, name) => {
    pairs.push([name, value]);
  });
  pairs.sort(([a], [b]) => {
    if (a === b) return 0;
    return a < b ? -1 : 1;
  });
  return pairs
    .map(([k, v]) => `${uriEncode(k, true)}=${uriEncode(v, true)}`)
    .join('&');
}

/** AWS-style URI encoding: same as RFC 3986 but optionally encodes `/`. */
function uriEncode(s: string, encodeSlash: boolean): string {
  let out = '';
  for (const ch of s) {
    if (
      (ch >= 'A' && ch <= 'Z') ||
      (ch >= 'a' && ch <= 'z') ||
      (ch >= '0' && ch <= '9') ||
      ch === '-' ||
      ch === '_' ||
      ch === '.' ||
      ch === '~'
    ) {
      out += ch;
    } else if (ch === '/' && !encodeSlash) {
      out += ch;
    } else {
      out += encodeURIComponent(ch).replace(/!/g, '%21').replace(/'/g, '%27').replace(/\(/g, '%28').replace(/\)/g, '%29').replace(/\*/g, '%2A');
    }
  }
  return out;
}

// ── Crypto primitives over Web Crypto ───────────────────────────────────────

/** Format a Date as `YYYYMMDDTHHMMSSZ` per SigV4. */
function formatAmzDate(d: Date): string {
  const pad = (n: number, w = 2) => String(n).padStart(w, '0');
  return (
    `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}` +
    `T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}Z`
  );
}

/** Hex-encode bytes. */
function bytesToHex(bytes: ArrayBuffer | Uint8Array): string {
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out = '';
  for (const b of arr) out += b.toString(16).padStart(2, '0');
  return out;
}

/** Copy a Uint8Array's bytes into a fresh ArrayBuffer (BufferSource-compatible). */
function toArrayBuffer(view: Uint8Array): ArrayBuffer {
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(view);
  return out;
}

/** SHA-256 of a UTF-8 string, returned hex-encoded. */
async function sha256Hex(s: string): Promise<string> {
  const data = toArrayBuffer(new TextEncoder().encode(s));
  const hash = await getSubtle().digest('SHA-256', data);
  return bytesToHex(hash);
}

/** HMAC-SHA-256 with a raw-bytes key, returning the raw signature bytes. */
async function hmacSha256(
  key: Uint8Array,
  message: string | Uint8Array,
): Promise<Uint8Array> {
  const subtle = getSubtle();
  const cryptoKey = await subtle.importKey(
    'raw',
    toArrayBuffer(key),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const data = typeof message === 'string' ? new TextEncoder().encode(message) : message;
  const sig = await subtle.sign('HMAC', cryptoKey, toArrayBuffer(data));
  return new Uint8Array(sig);
}

/**
 * Derive the signing key per SigV4 §2:
 *   kDate   = HMAC("AWS4" + secretKey, dateStamp)
 *   kRegion = HMAC(kDate, region)
 *   kService = HMAC(kRegion, service)
 *   kSigning = HMAC(kService, "aws4_request")
 */
async function deriveSigningKey(
  secretAccessKey: string,
  dateStamp: string,
  region: string,
  service: string,
): Promise<Uint8Array> {
  const kSecret = new TextEncoder().encode(`AWS4${secretAccessKey}`);
  const kDate = await hmacSha256(kSecret, dateStamp);
  const kRegion = await hmacSha256(kDate, region);
  const kService = await hmacSha256(kRegion, service);
  return await hmacSha256(kService, 'aws4_request');
}

/** Look up Web Crypto's SubtleCrypto, with a clear error if missing. */
function getSubtle(): SubtleCrypto {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error(
      'AWS SigV4 signing requires Web Crypto API (globalThis.crypto.subtle); ' +
        'available in browsers and Node 18+',
    );
  }
  return subtle;
}

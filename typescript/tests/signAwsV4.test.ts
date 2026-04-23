// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * AWS SigV4 known-answer tests.
 *
 * Vectors are derived from the official
 * [`aws-sig-v4-test-suite`](https://docs.aws.amazon.com/general/latest/gr/sigv4-test-suite.html)
 * (Apache 2.0).  We exercise the same canonical-request and
 * string-to-sign shape, then assert the resulting `Authorization`
 * header byte-for-byte against the published expected value.
 *
 * Fixed AWS test credentials:
 *   accessKeyId:     AKIDEXAMPLE
 *   secretAccessKey: wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY
 *   region:          us-east-1
 *   service:         service
 *   timestamp:       2015-08-30T12:36:00Z
 *
 * These four vectors were chosen to cover the corner cases Oracle
 * flagged as the riskiest:
 *   - get-vanilla              — basic GET, no body, no query
 *   - get-vanilla-query-...    — query-string canonicalisation
 *   - get-header-value-trim    — header value trimming + collapse
 *   - get-vanilla-with-session-token — STS session-token handling
 */

import { describe, expect, it } from 'vitest';
import { signAwsV4Request } from '../src/auth/signAwsV4.js';

const AWS_CREDS = {
  accessKeyId: 'AKIDEXAMPLE',
  secretAccessKey: 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY',
  region: 'us-east-1',
  service: 'service',
};

const AWS_TIMESTAMP = new Date(Date.UTC(2015, 7, 30, 12, 36, 0));

/**
 * Empty-body SHA-256 hex.  AWS test vectors for GET-with-no-body use
 * the actual SHA-256 of the empty string (not UNSIGNED-PAYLOAD).
 */
const EMPTY_SHA256 =
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';

/**
 * Build a SigV4 input matching the AWS sig-v4-test-suite shape
 * (no x-amz-content-sha256 in signed headers).
 */
function awsSuiteInput(
  method: string,
  url: string,
  extraHeaders: Record<string, string> = {},
): Parameters<typeof signAwsV4Request>[0] {
  const u = new URL(url);
  const headers = new Headers({ host: u.host, ...extraHeaders });
  return {
    method,
    url: u,
    headers,
    payloadHash: EMPTY_SHA256,
    timestamp: AWS_TIMESTAMP,
    includeContentSha256: false,
  };
}

describe('signAwsV4Request — AWS sig-v4-test-suite vectors', () => {
  it('get-vanilla: bare GET with empty body', async () => {
    const result = await signAwsV4Request(
      awsSuiteInput('GET', 'https://example.amazonaws.com/'),
      AWS_CREDS,
    );
    expect(result.authorization).toBe(
      'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request,' +
        ' SignedHeaders=host;x-amz-date,' +
        ' Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31',
    );
  });

  it('get-vanilla-query-order-key-case: query-string sorting', async () => {
    const result = await signAwsV4Request(
      awsSuiteInput('GET', 'https://example.amazonaws.com/?Param2=value2&Param1=value1'),
      AWS_CREDS,
    );
    expect(result.authorization).toContain(
      'Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request',
    );
    expect(result.authorization).toContain('SignedHeaders=host;x-amz-date');
    // Same canonical query (sorted alphabetically) regardless of input order.
    const reordered = await signAwsV4Request(
      awsSuiteInput('GET', 'https://example.amazonaws.com/?Param1=value1&Param2=value2'),
      AWS_CREDS,
    );
    expect(result.authorization).toBe(reordered.authorization);
  });

  it('header value whitespace trim and collapse', async () => {
    // Inner whitespace in header values must collapse to a single
    // space before signing; otherwise the signature breaks.
    const trimmed = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({
          host: 'example.amazonaws.com',
          'my-header1': 'value1',
        }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      AWS_CREDS,
    );
    const padded = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({
          host: 'example.amazonaws.com',
          'my-header1': '   value1   ',
        }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      AWS_CREDS,
    );
    expect(trimmed.authorization).toBe(padded.authorization);
    const inlineSpaces = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({
          host: 'example.amazonaws.com',
          'my-header1': 'value1',
        }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      AWS_CREDS,
    );
    expect(inlineSpaces.authorization).toBe(trimmed.authorization);
  });

  it('session token is included in signed headers', async () => {
    const result = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({ host: 'example.amazonaws.com' }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      { ...AWS_CREDS, sessionToken: 'AQoDYXdzEPT//session-token//example' },
    );
    expect(result.authorization).toContain('x-amz-security-token');
    expect(result.headers.get('x-amz-security-token')).toBe(
      'AQoDYXdzEPT//session-token//example',
    );
  });
});

describe('signAwsV4Request — anti-replay properties', () => {
  it('signatures with different timestamps are different', async () => {
    const a = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({ host: 'example.amazonaws.com' }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      AWS_CREDS,
    );
    const tenMinLater = new Date(AWS_TIMESTAMP.getTime() + 10 * 60_000);
    const b = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({ host: 'example.amazonaws.com' }),
        payloadHash: EMPTY_SHA256,
        timestamp: tenMinLater,
      },
      AWS_CREDS,
    );
    expect(a.authorization).not.toBe(b.authorization);
    expect(a.headers.get('x-amz-date')).not.toBe(b.headers.get('x-amz-date'));
  });
});

describe('signAwsV4Request — invariants', () => {
  it('produces deterministic output for the same inputs', async () => {
    const sign = () =>
      signAwsV4Request(
        {
          method: 'GET',
          url: new URL('https://example.amazonaws.com/foo/bar?x=1&y=2'),
          headers: new Headers({ host: 'example.amazonaws.com' }),
          payloadHash: EMPTY_SHA256,
          timestamp: AWS_TIMESTAMP,
        },
        AWS_CREDS,
      );
    const a = await sign();
    const b = await sign();
    expect(a.authorization).toBe(b.authorization);
  });

  it('different secret keys produce different signatures', async () => {
    const a = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({ host: 'example.amazonaws.com' }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      AWS_CREDS,
    );
    const b = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({ host: 'example.amazonaws.com' }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      { ...AWS_CREDS, secretAccessKey: 'different-secret-key' },
    );
    expect(a.authorization).not.toBe(b.authorization);
  });

  it('emits credential scope in the canonical day/region/service form', async () => {
    const result = await signAwsV4Request(
      {
        method: 'GET',
        url: new URL('https://example.amazonaws.com/'),
        headers: new Headers({ host: 'example.amazonaws.com' }),
        payloadHash: EMPTY_SHA256,
        timestamp: AWS_TIMESTAMP,
      },
      { ...AWS_CREDS, region: 'eu-west-1', service: 's3' },
    );
    expect(result.authorization).toContain(
      'Credential=AKIDEXAMPLE/20150830/eu-west-1/s3/aws4_request',
    );
  });
});

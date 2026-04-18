// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `StreamingEncoder` tests.  Contract: frame-at-a-time writes produce a
 * message whose semantic content (decoded values + metadata) equals the
 * same message built via the one-shot `encode()` API.  Wire bytes
 * differ — streaming puts the index / hash frames in the footer — so
 * we compare decoded state, not byte-for-byte.
 */

import { describe, expect, it } from 'vitest';
import {
  decode,
  encode,
  InvalidArgumentError,
  StreamingEncoder,
} from '../src/index.js';
import { defaultMeta, initOnce, makeDescriptor } from './helpers.js';

describe('Scope C.1 — StreamingEncoder', () => {
  initOnce();

  it('single-object round-trip matches encode() semantics', () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const enc = new StreamingEncoder(defaultMeta());
    try {
      expect(enc.objectCount).toBe(0);
      enc.writeObject(makeDescriptor([4], 'float32'), values);
      expect(enc.objectCount).toBe(1);
      const bytes = enc.finish();

      const decoded = decode(bytes);
      try {
        expect(decoded.objects).toHaveLength(1);
        expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
      } finally {
        decoded.close();
      }
    } finally {
      enc.close();
    }
  });

  it('writes multiple objects in order', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      enc.writeObject(makeDescriptor([2], 'float32'), new Float32Array([1, 2]));
      enc.writeObject(makeDescriptor([3], 'float64'), new Float64Array([10, 20, 30]));
      enc.writeObject(makeDescriptor([1], 'int64'), new BigInt64Array([42n]));
      expect(enc.objectCount).toBe(3);
      expect(enc.bytesWritten).toBeGreaterThan(0);
      const bytes = enc.finish();

      const decoded = decode(bytes);
      try {
        expect(decoded.objects).toHaveLength(3);
        expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1, 2]);
        expect(Array.from(decoded.objects[1].data() as Float64Array)).toEqual([10, 20, 30]);
        expect(Array.from(decoded.objects[2].data() as BigInt64Array)).toEqual([42n]);
      } finally {
        decoded.close();
      }
    } finally {
      enc.close();
    }
  });

  it('writePreceder merges into base[i] on decode', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      enc.writePreceder({ units: 'K', mars: { param: '2t' } });
      enc.writeObject(makeDescriptor([2], 'float32'), new Float32Array([273.15, 274.0]));
      const bytes = enc.finish();
      const decoded = decode(bytes);
      try {
        const base0 = decoded.metadata.base?.[0];
        expect(base0?.['units']).toBe('K');
        const mars = base0?.['mars'] as Record<string, unknown> | undefined;
        expect(mars?.['param']).toBe('2t');
      } finally {
        decoded.close();
      }
    } finally {
      enc.close();
    }
  });

  it('rejects consecutive preceders without an intervening object', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      enc.writePreceder({ a: 1 });
      expect(() => enc.writePreceder({ b: 2 })).toThrow();
    } finally {
      enc.close();
    }
  });

  it('rejects a dangling preceder at finish', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      enc.writePreceder({ a: 1 });
      expect(() => enc.finish()).toThrow();
    } finally {
      enc.close();
    }
  });

  it('semantic equivalence with encode() for the same inputs', () => {
    const data = new Float32Array([1.5, 2.5, 3.5, 4.5]);
    const desc = makeDescriptor([4], 'float32');

    const buffered = encode(defaultMeta(), [{ descriptor: desc, data }]);
    const decBuf = decode(buffered);

    const enc = new StreamingEncoder(defaultMeta());
    enc.writeObject(desc, data);
    const streamed = enc.finish();
    enc.close();
    const decStr = decode(streamed);

    try {
      expect(decStr.objects).toHaveLength(decBuf.objects.length);
      expect(Array.from(decStr.objects[0].data() as Float32Array)).toEqual(
        Array.from(decBuf.objects[0].data() as Float32Array),
      );
      // Wire bytes intentionally differ (header-index vs footer-index).
      // Hash value must match — it's the xxh3 of the same encoded payload.
      expect(decStr.objects[0].descriptor.hash?.value).toBe(
        decBuf.objects[0].descriptor.hash?.value,
      );
    } finally {
      decBuf.close();
      decStr.close();
    }
  });

  it('finish() closes the encoder — subsequent calls throw', () => {
    const enc = new StreamingEncoder(defaultMeta());
    enc.writeObject(makeDescriptor([1], 'uint8'), new Uint8Array([7]));
    enc.finish();
    expect(() => enc.objectCount).toThrow(InvalidArgumentError);
    expect(() =>
      enc.writeObject(makeDescriptor([1], 'uint8'), new Uint8Array([8])),
    ).toThrow(InvalidArgumentError);
    expect(() => enc.finish()).toThrow(InvalidArgumentError);
  });

  it('close() is idempotent', () => {
    const enc = new StreamingEncoder(defaultMeta());
    enc.close();
    expect(() => enc.close()).not.toThrow();
  });

  it('writeObjectPreEncoded writes verbatim bytes', () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const enc = new StreamingEncoder(defaultMeta(), { hash: 'xxh3' });
    try {
      enc.writeObjectPreEncoded(makeDescriptor([4], 'float32'), bytes);
      const msg = enc.finish();
      const decoded = decode(msg);
      try {
        expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
      } finally {
        decoded.close();
      }
    } finally {
      enc.close();
    }
  });

  it('rejects invalid metadata up front', () => {
    // @ts-expect-error intentional: version is required
    expect(() => new StreamingEncoder({})).toThrow(InvalidArgumentError);
  });

  it('rejects a non-ArrayBufferView data argument', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      expect(() =>
        enc.writeObject(
          makeDescriptor([1], 'uint8'),
          // @ts-expect-error intentional bad input
          [0, 1, 2],
        ),
      ).toThrow(InvalidArgumentError);
    } finally {
      enc.close();
    }
  });

  it('rejects writePreceder on a non-plain-object entry', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      expect(() =>
        // @ts-expect-error intentional: arrays are rejected
        enc.writePreceder([1, 2, 3]),
      ).toThrow(InvalidArgumentError);
      expect(() =>
        // @ts-expect-error intentional: null is rejected
        enc.writePreceder(null),
      ).toThrow(InvalidArgumentError);
    } finally {
      enc.close();
    }
  });

  it('rejects writePreceder when entry contains _reserved_', () => {
    // `_reserved_` is a valid JS key, so TypeScript doesn't catch it
    // structurally — runtime validation carries the contract.
    const enc = new StreamingEncoder(defaultMeta());
    try {
      expect(() =>
        enc.writePreceder({ _reserved_: { foo: 'bar' } }),
      ).toThrow(InvalidArgumentError);
    } finally {
      enc.close();
    }
  });

  it('rejects writeObjectPreEncoded when data is not a Uint8Array', () => {
    const enc = new StreamingEncoder(defaultMeta());
    try {
      expect(() =>
        enc.writeObjectPreEncoded(
          makeDescriptor([1], 'float32'),
          // @ts-expect-error intentional: only Uint8Array is accepted for pre-encoded bytes
          new Float32Array([1]),
        ),
      ).toThrow(InvalidArgumentError);
    } finally {
      enc.close();
    }
  });
});

// ── Streaming-callback mode (Pass 6) ────────────────────────────────────────

/**
 * Concatenate every chunk delivered to a collector into a single
 * `Uint8Array`.  Used to rebuild the whole wire-format message from
 * a stream-mode encoding and verify it decodes identically to the
 * buffered path.
 */
function joinChunks(chunks: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const c of chunks) total += c.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}

describe('Scope C.1 — StreamingEncoder streaming mode', () => {
  initOnce();

  it('delivers preamble + header bytes during construction', () => {
    const chunks: Uint8Array[] = [];
    const enc = new StreamingEncoder(defaultMeta(), {
      onBytes: (c) => chunks.push(new Uint8Array(c)),
    });
    try {
      expect(chunks.length).toBeGreaterThan(0);
      // First 8 bytes must be the preamble magic.
      const first = chunks[0];
      const magic = new TextDecoder('ascii').decode(first.subarray(0, 8));
      expect(magic).toBe('TENSOGRM');
      expect(enc.streaming).toBe(true);
    } finally {
      enc.close();
    }
  });

  it('finish() returns an empty Uint8Array in streaming mode', () => {
    const chunks: Uint8Array[] = [];
    const enc = new StreamingEncoder(defaultMeta(), {
      onBytes: (c) => chunks.push(new Uint8Array(c)),
    });
    enc.writeObject(makeDescriptor([2], 'float32'), new Float32Array([1, 2]));
    const returned = enc.finish();
    expect(returned).toBeInstanceOf(Uint8Array);
    expect(returned.byteLength).toBe(0);
    // But every byte must have flowed through the callback.
    expect(chunks.length).toBeGreaterThan(0);
  });

  it('streamed bytes decode to the same message as buffered bytes', () => {
    const values = new Float32Array([1.5, 2.5, 3.5, 4.5, 5.5]);
    const desc = makeDescriptor([values.length], 'float32');

    // Buffered baseline.
    const buffered = new StreamingEncoder(defaultMeta());
    buffered.writeObject(desc, values);
    const bufferedBytes = buffered.finish();
    buffered.close();

    // Streaming variant — reassemble the bytes from chunks.
    const chunks: Uint8Array[] = [];
    const streamed = new StreamingEncoder(defaultMeta(), {
      onBytes: (c) => chunks.push(new Uint8Array(c)),
    });
    streamed.writeObject(desc, values);
    streamed.finish();
    streamed.close();
    const streamedBytes = joinChunks(chunks);

    // Decode both and compare semantic state — wire bytes differ
    // across calls because `_reserved_.uuid` is freshly generated.
    const bufMsg = decode(bufferedBytes);
    const strMsg = decode(streamedBytes);
    try {
      expect(strMsg.objects).toHaveLength(bufMsg.objects.length);
      expect(Array.from(strMsg.objects[0].data() as Float32Array)).toEqual(
        Array.from(bufMsg.objects[0].data() as Float32Array),
      );
    } finally {
      bufMsg.close();
      strMsg.close();
    }
  });

  it('bytesWritten tracks chunks delivered (not a zero buffer)', () => {
    const chunks: Uint8Array[] = [];
    const enc = new StreamingEncoder(defaultMeta(), {
      onBytes: (c) => chunks.push(new Uint8Array(c)),
    });
    try {
      const before = enc.bytesWritten;
      expect(before).toBeGreaterThan(0); // preamble + header already emitted
      enc.writeObject(makeDescriptor([4], 'float32'), new Float32Array([1, 2, 3, 4]));
      expect(enc.bytesWritten).toBeGreaterThan(before);
      // bytesWritten equals the running sum of chunk lengths.
      const callbackTotal = chunks.reduce((n, c) => n + c.byteLength, 0);
      expect(enc.bytesWritten).toBe(callbackTotal);
    } finally {
      enc.close();
    }
  });

  it('multi-object streaming delivers every frame through the callback', () => {
    const chunks: Uint8Array[] = [];
    const enc = new StreamingEncoder(defaultMeta(), {
      onBytes: (c) => chunks.push(new Uint8Array(c)),
    });
    enc.writeObject(makeDescriptor([2], 'float32'), new Float32Array([1, 2]));
    enc.writeObject(makeDescriptor([3], 'float64'), new Float64Array([10, 20, 30]));
    enc.writeObject(makeDescriptor([1], 'int64'), new BigInt64Array([42n]));
    enc.finish();
    enc.close();

    const bytes = joinChunks(chunks);
    const msg = decode(bytes);
    try {
      expect(msg.objects).toHaveLength(3);
      expect(Array.from(msg.objects[0].data() as Float32Array)).toEqual([1, 2]);
      expect(Array.from(msg.objects[1].data() as Float64Array)).toEqual([10, 20, 30]);
      expect(Array.from(msg.objects[2].data() as BigInt64Array)).toEqual([42n]);
    } finally {
      msg.close();
    }
  });

  it('propagates a callback exception as an error from the next write', () => {
    let throwOnNextCall = false;
    const enc = new StreamingEncoder(defaultMeta(), {
      onBytes: (_chunk) => {
        if (throwOnNextCall) {
          throw new Error('sink refused the chunk');
        }
      },
    });
    try {
      // Flip the flag so the next emission raises.
      throwOnNextCall = true;
      // writeObject flushes a frame → callback fires → exception →
      // Rust-side io::Error → TensogramError::Io → TS IoError.
      expect(() =>
        enc.writeObject(makeDescriptor([1], 'uint8'), new Uint8Array([7])),
      ).toThrow();
    } finally {
      enc.close();
    }
  });

  it('rejects a non-function onBytes with InvalidArgumentError', () => {
    expect(
      () =>
        new StreamingEncoder(defaultMeta(), {
          // @ts-expect-error intentional: onBytes must be callable
          onBytes: 'not a function',
        }),
    ).toThrow(InvalidArgumentError);
  });

  it('streaming: get streaming returns true, buffered: false', () => {
    const buffered = new StreamingEncoder(defaultMeta());
    expect(buffered.streaming).toBe(false);
    buffered.close();

    const streamed = new StreamingEncoder(defaultMeta(), { onBytes: () => {} });
    expect(streamed.streaming).toBe(true);
    streamed.close();
  });

  it('honours hash: false alongside streaming', () => {
    const chunks: Uint8Array[] = [];
    const enc = new StreamingEncoder(defaultMeta(), {
      hash: false,
      onBytes: (c) => chunks.push(new Uint8Array(c)),
    });
    enc.writeObject(makeDescriptor([2], 'float32'), new Float32Array([1, 2]));
    enc.finish();
    enc.close();

    const msg = decode(joinChunks(chunks));
    try {
      expect(msg.objects[0].descriptor.hash).toBeUndefined();
    } finally {
      msg.close();
    }
  });
});

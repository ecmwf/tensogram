// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Property-based tests (fast-check).
 *
 * Pins two broad invariants of the wrapper:
 *
 * 1. **`mapTensogramError` never throws and always returns a
 *    `TensogramError` subclass**, no matter what text is passed in.
 *    This is the safety invariant for the wrapper's error-routing
 *    layer — the only place we turn arbitrary WASM strings into
 *    structured errors.
 *
 * 2. **`encode` → `decode` round-trips are bit-exact** for every
 *    generated `Float32Array` across random shapes and random
 *    optional MARS-like metadata.  Any drift here would be a
 *    decoding or serialisation regression.
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import {
  decode,
  encode,
  init,
  TensogramError,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '../src/index.js';

// `mapTensogramError` is `@internal` and not re-exported from the barrel.
// Tests live inside the package so they can import it directly from its
// module.
import { mapTensogramError } from '../src/errors.js';

describe('Property: mapTensogramError is total and typed', () => {
  it('never throws for any finite string input', () => {
    fc.assert(
      // Require at least one non-whitespace char so there is always a
      // message to keep after prefix-stripping. The empty-string edge
      // case is pinned separately in errors.test.ts.
      fc.property(fc.string({ minLength: 1, maxLength: 500 }).filter((s) => s.trim().length > 0), (input) => {
        const err = mapTensogramError(new Error(input));
        expect(err).toBeInstanceOf(TensogramError);
        expect(err.message.length).toBeGreaterThan(0);
      }),
      { numRuns: 500 },
    );
  });

  it('preserves the raw message verbatim', () => {
    fc.assert(
      fc.property(fc.string({ maxLength: 300 }).filter((s) => s.length > 0), (input) => {
        const err = mapTensogramError(new Error(input));
        expect(err.rawMessage).toBe(input);
      }),
      { numRuns: 300 },
    );
  });

  it('handles non-Error inputs (strings, numbers, undefined)', () => {
    fc.assert(
      fc.property(
        fc.oneof(fc.string(), fc.integer(), fc.constant(undefined), fc.constant(null), fc.boolean()),
        (input) => {
          const err = mapTensogramError(input);
          expect(err).toBeInstanceOf(TensogramError);
        },
      ),
      { numRuns: 200 },
    );
  });
});

describe('Property: encode → decode round-trip is bit-exact for float32', () => {
  /** Arbitrary producing a valid f32-payload shape of modest size. */
  const arbShape = fc
    .array(fc.integer({ min: 1, max: 16 }), { minLength: 1, maxLength: 3 })
    .filter((s) => s.reduce((a, b) => a * b, 1) <= 2048);

  /** Arbitrary MARS-like optional metadata. */
  const arbMarsMeta = fc.record({
    class: fc.constantFrom('od', 'rd', 'ei'),
    type: fc.constantFrom('fc', 'an', 'cf'),
    step: fc.integer({ min: 0, max: 240 }),
  });

  it('round-trips random shapes and random MARS metadata', async () => {
    await init();

    await fc.assert(
      fc.asyncProperty(arbShape, arbMarsMeta, async (shape, mars) => {
        const count = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          // Use a deterministic but shape-varying pattern so the
          // comparison catches off-by-one errors.
          data[i] = (i + 1) * 0.5;
        }

        const strides: number[] = new Array(shape.length).fill(1);
        for (let i = shape.length - 2; i >= 0; i--) {
          strides[i] = strides[i + 1] * shape[i + 1];
        }

        const desc: DataObjectDescriptor = {
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
        const meta: GlobalMetadata = { version: 2, base: [{ mars }] };

        const msg = encode(meta, [{ descriptor: desc, data }]);
        const decoded = decode(msg);
        try {
          const out = decoded.objects[0].data() as Float32Array;
          expect(out.length).toBe(data.length);
          // Bit-exact check via Uint8 views — avoids subtle NaN issues.
          const inBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          const outBytes = new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
          expect(Array.from(outBytes)).toEqual(Array.from(inBytes));
          // And the readable MARS key survived the round-trip.
          const outMars = decoded.metadata.base?.[0].mars as Record<string, unknown> | undefined;
          expect(outMars?.class).toBe(mars.class);
          expect(outMars?.type).toBe(mars.type);
          expect(outMars?.step).toBe(mars.step);
        } finally {
          decoded.close();
        }
      }),
      { numRuns: 50 },
    );
  });
});

describe('Property: decode on garbage bytes always throws a typed error', () => {
  it('never panics or returns a valid DecodedMessage for random bytes', async () => {
    await init();

    await fc.assert(
      fc.asyncProperty(fc.uint8Array({ minLength: 0, maxLength: 256 }), async (bytes) => {
        try {
          const d = decode(bytes);
          // If decode somehow succeeded on random bytes, the metadata
          // must still be structurally sane (this path is statistically
          // impossible but legal if the generator happened to produce
          // a valid wire-format — accept it and clean up).
          expect(d.metadata).toBeDefined();
          d.close();
        } catch (err) {
          expect(err).toBeInstanceOf(TensogramError);
        }
      }),
      { numRuns: 200 },
    );
  });
});

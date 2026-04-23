// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Tests for `internal/pool.ts` — the bounded-concurrency limiter used
 * by every Range-heavy TS API (`prefetchLayouts`, `messageObjectBatch`,
 * descriptor-prefix fetches).
 */

import { describe, expect, it } from 'vitest';
import { createLimiter, DEFAULT_CONCURRENCY } from '../src/internal/pool.js';

/** Deferred helper: returns a promise plus its resolvers. */
function defer<T>(): { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void } {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('createLimiter', () => {
  it('returns results in submission order', async () => {
    const limit = createLimiter(3);
    const results = await Promise.all(
      [5, 3, 1, 4, 2].map((v, i) =>
        limit(async () => {
          // Reverse-order completion; limiter must still preserve caller order.
          await new Promise((r) => setTimeout(r, 20 - i * 4));
          return v;
        }),
      ),
    );
    expect(results).toEqual([5, 3, 1, 4, 2]);
  });

  it('enforces the concurrency cap exactly (in-flight never exceeds limit)', async () => {
    const cap = 3;
    const limit = createLimiter(cap);
    let inFlight = 0;
    let peak = 0;
    const tasks = Array.from({ length: 20 }, () =>
      limit(async () => {
        inFlight++;
        peak = Math.max(peak, inFlight);
        await new Promise((r) => setTimeout(r, 5));
        inFlight--;
      }),
    );
    await Promise.all(tasks);
    expect(peak).toBe(cap);
    expect(inFlight).toBe(0);
  });

  it('makes progress with concurrency = 1 (single-slot serialisation)', async () => {
    const limit = createLimiter(1);
    const order: number[] = [];
    await Promise.all(
      [10, 5, 1].map((delay, i) =>
        limit(async () => {
          await new Promise((r) => setTimeout(r, delay));
          order.push(i);
        }),
      ),
    );
    expect(order).toEqual([0, 1, 2]);
  });

  it('propagates rejections without starving subsequent tasks', async () => {
    const limit = createLimiter(2);
    const results: Array<{ ok: boolean; value: unknown }> = [];
    const tasks = [
      limit(async () => {
        throw new Error('boom');
      }),
      limit(async () => 'ok-a'),
      limit(async () => {
        throw new Error('boom-2');
      }),
      limit(async () => 'ok-b'),
    ];
    for (const t of tasks) {
      try {
        results.push({ ok: true, value: await t });
      } catch (err) {
        results.push({ ok: false, value: err });
      }
    }
    expect(results[0].ok).toBe(false);
    expect(results[1]).toEqual({ ok: true, value: 'ok-a' });
    expect(results[2].ok).toBe(false);
    expect(results[3]).toEqual({ ok: true, value: 'ok-b' });
  });

  it('continues after a pending fn never resolves (does not starve independent tasks)', async () => {
    const limit = createLimiter(2);
    const pending = defer<never>();
    // A task that never resolves occupies one slot forever.
    const _forever = limit(() => pending.promise);
    // Other tasks must still progress in the remaining slot.
    const [a, b, c] = await Promise.all([
      limit(async () => 'a'),
      limit(async () => 'b'),
      limit(async () => 'c'),
    ]);
    expect([a, b, c]).toEqual(['a', 'b', 'c']);
    // Prevent an unhandled-rejection warning when the test's Node
    // process exits.
    pending.reject(new Error('cancelled'));
    await _forever.catch(() => {});
  });

  it('rejects invalid concurrency at construction time', () => {
    expect(() => createLimiter(0)).toThrow(RangeError);
    expect(() => createLimiter(-1)).toThrow(RangeError);
    expect(() => createLimiter(1.5)).toThrow(RangeError);
    expect(() => createLimiter(Number.NaN)).toThrow(RangeError);
  });

  it('default concurrency is 6 (matches typical browser per-host limit)', () => {
    expect(DEFAULT_CONCURRENCY).toBe(6);
  });
});

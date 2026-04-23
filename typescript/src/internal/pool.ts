// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Bounded-concurrency task limiter.
 *
 * `@internal` — not re-exported from `index.ts`.
 *
 * `prefetchLayouts`, `messageObjectBatch`, and the descriptor-prefix
 * fetch all fan out to many concurrent HTTP Range GETs.  Browsers cap
 * simultaneous connections at ~6 per host; raw `Promise.all` over
 * hundreds of targets exceeds that cap, queues in the browser, and
 * blocks unrelated app traffic while holding a large heap footprint.
 *
 * The limiter returned by [`createLimiter`] wraps each task so that at
 * most `concurrency` are running at any moment.  Tasks resolve in the
 * order they are submitted (FIFO), letting callers build
 * `Promise.all(tasks.map(limiter))` with predictable scheduling.
 */

/** Default concurrency limit for Range-heavy operations. */
export const DEFAULT_CONCURRENCY = 6;

/** A thunk plus the deferred slot that will receive its result. */
interface Slot<T> {
  fn: () => Promise<T>;
  resolve: (v: T) => void;
  reject: (err: unknown) => void;
}

/**
 * Create a FIFO limiter that allows at most `concurrency` outstanding
 * tasks.  Returns a function that accepts an async thunk and returns
 * a promise that resolves (or rejects) with the thunk's result.
 *
 * `concurrency` must be a positive integer.
 */
export function createLimiter(
  concurrency: number = DEFAULT_CONCURRENCY,
): <T>(fn: () => Promise<T>) => Promise<T> {
  if (!Number.isInteger(concurrency) || concurrency < 1) {
    throw new RangeError(`concurrency must be a positive integer, got ${concurrency}`);
  }

  let active = 0;
  const queue: Slot<unknown>[] = [];

  const runNext = (): void => {
    while (active < concurrency && queue.length > 0) {
      const slot = queue.shift();
      if (!slot) break;
      active++;
      slot
        .fn()
        .then(
          (v) => slot.resolve(v),
          (err) => slot.reject(err),
        )
        .finally(() => {
          active--;
          runNext();
        });
    }
  };

  return <T>(fn: () => Promise<T>): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      queue.push({
        fn: fn as () => Promise<unknown>,
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      runNext();
    });
}

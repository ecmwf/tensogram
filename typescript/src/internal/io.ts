// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Tiny shared helpers for file / URL / error formatting paths.
 *
 * `@internal` — not re-exported from `index.ts`.
 */

/** Format any thrown value as a short human-readable string. */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

/**
 * Attach a `cause` to an error and return it — equivalent to the
 * ES2022 `{ cause }` option to `Error` constructors but usable with
 * our custom error classes that don't take an options bag.
 */
export function withCause<E extends Error>(err: E, cause: unknown): E {
  Object.defineProperty(err, 'cause', { value: cause, configurable: true });
  return err;
}

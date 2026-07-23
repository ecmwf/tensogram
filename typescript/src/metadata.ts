// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Metadata helpers.
 *
 * Mirrors the Rust / Python / CLI lookup semantics:
 *   - Dot-path traversal (`"mars.param"`, `"grib.geography.Ni"`)
 *   - First-match across `base[0]`, `base[1]`, ... (skipping `_reserved_`
 *     in each entry)
 *   - Fallback to `_extra_`
 *   - Explicit `_extra_.x` / `extra.x` prefix short-circuits to `_extra_`
 *   - `_reserved_` is always hidden from dict-style access
 *
 * `computeCommon()` mirrors `tensogram_core::compute_common`: returns the
 * set of keys present with identical values across **every** `base[i]`
 * entry. Useful for display, merge, or application-layer compaction.
 */

import type { BaseEntry, CborValue, GlobalMetadata } from './types.js';

const RESERVED = '_reserved_';
const EXTRA = '_extra_';

/**
 * Look up a dotted key in a `GlobalMetadata`.
 *
 * @returns the value, or `undefined` if not found.
 */
export function getMetaKey(meta: GlobalMetadata, path: string): CborValue | undefined {
  if (typeof path !== 'string' || path.length === 0) return undefined;

  // Explicit _extra_ / extra prefix bypasses base search.
  const parts = path.split('.');
  if (parts[0] === EXTRA || parts[0] === 'extra') {
    if (parts.length === 1) {
      return (meta._extra_ ?? undefined) as CborValue | undefined;
    }
    return resolvePath(meta._extra_ ?? undefined, parts.slice(1));
  }

  // `_reserved_` is always hidden from lookup.
  if (parts[0] === RESERVED || parts[0] === 'reserved') {
    return undefined;
  }

  // First-match across base entries (skipping _reserved_ in each).
  if (meta.base) {
    for (const entry of meta.base) {
      const found = resolveEntryPath(entry, parts);
      if (found !== undefined) return found;
    }
  }

  // Fall back to _extra_.
  return resolvePath(meta._extra_ ?? undefined, parts);
}

function resolveEntryPath(entry: BaseEntry, parts: readonly string[]): CborValue | undefined {
  if (entry == null) return undefined;
  // Inside an entry, skip _reserved_ regardless of the path's prefix.
  if (parts[0] === RESERVED) return undefined;
  return resolvePath(entry, parts);
}

/**
 * Message-level existence check for a dotted key.
 *
 * Thin convenience over {@link getMetaKey}: `true` iff the key resolves to
 * a present value (of any type, including CBOR `null` — decoded as JS
 * `null`). Because absent keys yield `undefined`, `undefined` is the
 * unambiguous "absent" signal and a stored `null`/`0`/`""` still counts
 * as present.
 *
 * @returns `true` if the key is present, `false` if absent.
 */
export function hasMetaKey(meta: GlobalMetadata, path: string): boolean {
  return getMetaKey(meta, path) !== undefined;
}

/**
 * Look up a dotted key **scoped to a single `base[obj]` entry**.
 *
 * Per-object semantics (mirrors core `get_at`): no cross-object
 * first-match, no `_extra_` fallback, and `_reserved_` is hidden at the
 * first path segment. A key present only in a *different* `base[i]` is not
 * visible here.
 *
 * @param obj  zero-based index into `meta.base`.
 * @returns the value, or `undefined` if the path is empty, `obj` is out of
 *   range, or the key is absent within that entry. (`undefined` = absent;
 *   a stored `null`/`0`/`""` is returned as-is.)
 */
export function getMetaKeyAt(meta: GlobalMetadata, obj: number, path: string): CborValue | undefined {
  if (typeof path !== 'string' || path.length === 0) return undefined;
  const entry = meta.base?.[obj];
  if (entry === undefined) return undefined;
  return resolveEntryPath(entry, path.split('.'));
}

/**
 * Per-object existence check for a dotted key.
 *
 * Thin convenience over {@link getMetaKeyAt} with the same per-object
 * scoping (no first-match, no `_extra_` fallback, `_reserved_` hidden).
 *
 * @returns `true` if the key is present in `base[obj]`, else `false`.
 */
export function hasMetaKeyAt(meta: GlobalMetadata, obj: number, path: string): boolean {
  return getMetaKeyAt(meta, obj, path) !== undefined;
}

// ── Typed getters (precise — no coercion) ─────────────────────────────────
//
// Each returns the value only when it is *both* present *and* of the exact
// expected JS type; otherwise `undefined` (mirroring the "undefined =
// absent OR wrong-type" contract of the core `as_*` extractors). A present
// value of the wrong type is indistinguishable from absent through these
// helpers — use {@link getMetaKey} / {@link hasMetaKey} when you need to
// tell them apart.

/**
 * Message-level string getter. Returns the value only if present *and* a
 * JS `string` (mirrors core `as_str`); otherwise `undefined`.
 */
export function getMetaString(meta: GlobalMetadata, path: string): string | undefined {
  const v = getMetaKey(meta, path);
  return typeof v === 'string' ? v : undefined;
}

/**
 * Message-level integer getter. Returns a present integer across the full
 * range (mirrors core `as_i64` / `as_u64`): a safe JS `number` for
 * `Number.isInteger` values, or a `bigint` for CBOR integers beyond the JS
 * safe-integer range (see {@link CborValue}). `undefined` for a non-integer,
 * absent, or wrong-type value. A `bigint` result cannot be mixed with
 * `number` arithmetic without an explicit conversion.
 */
export function getMetaInt(meta: GlobalMetadata, path: string): number | bigint | undefined {
  const v = getMetaKey(meta, path);
  if (typeof v === 'bigint') return v;
  return typeof v === 'number' && Number.isInteger(v) ? v : undefined;
}

/**
 * Coerce a CBOR value to a float, mirroring core `as_f64`: a present `number`
 * as-is — including `NaN` / `±Infinity`, which are genuine float values, not
 * "absent" — or a `bigint` (a CBOR integer beyond the JS safe-integer range,
 * per {@link CborValue}) widened to `number` like Rust's `n as f64` (which
 * may lose precision, or saturate to `±Infinity` for an out-of-range magnitude).
 * `undefined` only for a non-numeric value.
 */
function asFloat(v: CborValue | undefined): number | undefined {
  if (typeof v === 'number') return v;
  if (typeof v === 'bigint') return Number(v);
  return undefined;
}

/**
 * Message-level float getter. Returns a present numeric value as a JS
 * `number` — integers included (mirrors core `as_f64`, which widens ints to
 * float), a large `bigint` widened, and `NaN` / `±Infinity` passed through;
 * `undefined` only for a non-numeric or absent value.
 */
export function getMetaFloat(meta: GlobalMetadata, path: string): number | undefined {
  return asFloat(getMetaKey(meta, path));
}

/**
 * Message-level boolean getter. Returns the value only if present *and* a
 * JS `boolean` (mirrors core `as_bool`); otherwise `undefined`.
 */
export function getMetaBool(meta: GlobalMetadata, path: string): boolean | undefined {
  const v = getMetaKey(meta, path);
  return typeof v === 'boolean' ? v : undefined;
}

/**
 * Per-object string getter, scoped to `base[obj]`. See
 * {@link getMetaString} for typing and {@link getMetaKeyAt} for scoping.
 */
export function getMetaStringAt(meta: GlobalMetadata, obj: number, path: string): string | undefined {
  const v = getMetaKeyAt(meta, obj, path);
  return typeof v === 'string' ? v : undefined;
}

/**
 * Per-object integer getter, scoped to `base[obj]`. See {@link getMetaInt}
 * for typing and {@link getMetaKeyAt} for scoping.
 */
export function getMetaIntAt(
  meta: GlobalMetadata,
  obj: number,
  path: string,
): number | bigint | undefined {
  const v = getMetaKeyAt(meta, obj, path);
  if (typeof v === 'bigint') return v;
  return typeof v === 'number' && Number.isInteger(v) ? v : undefined;
}

/**
 * Per-object float getter, scoped to `base[obj]`. See {@link getMetaFloat}
 * for typing and {@link getMetaKeyAt} for scoping.
 */
export function getMetaFloatAt(meta: GlobalMetadata, obj: number, path: string): number | undefined {
  return asFloat(getMetaKeyAt(meta, obj, path));
}

/**
 * Per-object boolean getter, scoped to `base[obj]`. See {@link getMetaBool}
 * for typing and {@link getMetaKeyAt} for scoping.
 */
export function getMetaBoolAt(meta: GlobalMetadata, obj: number, path: string): boolean | undefined {
  const v = getMetaKeyAt(meta, obj, path);
  return typeof v === 'boolean' ? v : undefined;
}

function resolvePath(
  root: CborValue | { readonly [k: string]: CborValue } | undefined,
  parts: readonly string[],
): CborValue | undefined {
  if (root === undefined || root === null) return undefined;
  let cur: unknown = root;
  for (const part of parts) {
    if (cur === null || typeof cur !== 'object' || Array.isArray(cur)) {
      return undefined;
    }
    const obj = cur as Record<string, unknown>;
    if (!(part in obj)) return undefined;
    cur = obj[part];
  }
  return cur as CborValue | undefined;
}

/**
 * Extract keys that appear with identical values in every `base[i]`
 * entry. Mirrors `tensogram_core::compute_common`.
 *
 * `_reserved_` is ignored in the candidate set.
 *
 * Equality uses value-level semantics: arrays are compared elementwise,
 * objects key-by-key, and two `NaN`s are treated as equal (matching the
 * Rust `cbor_values_equal` helper).
 */
export function computeCommon(meta: GlobalMetadata): Record<string, CborValue> {
  const base = meta.base ?? [];
  if (base.length === 0) return {};
  if (base.length === 1) {
    const first = base[0];
    const out: Record<string, CborValue> = {};
    for (const [k, v] of Object.entries(first)) {
      if (k === RESERVED) continue;
      out[k] = v as CborValue;
    }
    return out;
  }

  const first = base[0];
  const result: Record<string, CborValue> = {};
  for (const [k, v] of Object.entries(first)) {
    if (k === RESERVED) continue;
    let allEqual = true;
    for (let i = 1; i < base.length; i++) {
      const entry = base[i];
      if (!(k in entry) || !cborValuesEqual(entry[k] as CborValue, v as CborValue)) {
        allEqual = false;
        break;
      }
    }
    if (allEqual) result[k] = v as CborValue;
  }
  return result;
}

/**
 * Value-level equality for `CborValue`. Mirrors Rust's
 * `cbor_values_equal` — positional map / array comparison, with any two
 * `NaN` values treated as equal. Note this is value-level, not bit-pattern,
 * equality: different NaN payloads are considered the same. That matches
 * the operational need because canonical CBOR (RFC 8949 §4.2) normalises
 * all NaN encodings, so distinguishing payloads has no observable effect
 * after a round-trip.
 */
export function cborValuesEqual(a: CborValue, b: CborValue): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;

  // Any NaN is treated as equal to any other NaN.
  if (typeof a === 'number' && typeof b === 'number') {
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    return a === b;
  }

  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!cborValuesEqual(a[i], b[i])) return false;
    }
    return true;
  }

  if (typeof a === 'object' && a !== null && typeof b === 'object' && b !== null) {
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    // Positional comparison: Rust stores canonical CBOR, so both maps
    // iterate in key-sorted order. JS property order is insertion order,
    // not key-sorted, so sort explicitly.
    ka.sort();
    kb.sort();
    for (let i = 0; i < ka.length; i++) {
      if (ka[i] !== kb[i]) return false;
      if (!cborValuesEqual((a as Record<string, CborValue>)[ka[i]], (b as Record<string, CborValue>)[kb[i]])) {
        return false;
      }
    }
    return true;
  }

  return false;
}

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
 * `cbor_values_equal` — positional map / array comparison, NaN bit-pattern
 * equality for floats.
 */
export function cborValuesEqual(a: CborValue, b: CborValue): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;

  // NaN equality by bit pattern (float64 route; JS numbers are IEEE 754 f64).
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

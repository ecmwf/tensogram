// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `simplePackingComputeParams` — GRIB-style simple-packing parameter
 * computation for a float64 array.
 *
 * Mirrors Rust `tensogram_encodings::simple_packing::compute_params` /
 * Python `tensogram.compute_packing_params` / FFI
 * `tgm_simple_packing_compute_params`.
 *
 * Returns snake-case keys matching the on-wire descriptor params so the
 * result can be spread directly into a descriptor:
 *
 * ```ts
 * const params = simplePackingComputeParams(values, 16);
 * const desc = {
 *   ...baseDescriptor,
 *   encoding: 'simple_packing',
 *   compression: 'szip',
 *   ...params,  // contributes reference_value, binary_scale_factor, …
 * };
 * ```
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError } from './errors.js';
import type { SimplePackingParams } from './types.js';

/** Maximum / minimum signed 32-bit integer — the range the WASM boundary accepts. */
const I32_MAX = 0x7fff_ffff;
const I32_MIN = -0x8000_0000;

/**
 * Compute simple-packing parameters for a set of f64 values.
 *
 * @param values                - `Float64Array` or plain array of `number`
 *                                (will be converted to `Float64Array`).
 *                                Every entry must be finite — NaN and
 *                                ±Infinity are rejected because they
 *                                produce meaningless scale factors.
 * @param bitsPerValue          - Quantization depth.  Typical values: 16,
 *                                24, 32.  `0` produces a constant-field
 *                                packing (no payload bytes emitted).
 *                                Must satisfy `0 ≤ bitsPerValue ≤ 64`.
 * @param decimalScaleFactor    - Power-of-10 scale applied before packing.
 *                                Default `0`.  Must fit in signed 32-bit
 *                                integer range.
 * @returns                     - Snake-cased {@link SimplePackingParams}.
 * @throws {EncodingError}      - NaN detected (in the underlying encoder).
 * @throws {InvalidArgumentError} - Bad argument type/value, non-finite
 *                                  sample, out-of-range scale factor or
 *                                  bit width.
 */
export function simplePackingComputeParams(
  values: Float64Array | readonly number[],
  bitsPerValue: number,
  decimalScaleFactor = 0,
): SimplePackingParams {
  if (!Number.isInteger(bitsPerValue) || bitsPerValue < 0 || bitsPerValue > 64) {
    throw new InvalidArgumentError(
      `bitsPerValue must be an integer in [0, 64], got ${String(bitsPerValue)}`,
    );
  }
  if (
    !Number.isInteger(decimalScaleFactor) ||
    decimalScaleFactor < I32_MIN ||
    decimalScaleFactor > I32_MAX
  ) {
    throw new InvalidArgumentError(
      `decimalScaleFactor must be a signed 32-bit integer, got ${String(decimalScaleFactor)}`,
    );
  }
  const f64 =
    values instanceof Float64Array ? values : Float64Array.from(values as readonly number[]);
  // Reject ±Infinity here — the Rust core only guards against NaN,
  // and ±Inf would produce a nonsensical `binary_scale_factor`.
  for (let i = 0; i < f64.length; i++) {
    if (!Number.isFinite(f64[i])) {
      throw new InvalidArgumentError(
        `values[${i}] must be finite, got ${f64[i]}`,
      );
    }
  }
  const wbg = getWbg();
  return rethrowTyped(
    () =>
      wbg.simple_packing_compute_params(f64, bitsPerValue, decimalScaleFactor) as SimplePackingParams,
  );
}

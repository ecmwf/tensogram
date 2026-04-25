// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `doctor` — environment diagnostics for the Tensogram WASM build.
 *
 * Returns the same data model as the `tensogram doctor` CLI subcommand
 * (see `docs/src/cli/doctor.md`) but limited to the core encode/decode
 * pipeline and the codecs that were compiled into the WASM bundle.
 * Converter rows (`grib`, `netcdf`) are CLI-only and not present here.
 */

import { getWbg } from './init.js';
import { rethrowTyped } from './errors.js';

/** Linkage model for a backend library. */
export type Linkage = 'ffi' | 'pure-rust';

/** Broad category of a compiled-in feature. */
export type FeatureKind = 'compression' | 'threading' | 'io' | 'converter';

/** Build profile reported by `BuildInfo.profile`.  Closed set, mirrors the
 *  `cfg!(debug_assertions)` discriminator on the Rust side. */
export type BuildProfile = 'release' | 'debug';

/** Compile-time build metadata (mirrors `BuildInfo` in Rust). */
export interface BuildInfo {
  /** Crate version from `Cargo.toml`, e.g. `"0.19.0"`. */
  version: string;
  /** Wire-format version integer, e.g. `3`. */
  wire_version: number;
  /** Rustc target triple, e.g. `"wasm32-unknown-unknown"`. */
  target: string;
  /** Build profile: `"release"` or `"debug"`. */
  profile: BuildProfile;
}

/**
 * One compiled-in feature row.  `state === 'on'` rows additionally carry
 * `backend`, `linkage`, and (where available) `version`.
 */
export type FeatureStatus =
  | {
      name: string;
      kind: FeatureKind;
      state: 'on';
      backend: string;
      linkage: Linkage;
      version: string | null;
    }
  | {
      name: string;
      kind: FeatureKind;
      state: 'off';
    };

/** Outcome of a single self-test row. */
export type SelfTestResult =
  | { label: string; outcome: 'ok' }
  | { label: string; outcome: 'failed'; error: string }
  | { label: string; outcome: 'skipped'; reason: string };

/** Top-level doctor report. */
export interface DoctorReport {
  build: BuildInfo;
  features: FeatureStatus[];
  self_test: SelfTestResult[];
}

/**
 * Collect environment diagnostics: build metadata, compiled-in feature
 * states, and core encode/decode self-test results.
 *
 * Mirrors the Rust `tensogram::doctor::run_diagnostics()` and the
 * `tensogram doctor` CLI subcommand.  The WASM build does **not** run
 * the GRIB or NetCDF converter self-tests — those features are CLI-only
 * — so the `self_test` array covers only the core encode/decode pipeline
 * plus the codecs that were compiled into this WASM bundle.
 *
 * @example
 * ```typescript
 * import init, { doctor } from "@ecmwf.int/tensogram";
 * await init();
 * const report = doctor();
 * console.log(report.build.version, report.build.target);
 * for (const f of report.features) {
 *   console.log(f.name, f.state);
 * }
 * ```
 */
export function doctor(): DoctorReport {
  const wbg = getWbg();
  return rethrowTyped(() => wbg.doctor()) as DoctorReport;
}

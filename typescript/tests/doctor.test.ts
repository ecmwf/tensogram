// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `doctor` — environment diagnostics test.  Locks in the schema shape
 * promised in `docs/src/cli/doctor.md`.
 */

import { describe, expect, it } from 'vitest';
import { doctor } from '../src/index.js';
import { initOnce } from './helpers.js';

describe('doctor — environment diagnostics', () => {
  initOnce();

  it('returns the three top-level sections', () => {
    const report = doctor();
    expect(report.build).toBeDefined();
    expect(Array.isArray(report.features)).toBe(true);
    expect(Array.isArray(report.self_test)).toBe(true);
  });

  it('build section carries version + wire-format + target + profile', () => {
    const { build } = doctor();
    expect(typeof build.version).toBe('string');
    expect(build.version.length).toBeGreaterThan(0);
    expect(typeof build.wire_version).toBe('number');
    expect(build.wire_version).toBeGreaterThan(0);
    expect(typeof build.target).toBe('string');
    expect(build.target.length).toBeGreaterThan(0);
    expect(['release', 'debug']).toContain(build.profile);
  });

  it('emits at least one feature row, all with name + kind + state', () => {
    const { features } = doctor();
    expect(features.length).toBeGreaterThan(0);
    for (const f of features) {
      expect(typeof f.name).toBe('string');
      expect(typeof f.kind).toBe('string');
      expect(['on', 'off']).toContain(f.state);
    }
  });

  it('on-state feature rows carry backend + linkage', () => {
    const { features } = doctor();
    const onRows = features.filter((f) => f.state === 'on');
    expect(onRows.length).toBeGreaterThan(0);
    for (const f of onRows) {
      // TS narrows the union here.
      if (f.state === 'on') {
        expect(typeof f.backend).toBe('string');
        expect(['ffi', 'pure-rust']).toContain(f.linkage);
      }
    }
  });

  it('off-state feature rows have no backend / linkage / version', () => {
    const { features } = doctor();
    const offRows = features.filter((f) => f.state === 'off');
    for (const f of offRows) {
      // The serde flatten + tagged enum produces a strict shape; check it.
      expect(Object.keys(f).sort()).toEqual(['kind', 'name', 'state']);
    }
  });

  it('self-test rows carry label + outcome', () => {
    const { self_test } = doctor();
    expect(self_test.length).toBeGreaterThan(0);
    for (const r of self_test) {
      expect(typeof r.label).toBe('string');
      expect(['ok', 'failed', 'skipped']).toContain(r.outcome);
    }
  });

  it('passes its own self-test on a healthy build', () => {
    // Every WASM-compiled codec self-test should pass.  If a future
    // change breaks the round-trip, this surfaces immediately.
    const { self_test } = doctor();
    const failures = self_test.filter((r) => r.outcome === 'failed');
    expect(failures, `self-test failures: ${JSON.stringify(failures, null, 2)}`).toHaveLength(0);
  });
});

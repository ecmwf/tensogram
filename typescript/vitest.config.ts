// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    globals: false,
    testTimeout: 30_000,
    // WASM init is global; run tests sequentially within a file.
    // Across files vitest's default thread pool is fine because each worker
    // instantiates its own WASM module.
    pool: 'threads',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary'],
      include: ['src/**/*.ts'],
      // types.ts is pure TypeScript type declarations with zero runtime
      // code — it contributes 0/0 statements and would skew the total.
      exclude: ['src/types.ts'],
      thresholds: {
        lines: 90,
        functions: 90,
        statements: 90,
        branches: 80,
      },
    },
    benchmark: {
      include: ['tests/**/*.bench.ts'],
    },
  },
});

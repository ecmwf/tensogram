# Design: Publish TypeScript Package to npmjs.com

**Date:** 2026-04-21
**Status:** Approved

## Summary

A GitHub Actions workflow (`publish-npm.yml`) that publishes `@ecmwf.int/tensogram` to the public npm registry on manual dispatch. Consistent in style and structure with the existing `publish-crates.yml` and `publish-pypi-tensogram.yml` workflows.

## Trigger

`workflow_dispatch` only — same as all other publish workflows in this repo.

## Runner

Self-hosted Linux runner with the ECMWF CI Docker image (`eccr.ecmwf.int/tensogram/ci:1.2.0`). This image already has Rust and wasm-pack installed, exactly as used by the `typescript` CI job. Requires the same ECMWF docker registry credentials already present in repo secrets.

## GitHub Environment

Named environment: `npm`. Mirrors the `crates-io` and `pypi` environments. Allows gating the `NPM_TOKEN` secret and optionally adding required reviewers.

**Required secret:** `NPMJS_API_TOKEN` — a publish-scoped npm access token for the `@ecmwf.int` scope, stored in the `npm` environment.

## Steps

1. `actions/checkout@v4` with `submodules: true`
2. `mkdir -p "$TMPDIR"` (matches pattern used by all other CI jobs)
3. Install Node.js from the `NODE_VERSION` env var (same version pinned in `ci.yml`)
4. Build WASM: `wasm-pack build rust/tensogram-wasm --release --target web --out-dir ../../typescript/wasm --out-name tensogram_wasm`
5. Install TypeScript deps: `npm ci` in `typescript/`
6. Build distributable: `npx tsc` in `typescript/`
7. **Package rename:** `typescript/package.json` `name` field changes from `@ecmwf/tensogram` to `@ecmwf.int/tensogram`. All import statements and `package.json` dependency references across the repo are updated to match (40 files).

**Idempotency check:** Query `https://registry.npmjs.org/%40ecmwf.int%2Ftensogram/<version>`. HTTP 200 → already published, skip (exit 0). HTTP 404 → proceed.
8. Write `.npmrc`: `//registry.npmjs.org/:_authToken=${NODE_AUTH_TOKEN}` (inline, since Node is installed manually rather than via `actions/setup-node`)
9. `npm publish --access public` in `typescript/` with `NODE_AUTH_TOKEN: ${{ secrets.NPMJS_API_TOKEN }}`

## Package rename

The package name changes from `@ecmwf/tensogram` → `@ecmwf.int/tensogram`. This affects 40 files (source imports, `package.json` dependency entries, lock files, and documentation). Lock files are regenerated rather than hand-edited.

## What is not included

- No sccache — single publish run, not worth the configuration overhead.
- No build matrix — unlike Python wheels, the WASM + TS build is platform-independent.
- No two-job split — no cross-platform artifact fan-in needed; a single job is sufficient.

## Prerequisites for first use

1. Create a `npm` environment in the GitHub repo settings.
2. Add `NPMJS_API_TOKEN` (publish-scoped npm token for `@ecmwf.int`) to that environment's secrets.
3. Verify the `@ecmwf.int` scope is registered on npmjs.com (org names with dots are non-standard — confirm availability before the first publish).

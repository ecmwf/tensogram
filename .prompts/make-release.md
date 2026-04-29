---
description: "Cut a new release — usage: /make-release <X.Y.Z>"
agent: build
---

# Make Release

Create a new release of Tensogram.

## Arguments

Provide the version as argument, e.g. `/make-release 0.14.0`
If no version given, auto-increment MINOR from the current VERSION file.

## Pre-release Checks

Run ALL of the following. If ANY step fails, STOP and prompt the user.

### 1. Clean Working Tree
```bash
git status  # must be clean — all changes committed and pushed
git diff --stat origin/main  # must be empty
```

### 2. Full Build and Test
```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test -p tensogram --features "remote,async"
cargo test --manifest-path rust/tensogram-grib/Cargo.toml
cargo test --manifest-path rust/tensogram-netcdf/Cargo.toml
```

### 3. Python
```bash
source .venv/bin/activate
uv pip install -e "python/tensogram-xarray/[dask]"
uv pip install -e python/tensogram-zarr/
ruff check --config python/bindings/pyproject.toml python/tests/
python -m pytest python/tests/ -v
python -m pytest python/tensogram-xarray/tests/ -v
python -m pytest python/tensogram-zarr/tests/ -v
```

### 4. Examples Build
```bash
cargo build --workspace
```

### 5. Documentation
```bash
which mdbook && mdbook build docs/ || echo "mdbook not installed, skip"
```

If ANY of the above fails, STOP and report the failure to the user.

## Release Process

### 1. Version Bump

Read the current version from `VERSION` file. Bump to the target version.

Update ALL of these locations (they MUST all match):
- `VERSION` (the source of truth)
- Root `Cargo.toml` `[workspace.package] version` — since #74 the 9
  in-workspace member crates (tensogram, tensogram-encodings, tensogram-szip,
  tensogram-sz3, tensogram-sz3-sys, tensogram-cli, tensogram-ffi, benchmarks,
  examples/rust) inherit via `version.workspace = true`, so this single edit
  covers all of them.
- `Cargo.toml` of each EXCLUDED crate (not workspace members — must be bumped
  individually): tensogram-python (at `python/bindings/Cargo.toml`),
  tensogram-grib, tensogram-netcdf, tensogram-wasm.
- `pyproject.toml` in EVERY Python package: python/bindings, python/tensogram-xarray,
  python/tensogram-zarr, python/tensogram-anemoi, examples/jupyter
- `typescript/package.json` AND `examples/typescript/package.json`
- `CHANGELOG.md` — add new release entry header with today's date

Also bump inter-crate dependency version pins (`version = "=X.Y.Z"` in path
dependencies). These locations contain exact version pins that must match:
- Root `Cargo.toml` workspace deps (tensogram-szip, tensogram-sz3, tensogram-sz3-sys)
- `rust/tensogram-sz3/Cargo.toml` dep on tensogram-sz3-sys
- `rust/tensogram/Cargo.toml` dep on tensogram-encodings
- `rust/tensogram-grib/Cargo.toml` deps on tensogram + tensogram-encodings
- `rust/tensogram-netcdf/Cargo.toml` deps on tensogram + tensogram-encodings
- `rust/tensogram-cli/Cargo.toml` deps on tensogram + tensogram-grib + tensogram-netcdf
- `rust/tensogram-ffi/Cargo.toml` deps on tensogram + tensogram-encodings
- `rust/tensogram-wasm/Cargo.toml` dep on tensogram
- `python/tensogram-anemoi/pyproject.toml` dependency pin on `tensogram`
  (uses `>=X.Y.Z,<X.(Y+1)` — bump both bounds, mirrors the jupyter example)
- `examples/jupyter/pyproject.toml` dependency pins on `tensogram` and
  `tensogram[xarray]` (uses `>=X.Y.Z,<X.(Y+1).0` — bump both bounds)

Verify no stale version strings remain in first-party files only:
```bash
# Rust
grep -r 'version = "OLD_VERSION"' --include='Cargo.toml' \
  VERSION Cargo.toml rust/tensogram-*/Cargo.toml python/bindings/Cargo.toml \
  rust/benchmarks/Cargo.toml examples/rust/Cargo.toml
# Python
grep 'version = "OLD_VERSION"' \
  python/bindings/pyproject.toml \
  python/tensogram-xarray/pyproject.toml \
  python/tensogram-zarr/pyproject.toml \
  python/tensogram-anemoi/pyproject.toml \
  examples/jupyter/pyproject.toml
# TypeScript
grep '"version": "OLD_VERSION"' \
  typescript/package.json \
  examples/typescript/package.json
```

**WARNING**: Do NOT grep all `pyproject.toml` files recursively — the vendored
`rust/tensogram-sz3-sys/SZ3/tools/pysz/pyproject.toml` has `version = "1.0.3"`
which must NOT be changed.

### 2. Write CHANGELOG Entry

Add an entry to `CHANGELOG.md` following the existing format (Keep a Changelog).
Summarize changes since the last release using `git log <last-tag>..HEAD --oneline`.
Organize into Added / Changed / Fixed / Removed sections.
Include a Stats section with test counts and coverage.

### 3. Commit and Push
```bash
cargo check --workspace  # verify Cargo.lock updates cleanly
git add -A  # but NOT build artifacts
git commit -m "chore: release X.Y.Z"
git push
```

### 4. Preflight and Tag
```bash
# Trigger release-preflight.yml workflow (workflow_dispatch, input: version)
# Wait for it to pass — all green required.
# release-preflight now also exercises `cargo cbuild -p tensogram-ffi` and
# the C-API smoke test, so a broken cargo-c metadata or header drift is
# caught here before tagging.
git tag X.Y.Z
git push --tags
```

Pushing the tag also triggers `publish-ffi.yml`, which builds the four
release tarballs (Linux x86_64/aarch64, macOS x86_64/aarch64) and
**creates the GitHub Release** with binaries attached. This runs in
parallel with the registry publishes below; you must wait for it to
finish before editing release notes (step 6).

### 5. Publish to Registries

Trigger each workflow manually and wait for completion between steps.
All three registry workflows are `workflow_dispatch` only — none triggers
the next automatically.

1. Run `publish-crates.yml` (workflow_dispatch) — publishes 10 Rust crates
   in strict dependency order: tensogram-szip → tensogram-sz3-sys →
   tensogram-sz3 → tensogram-encodings → tensogram → tensogram-grib →
   tensogram-netcdf → tensogram-ffi → tensogram-cli → tensogram-wasm.
   The workflow skips any crate whose version is already on crates.io,
   so it is safe to re-run.
2. Smoke test: `cargo add tensogram@X.Y.Z` in a fresh project, `cargo build`
3. Run `publish-pypi.yml` (workflow_dispatch, input `use_test_pypi=false`
   for production, `true` for Test PyPI). This single workflow builds
   and publishes ALL Python packages in one run:
   - native `tensogram` wheels (Linux x86_64 + macOS arm64, CPython
     3.11 / 3.12 / 3.13 / 3.14 + free-threaded 3.13t / 3.14t)
   - pure-Python extras: `tensogram-xarray`, `tensogram-zarr`,
     `tensogram-anemoi` (built via `make python-dist-extras`)
   Uses `PYPI_API_TOKEN` (production) or `PYPI_TEST_API_TOKEN` (test).
4. Wait for PyPI propagation (~1–2 min), then smoke test each package:
   ```bash
   pip install tensogram==X.Y.Z
   pip install tensogram-xarray==X.Y.Z
   pip install tensogram-zarr==X.Y.Z
   pip install tensogram-anemoi==X.Y.Z
   ```
5. Run `publish-npm.yml` (workflow_dispatch) — publishes
   `@ecmwf.int/tensogram` to npmjs.com. Runs on the self-hosted
   `platform-builder-docker-xl` runner inside the
   `eccr.ecmwf.int/tensogram/ci:1.3.0` image (Node 22.22.2 pre-baked),
   builds WASM + TypeScript via `make ts-install` and `make ts-build`,
   checks `https://registry.npmjs.org/@ecmwf.int/tensogram/X.Y.Z`, and
   only publishes if the version is missing (safe to re-run).
   **About the token name**: the workflow reads `TEMPORARY_NPMJS_APIKEY`
   (see `publish-npm.yml`). "Temporary" refers to the planned migration
   to npm trusted publishing (OIDC) once it is available for the
   `@ecmwf.int` scope — the secret itself IS the production credential
   and is the one to use for real releases.
6. Wait for npm propagation (~1 min), then smoke test:
   ```bash
   npm view @ecmwf.int/tensogram@X.Y.Z version
   # optionally in a throwaway project:
   npm install @ecmwf.int/tensogram@X.Y.Z
   ```

### 6. Edit GitHub Release notes

The release was already created by `publish-ffi.yml` (step 4) with the
four binary tarballs attached. **Wait for that specific run to finish**
before editing notes (`gh release edit` fails if the release does not
yet exist):

```bash
# Identify the publish-ffi.yml run for THIS tag's commit, not just the
# latest run on the workflow (which could be a manual rerun for another
# tag). Filtering by --commit guarantees we watch the correct run.
TAG_SHA=$(git rev-list -n1 X.Y.Z)
RUN_ID=$(gh run list --workflow=publish-ffi.yml \
    --event push --commit "$TAG_SHA" \
    --limit 1 --json databaseId --jq '.[0].databaseId')
test -n "$RUN_ID" || { echo "no publish-ffi.yml run for $TAG_SHA"; exit 1; }
gh run watch "$RUN_ID"

# Verify the four expected assets are attached.
gh release view X.Y.Z --json assets --jq '.assets[].name'

# Now add release notes (Highlights, Added/Changed/Fixed, Stats,
# link to the full CHANGELOG entry).
gh release edit X.Y.Z --title "X.Y.Z" --notes-file <(cat <<'EOF'
## Highlights
...
EOF
)
```

If the FFI binary release workflow failed, re-trigger it manually
before editing notes (workflow_dispatch picks up the `--ref` you pass
on the CLI; pass the tag name so the rerun checks out the correct
source):
```bash
gh workflow run publish-ffi.yml --ref X.Y.Z -f version=X.Y.Z
```

**IMPORTANT:** Version tags are bare semver (e.g. `0.14.0`), NEVER prefixed with `v`.
**IMPORTANT:** NEVER bump MAJOR unless the user explicitly says so.
**IMPORTANT:** NEVER push to remote without explicit user approval.

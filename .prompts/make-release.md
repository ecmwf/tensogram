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
cargo test -p tensogram-core --features "remote,async"
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
- `Cargo.toml` in EVERY crate: tensogram-core, tensogram-encodings, tensogram-sz3,
  tensogram-sz3-sys, tensogram-szip, tensogram-cli, tensogram-ffi, tensogram-python,
  tensogram-grib, tensogram-netcdf, tensogram-wasm, benchmarks, examples/rust
- `pyproject.toml` in EVERY Python package: python/bindings, python/tensogram-xarray,
  python/tensogram-zarr
- `typescript/package.json`
- `CHANGELOG.md` — add new release entry header with today's date

Also bump inter-crate dependency version pins (`version = "=X.Y.Z"` in path
dependencies). These locations contain exact version pins that must match:
- Root `Cargo.toml` workspace deps (tensogram-szip, tensogram-sz3, tensogram-sz3-sys)
- `rust/tensogram-sz3/Cargo.toml` dep on tensogram-sz3-sys
- `rust/tensogram-core/Cargo.toml` dep on tensogram-encodings
- `rust/tensogram-grib/Cargo.toml` deps on tensogram-core + tensogram-encodings
- `rust/tensogram-netcdf/Cargo.toml` deps on tensogram-core + tensogram-encodings
- `rust/tensogram-cli/Cargo.toml` deps on tensogram-core + tensogram-grib + tensogram-netcdf
- `rust/tensogram-ffi/Cargo.toml` deps on tensogram-core + tensogram-encodings
- `rust/tensogram-wasm/Cargo.toml` dep on tensogram-core

Verify no stale version strings remain in first-party files only:
```bash
grep -r 'version = "OLD_VERSION"' --include='Cargo.toml' \
  VERSION Cargo.toml rust/tensogram-*/Cargo.toml python/bindings/Cargo.toml \
  rust/benchmarks/Cargo.toml examples/rust/Cargo.toml
grep 'version = "OLD_VERSION"' \
  python/bindings/pyproject.toml \
  python/tensogram-xarray/pyproject.toml \
  python/tensogram-zarr/pyproject.toml
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
# Wait for it to pass — all green required
git tag X.Y.Z
git push --tags
```

### 5. Publish to Registries

Trigger each workflow manually and wait for completion between steps:

1. Run `publish-crates.yml` (workflow_dispatch) — publishes 10 Rust crates
2. Smoke test: `cargo add tensogram-core@X.Y.Z` in a fresh project, `cargo build`
3. Run `publish-pypi-tensogram.yml` (workflow_dispatch) — native Python wheels
4. Wait for PyPI propagation, then smoke test: `pip install tensogram==X.Y.Z`
5. Run `publish-pypi-extras.yml` (dispatch from the release tag) — xarray + zarr
6. Wait for PyPI propagation, then smoke test: `pip install tensogram-xarray==X.Y.Z`

### 6. Create GitHub Release
```bash
gh release create X.Y.Z --title "X.Y.Z" --notes "..."
```

Release notes should include: Highlights, key Added/Changed items, Stats, and a
link to the full CHANGELOG entry.

**IMPORTANT:** Version tags are bare semver (e.g. `0.14.0`), NEVER prefixed with `v`.
**IMPORTANT:** NEVER bump MAJOR unless the user explicitly says so.
**IMPORTANT:** NEVER push to remote without explicit user approval.

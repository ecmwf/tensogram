# Make Release

Create a new release of Tensogram.

## Arguments

Provide the version as argument, e.g. `/make-release 0.11.0`
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
```

### 3. Python
```bash
source .venv/bin/activate
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
- `Cargo.toml` in EVERY crate: tensogram-core, tensogram-encodings, tensogram-cli,
  tensogram-ffi, tensogram-python, tensogram-grib, tensogram-netcdf, tensogram-szip,
  tensogram-wasm, benchmarks, examples/rust
- `pyproject.toml` in EVERY Python package: python/bindings, python/tensogram-xarray,
  python/tensogram-zarr
- `CHANGELOG.md` — add new release entry header with today's date

Verify no stale version strings remain:
```bash
grep -r 'version = "OLD_VERSION"' --include='Cargo.toml' --include='pyproject.toml' .
```

### 2. Write CHANGELOG Entry

Add an entry to `CHANGELOG.md` following the existing format (Keep a Changelog).
Summarize changes since the last release using `git log <last-tag>..HEAD --oneline`.
Organize into Added / Changed / Fixed / Removed sections.
Include a Stats section with test counts and coverage.

### 3. Commit, Tag, Push
```bash
cargo check --workspace  # verify Cargo.lock updates cleanly
git add -A  # but NOT build artifacts
git commit -m "chore: release X.Y.Z"
git push
```

### 4. Create GitHub Release
```bash
gh release create X.Y.Z --title "X.Y.Z" --notes "..."
```

Release notes should include: Highlights, key Added/Changed items, Stats, and a
link to the full CHANGELOG entry.

**IMPORTANT:** Version tags are bare semver (e.g. `0.10.0`), NEVER prefixed with `v`.
**IMPORTANT:** NEVER bump MAJOR unless the user explicitly says so.

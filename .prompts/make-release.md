---
description: "Cut a new release — usage: /make-release <X.Y.Z>"
agent: build
---

# Make Release

Create a new release of Tensogram.

## Arguments

Provide the version as argument, e.g. `/make-release 0.14.0`
If no version given, auto-increment MINOR from the current VERSION file.

> **Canonical procedure:** `docs/src/dev/releasing.md`. This skill automates it;
> keep the two in sync.

## Pre-release Checks

Run these in order. If ANY step fails, STOP and prompt the user.

### 1. Clean working tree
```bash
git status                    # must be clean — all changes committed
git diff --stat origin/main   # must be empty (also pushed)
```

### 2. Full gate
```bash
make all            # build + test + lint across every language
make release-check  # version-check, crate packaging + leaf dry-run publish,
                    # cargo-c header-drift diff, wheels + twine, npm wasm-blob guard
```

`make all` + `make release-check` are the single source of truth for the local
release gate — do not re-list the underlying `cargo` / `ruff` / `pytest`
commands here. The grib/netcdf and macOS matrices and the real dry-run publishes
run on a clean checkout in the `release-preflight` CI workflow (step 4 below).

## Release Process

### 1. Version bump

`VERSION` is the single source of truth. Bump every manifest with the tooling —
never hand-edit them:

```bash
make bump-version VERSION=X.Y.Z   # rewrites every manifest, then greps for stragglers
make version-check                # confirms everything matches VERSION
```

`make bump-version` (wrapping `scripts/bump_version.py`) is the authoritative
list of what carries a version string — including the workspace `version`, the
excluded-crate Cargo.tomls, the internal `version = "=X.Y.Z"` pins, every Python
`pyproject.toml`, the `package.json` files, and `fortran/`. It already knows to
skip the vendored `rust/tensogram-sz3-sys/SZ3/.../pyproject.toml`. The full
contract is in AGENTS.md "Version control".

It deliberately does NOT touch `CHANGELOG.md` or the Python dependency
*constraint ranges* (`tensogram>=X.Y.Z,<X.(Y+1)`) — handle those by hand (next
step).

### 2. Write CHANGELOG Entry

Add an entry to `CHANGELOG.md` following the existing format (Keep a Changelog).
Summarize changes since the last release using `git log <last-tag>..HEAD --oneline`.
Organize into Added / Changed / Fixed / Removed sections.
Include a Stats section with test counts and coverage.

### 3. Commit and open a release PR

Releases land on `main` through a PR like every other change (see AGENTS.md
"Review & merge") — NEVER push the release commit straight to `main`.

```bash
cargo check --workspace                 # verify Cargo.lock updates cleanly
git checkout -b chore/release-X.Y.Z
git add -A                              # but NOT build artifacts (dist/, build/, target/)
git commit -m "chore: release X.Y.Z"
git push -u origin chore/release-X.Y.Z
gh pr create --base main --title "chore: release X.Y.Z" --body "release X.Y.Z"
```

Get it green/reviewed, then **squash-merge** and delete the branch (add
`--admin` only to bypass a check that is red for known-infra reasons):

```bash
gh pr merge --squash --delete-branch
git checkout main && git pull            # sync before tagging
```

### 4. Preflight and Tag

With the release PR squash-merged and `main` checked out:

```bash
# Trigger release-preflight.yml workflow (workflow_dispatch, input: version)
# Wait for it to pass — all green required.
# release-preflight also exercises `cargo cbuild -p tensogram-ffi` and the
# C-API smoke test, so broken cargo-c metadata or header drift is caught
# here before tagging.
git tag X.Y.Z                            # on main, NO 'v' prefix
git push origin X.Y.Z
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
# latest run on the workflow (which could be a run for another tag).
# Filter by --commit so we watch the run for this tag whether it was
# triggered by 'git push --tags' (event=push) or re-triggered manually
# via 'gh workflow run' (event=workflow_dispatch). The '// empty' jq
# fallback makes the variable empty (rather than the literal string
# "null") when no run exists yet, so the -z guard fires correctly.
TAG_SHA=$(git rev-list -n1 X.Y.Z)
RUN_ID=$(gh run list --workflow=publish-ffi.yml \
    --commit "$TAG_SHA" \
    --limit 1 --json databaseId --jq '.[0].databaseId // empty')
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

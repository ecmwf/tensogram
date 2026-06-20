# Releasing

This page is the canonical procedure for cutting a Tensogram release. It is
written for both human maintainers and coding agents. The short version lives
in `AGENTS.md`; this page adds the per-step detail, the rationale, and the
prerequisites.

## The gate model

A release passes through three gates, in order:

1. **`make all`** — the everyday build/test/lint gate. Compiles every language
   surface (Rust, Python, TypeScript, C++, WASM, cargo-c, Fortran) and runs the
   Rust/Python/TS test suites plus all lints.
2. **`make release-check`** — the *release-only* gates that `make all` does not
   run: version consistency, crate packaging, the C header-drift diff, Python
   wheel metadata, and the npm tarball contents. Run it **after** a green
   `make all`.
3. **`release-preflight.yml`** (GitHub Actions, `workflow_dispatch`) — the
   authoritative pre-tag gate. It runs the same checks on a clean checkout and
   additionally covers the grib/netcdf and macOS matrices and real dry-run
   publishes. Local `make release-check` is the fast, locally-runnable subset of
   this.

`make release-check` is not folded into `make all` because it is slower
(it builds distributable wheels, dry-run-publishes crates, and builds the
cargo-c C library) and only relevant when preparing a release.

## Prerequisites

`make release-check` needs more than the core Rust toolchain:

| Tool | Used by | Install |
| --- | --- | --- |
| `cargo-c` | `cargo-c-header-check`, the cargo-c leg of `make all` | `cargo install cargo-c --locked` (CI pins `0.10.21 --features vendored-openssl`) |
| `uv` | Python build/test, wheel build, `twine` | see `astral.sh/uv` |
| Node ≥ 20 | TypeScript build, `npm-pack-check` | nodejs.org |
| `wasm-pack` | WASM build | `cargo install wasm-pack` |
| `gfortran` + `pkg-config` | Fortran leg of `make all` | system package manager |

`maturin` and `patchelf` are installed automatically into `.venv` by the
Python targets (`maturin[patchelf]`).

## Step-by-step

### 1. Confirm the work is landed and recorded

`main` must be green, and every user-facing change must already be in
`CHANGELOG.md` under the `[Unreleased]` section. The changelog is the single
backward-looking record — there is no separate status file.

### 2. Bump the version

```bash
make bump-version VERSION=X.Y.Z
```

The `VERSION` file at the repo root is the **single source of truth**. This
command (wrapping `scripts/bump_version.py`) rewrites every manifest that
carries a version string and then greps the tree for stragglers. Never
hand-edit individual manifests.

SemVer rules:

- **MAJOR** — never bump without the user explicitly asking.
- **MINOR** — new features.
- **MICRO** — bug fixes and documentation.

Then, **by hand**, move the `[Unreleased]` changelog entries under a new
`## [X.Y.Z] - YYYY-MM-DD` heading. The bump script deliberately does not touch
`CHANGELOG.md` or the Python dependency *constraint ranges*
(`tensogram>=X.Y.Z,<X.Y+1`) — the `<` ceiling is a compatibility policy that is
surfaced for review rather than rewritten.

Verify everything is consistent at any time with:

```bash
make version-check
```

Why it matters: the provenance encoder reads the version via
`env!("CARGO_PKG_VERSION")`, so a manifest out of sync with `VERSION` stamps the
wrong provenance into encoded messages.

### 3. Run the full gate

```bash
make all
```

This must be green. It is now comprehensive across all languages, so it needs
the full toolchain (see Prerequisites). It is also idempotent — safe to re-run.

### 4. Run the release-readiness gate

```bash
make release-check
```

This runs, in order:

| Sub-target | Checks |
| --- | --- |
| `version-check` | Every manifest matches the `VERSION` file. |
| `feature-tests` | The optional-feature test surface (`remote`, `remote,async`) that `make all`'s default-feature run skips. |
| `crates-verify` | `cargo package --list` for every workspace crate + a real `cargo publish --dry-run` of the leaf crates (`tensogram-szip`, `tensogram-sz3-sys`), which compiles the packaged tarball and catches missing `include` files or bad metadata. |
| `cargo-c-header-check` | Diffs the in-tree `rust/tensogram-ffi/tensogram.h` against the header `cargo-c` generates — a drift guard for C/C++ consumers. |
| `python-release-check` | Builds the binding wheel for every discovered interpreter plus the pure-Python extra packages, then validates their metadata with `twine check`. |
| `npm-pack-check` | Verifies the published npm tarball would include `wasm/tensogram_wasm_bg.wasm` (wasm-pack writes a `.gitignore` that npm otherwise honours, silently dropping the wasm blob). |

`make release-check` may be run on a working tree that still has the
version-bump/changelog edits uncommitted — the packaging checks pass
`--allow-dirty` for exactly this reason. The *real* publish always runs on the
clean tagged commit.

### 5. Commit and push

Commit the version bump and changelog edits and push. **If anything is
uncommitted, STOP** — a tag must point at a clean tree.

### 6. (Optional but recommended) Dispatch the CI preflight

Run the `release-preflight` workflow (`workflow_dispatch`, with the expected
version as input). It re-runs the gates on a clean checkout and adds the
grib/netcdf and macOS coverage plus real dry-run publishes — things the local
gate cannot fully cover.

### 7. Tag and publish

```bash
git tag X.Y.Z          # NO leading 'v'
git push origin X.Y.Z
```

Then create the GitHub release. The tag push triggers the publish workflows:

- `publish-crates.yml` — crates.io, in dependency order.
- `publish-pypi.yml` — the binding wheel + pure-Python packages.
- `publish-npm.yml` — `@ecmwf.int/tensogram` (with the wasm-blob guard).
- `publish-ffi.yml` — pre-built C/C++ FFI tarballs attached to the release.

## Troubleshooting

- **`cargo package`/`publish` aborts on "uncommitted changes"** — expected on a
  dirty tree; `make release-check` already passes `--allow-dirty`. If you call
  the cargo commands directly, add `--allow-dirty` or commit first.
- **`maturin failed … Object is too small`** — a stale/zeroed binding cdylib
  from a previous `maturin develop`. `make python-build` removes the artifact
  before building to force a clean relink; if you hit it outside make, delete
  `python/bindings/target/release/libtensogram*.so` and rebuild.
- **Fortran configure fails with a non-existent include path** — a stale
  `build/fortran` CMake cache pinned an old prefix. `make build` wipes
  `build/fortran` first; otherwise `rm -rf build/fortran` and re-run.
- **`cargo cinstall: no such subcommand`** — `cargo-c` is not installed (see
  Prerequisites).

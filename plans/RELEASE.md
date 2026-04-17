# Tensogram Release Plan — crates.io + PyPI

> **Status**: DRAFT — pending decisions marked with ⚠️  
> **Target version**: 0.13.0  
> **Scope**: Complete — every publishable artifact ships  

---

## Summary

Tensogram is a Rust workspace of 13 crates plus 3 Python packages. None of them
have ever been published to crates.io or PyPI. This plan gets every publishable
artifact onto both registries in a single coordinated release, and creates the CI
workflows so future releases are push-button.

The work breaks into two parts:

1. **Manifest fixes** (Steps 1–10) — the crates today fail `cargo publish`
   because inter-crate dependencies lack version numbers, most Cargo.toml files
   are missing required metadata (license, description, readme), and there's a
   thiserror version split. These are all concrete edits to existing files.

2. **CI workflows + release command** (Steps 11–15) — four new GitHub Actions
   workflows (crates.io publish, PyPI maturin wheels, PyPI pure-Python extras,
   preflight validation) plus extending the existing `make-release` LLM command
   with publishing steps. The crates.io workflow publishes 10 crates sequentially
   with index-polling. The PyPI workflow builds native wheels for Linux x86_64
   and macOS (both arches) via `maturin-action`. The pure Python packages use
   ECMWF's existing `cd-pypi.yml` reusable workflow.

The release day is a 12-step runbook where every step is a manual gate — no
irreversible action fires automatically. Platforms: Linux x86_64, macOS x86_64,
macOS arm64. No Windows, no Linux aarch64.

Five decisions are needed before work starts: the SPDX licence expression for
the SZ3 crate, whether to ship free-threaded 3.13t wheels, the MSRV, verifying
the PyPI API token is still valid, and how to handle system-library deps in the
publish workflow.

---

## Goal

Publish all tensogram Rust crates on crates.io and all Python packages on PyPI
in a single coordinated release, with CI workflows that make future releases
repeatable.

## What ships

**crates.io** — 10 crates:

| Crate | Publish order | Why it exists |
|-------|:---:|---|
| `tensogram-szip` | 1 | Pure-Rust AEC/SZIP codec, no C deps, useful standalone |
| `tensogram-sz3-sys` | 2 | FFI bindings for SZ3 lossy compressor, vendors C++ source |
| `tensogram-sz3` | 3 | Safe high-level SZ3 API |
| `tensogram-encodings` | 4 | Encoding pipeline — the codec registry all higher crates use |
| `tensogram-core` | 5 | **The primary library.** Encode, decode, file I/O, streaming |
| `tensogram-grib` | 6 | GRIB→tensogram converter (requires system ecCodes at build time) |
| `tensogram-netcdf` | 7 | NetCDF→tensogram converter (requires system libnetcdf at build time) |
| `tensogram-ffi` | 8 | C FFI layer — produces `libtensogram.so`/`.a` + generated header |
| `tensogram-cli` | 9 | The `tensogram` CLI binary (`cargo install tensogram-cli`) |
| `tensogram-wasm` | 10 | WebAssembly bindings via wasm-bindgen (also usable as `rlib`) |

**PyPI** — 3 packages:

| Package | Build system | Why it exists |
|---------|---|---|
| `tensogram` | maturin (PyO3 native extension) | Python bindings — encode/decode/file API with NumPy |
| `tensogram-xarray` | hatchling (pure Python) | `xr.open_dataset("f.tgm", engine="tensogram")` |
| `tensogram-zarr` | hatchling (pure Python) | Zarr v3 store backend |

**Not publishing** (with reason):

| Crate | Why |
|-------|-----|
| `tensogram-benchmarks` | Internal. Already `publish = false` |
| `tensogram-rust-examples` | Internal. Already `publish = false` |

---

## Step 1 — Version bump to 0.13.0

**What**: Bump version from 0.12.0 to 0.13.0 in every manifest and the VERSION
file.

**Why**: The metadata and packaging changes we're about to make are a material
change to the published contract. 0.12.0 has already been tagged internally. A
fresh minor version communicates "this is the first public release" clearly.

**Files**: The existing `/make-release` command already bumps `VERSION` and all
`Cargo.toml`/`pyproject.toml` files (though it is currently missing
`tensogram-sz3` and `tensogram-sz3-sys` — fixed in Step 15). After Step 3, the
version string also appears in inter-crate dependency pins (`version = "=X.Y.Z"`
in path deps), which the updated `make-release` command must cover too.

---

## Step 2 — Guard non-public crates

**What**: Add `publish = false` to `[package]` in:

```
python/bindings/Cargo.toml
```

**Why**: The Python bindings crate (`tensogram-python`) is built via maturin and
published as a PyPI wheel — it must never go to crates.io. `benchmarks` and
`examples/rust` already have `publish = false`.

---

## Step 3 — Add version numbers to all inter-crate path dependencies

**What**: Every internal dependency that uses `path = "..."` must also specify
`version = "=0.13.0"`.

**Why**: `cargo publish` strips the `path` and resolves from the registry. Without
a version, the publish is rejected outright. Using exact pin (`=0.13.0`) because
in the 0.x semver range even minor bumps can break — this ensures consumers get
exactly the version that was tested together.

**Ongoing impact**: Once these version pins exist, every future version bump
must update them too — not just the `[package] version` fields. The
`make-release` command (Step 15) is updated to cover this.

**Root `Cargo.toml`** — workspace dependency table:

```toml
tensogram-szip = { path = "rust/tensogram-szip", version = "=0.13.0" }
tensogram-sz3 = { path = "rust/tensogram-sz3", version = "=0.13.0" }
tensogram-sz3-sys = { path = "rust/tensogram-sz3-sys", version = "=0.13.0" }
```

**`rust/tensogram-sz3/Cargo.toml`**:

```toml
tensogram-sz3-sys = { path = "../tensogram-sz3-sys", version = "=0.13.0" }
```

**`rust/tensogram-core/Cargo.toml`**:

```toml
tensogram-encodings = { path = "../tensogram-encodings", version = "=0.13.0", default-features = false }
```

**`rust/tensogram-grib/Cargo.toml`**:

```toml
tensogram-core = { path = "../tensogram-core", version = "=0.13.0" }
tensogram-encodings = { path = "../tensogram-encodings", version = "=0.13.0" }
```

**`rust/tensogram-netcdf/Cargo.toml`**:

```toml
tensogram-core = { path = "../tensogram-core", version = "=0.13.0" }
tensogram-encodings = { path = "../tensogram-encodings", version = "=0.13.0" }
```

**`rust/tensogram-cli/Cargo.toml`**:

```toml
tensogram-core = { path = "../tensogram-core", version = "=0.13.0" }
tensogram-grib = { path = "../tensogram-grib", version = "=0.13.0", optional = true }
tensogram-netcdf = { path = "../tensogram-netcdf", version = "=0.13.0", optional = true }
```

**`rust/tensogram-ffi/Cargo.toml`**:

```toml
tensogram-core = { path = "../tensogram-core", version = "=0.13.0" }
tensogram-encodings = { path = "../tensogram-encodings", version = "=0.13.0" }
```

**`rust/tensogram-wasm/Cargo.toml`**:

```toml
tensogram-core = { path = "../tensogram-core", version = "=0.13.0", default-features = false, features = [...] }
```

---

## Step 4 — Fix thiserror version split in tensogram-sz3

**What**: Change `thiserror = "1"` to `thiserror.workspace = true` in
`rust/tensogram-sz3/Cargo.toml` and update the crate's source for thiserror v2.

**Why**: The workspace uses thiserror 2 everywhere else. Having v1 in sz3 pulls
**both** v1 and v2 into the dependency tree of every consumer. That's ~30KB of
unnecessary compilation and two versions of the same thing in `Cargo.lock`.
thiserror v2 is nearly API-identical — migration is usually just re-running
`cargo check` and fixing any compile errors.

---

## Step 5 — Resolve licence expression for sz3 crates

**What**: Fix the licence metadata for `tensogram-sz3-sys` and `tensogram-sz3`.

**Why**: Two problems exist today:

1. Both crates claim `Apache-2.0 OR MIT`, but the repository has **no MIT
   licence text**. The top-level `LICENSE` is Apache-2.0 only. Claiming MIT
   without the grant text is legally meaningless.

2. `tensogram-sz3-sys` vendors the SZ3 header-only library. That tree contains
   **two** third-party licences:
   - `SZ3/copyright-and-BSD-license.txt` — a 4-clause BSD-like licence from
     UChicago Argonne (covers the SZ3 core)
   - `SZ3/include/SZ3/utils/ska_hash/LICENSE.txt` — **Boost Software License
     1.0** (covers the ska flat hash map)

   The SPDX `license` field must reflect what actually ships in the crate
   tarball. The vendored tree also contains `SZ3/tools/pysz/` (a whole Python
   package with its own LICENSE) — that directory must be excluded from the
   tarball (Step 7 handles this).

⚠️ **Decision needed** — pick one:

- **Option A** (recommended): Standardise our code to `Apache-2.0` (matching
  the rest of the project) and use `license-file` on `tensogram-sz3-sys` to
  point to a composite `LICENSES.md` that includes our Apache-2.0 grant plus the
  two vendored licence texts. Drop the `OR MIT` claim entirely since there is no
  MIT text.

  Concrete steps for this option:
  1. Create `rust/tensogram-sz3-sys/LICENSES.md` containing the full text of:
     Apache-2.0 (our code), Argonne BSD (SZ3 core), Boost-1.0 (ska_hash).
  2. Set `license-file = "LICENSES.md"` in `rust/tensogram-sz3-sys/Cargo.toml`
     (remove the `license` field — `license` and `license-file` are mutually
     exclusive).
  3. Add `"LICENSES.md"` to the Step 7 include list (replacing `"LICENSE"`).
  4. `tensogram-sz3` (which contains no vendored code) gets
     `license = "Apache-2.0"` (remove `OR MIT`).

- **Option B**: Add an MIT licence file to the repo and keep dual-licence on our
  wrapper code. Use `license = "(Apache-2.0 OR MIT) AND BSL-1.0"` on sz3-sys
  (since the Argonne licence is close to BSD but not standard SPDX, a
  `license-file` is more accurate).

Either way: ensure both `SZ3/copyright-and-BSD-license.txt` and
`SZ3/include/SZ3/utils/ska_hash/LICENSE.txt` are included in the crate tarball
(Step 7 covers this), and the `THIRD_PARTY_LICENSES` file is updated to cover
vendored-in-tarball source explicitly.

---

## Step 6 — Add package metadata to all 10 publishable crates

**What**: Add the full `[package]` metadata block to every publishable crate.

**Why**: crates.io requires `license` (or `license-file`) and `description`.
Without `repository`, `homepage`, `documentation`, the crate page has no links.
Without `readme`, the page is blank. Without `keywords`/`categories`, the crate
is unfindable. This is the difference between a professional release and a
placeholder.

Template (adapt `name`, `description`, `keywords`, `categories` per crate):

```toml
[package]
name = "tensogram-core"
version = "0.13.0"
edition = "2024"
license = "Apache-2.0"
description = "Fast binary N-tensor message format for scientific data — encode, decode, stream"
repository = "https://github.com/ecmwf/tensogram"
homepage = "https://github.com/ecmwf/tensogram"
documentation = "https://docs.rs/tensogram-core"
readme = "README.md"
keywords = ["tensogram", "tensor", "scientific-data", "encoding", "serialization"]
categories = ["science", "encoding"]
authors = ["ECMWF <software-support@ecmwf.int>"]
rust-version = "TBD"   # see Step 8
```

Suggested per-crate descriptions:

| Crate | description |
|-------|-------------|
| `tensogram-szip` | (keep existing) "Pure-Rust CCSDS 121.0-B-3 Adaptive Entropy Coding (AEC/SZIP) — encode, decode, and range decode" |
| `tensogram-sz3-sys` | (keep existing) "Clean-room FFI bindings for the SZ3 lossy compression library" |
| `tensogram-sz3` | (keep existing) "High-level SZ3 compression API for Tensogram" |
| `tensogram-encodings` | "Encoding pipeline and compression codec registry for the Tensogram message format" |
| `tensogram-core` | "Fast binary N-tensor message format for scientific data — encode, decode, file I/O, streaming" |
| `tensogram-grib` | "GRIB to Tensogram format converter using ecCodes" |
| `tensogram-netcdf` | "NetCDF to Tensogram format converter" |
| `tensogram-ffi` | "C FFI bindings for the Tensogram N-tensor message format library" |
| `tensogram-cli` | "CLI for inspecting, converting, and manipulating Tensogram .tgm files" |
| `tensogram-wasm` | (keep existing) "WebAssembly bindings for the Tensogram N-tensor message format" |

---

## Step 7 — Create per-crate README files and set include lists

**What**: Create a short `README.md` inside each of the 10 publishable crate
directories. Add an explicit `[package] include` list where needed.

**Why (README)**: crates.io renders the crate's README as its landing page. No
README = blank page. Each README should be 20–40 lines: what the crate does, a
minimal usage example, and a link to the full repository docs. Do **not** use
symlinks to the root README — they break on Windows and `cargo package` may not
follow them.

**Why (include)**: Only needed where the default `cargo package` would ship
unwanted files. Most crates are clean (`src/` + `Cargo.toml`) and can use Cargo
defaults safely. Add explicit include lists only for crates with vendored source,
build scripts, or extra artefacts.

For `tensogram-sz3-sys` (vendored C++ source + build script). The include list
must be precise — the full `SZ3/` tree contains tools, a Python package, and
test data that are not needed and add licence surface. The `build.rs` only reads
`SZ3/CMakeLists.txt` (version parsing), `SZ3/include/SZ3/version.hpp.in`
(template), and the `SZ3/include/` headers:

```toml
include = [
    "src/**",
    "build.rs",
    "cpp/sz3_ffi.cpp",
    "SZ3/CMakeLists.txt",
    "SZ3/include/**",
    "SZ3/copyright-and-BSD-license.txt",
    "Cargo.toml",
    "README.md",
    "LICENSES.md",       # composite license-file (Step 5 Option A)
]
```

Note: if using `license-file = "LICENSES.md"`, Cargo auto-includes it even
without the explicit entry, but listing it makes the intent obvious. If
Option B is chosen instead, replace `LICENSES.md` with `LICENSE`.

This excludes `SZ3/tools/` (H5Z-SZ3, pysz, mdz, paraview — none used by the
build), `SZ3/SZ3Config.cmake.in`, and `SZ3/README.md`. The `ska_hash/LICENSE.txt`
(Boost-1.0) is included via the `SZ3/include/**` glob — that's intentional since
the headers reference it.

For `tensogram-ffi` (has build.rs + cbindgen):

```toml
include = ["src/**", "build.rs", "cbindgen.toml", "Cargo.toml", "README.md", "LICENSE"]
```

After adding, verify with `cargo package --list -p <crate>` for each.

---

## Step 8 — Determine and set rust-version (MSRV)

**What**: Find the minimum supported Rust version and set `rust-version` in all
10 publishable crates.

**Why**: Cargo uses this field to give users a clear error ("your Rust is too
old") instead of cryptic compilation failures. It's also displayed on crates.io
and used by CI tools like `cargo-msrv`.

**How**: Run `cargo +1.85.0 check --workspace` (or use `cargo-msrv find`).
Given edition 2024 and deps like pyo3 0.28 / object_store 0.13, expect MSRV
around **1.85+**.

---

## Step 9 — Complete Python package metadata

**What**: Add missing metadata to all three `pyproject.toml` files.

**Why**: PyPI renders the metadata on the package page. Without `description`,
`authors`, `classifiers`, and project URLs, the package looks abandoned or
untrustworthy. Classifiers also control how PyPI indexes and filters the package.

**`python/bindings/pyproject.toml`** — add to `[project]`:

```toml
description = "Fast binary N-tensor message format for scientific data"
license = "Apache-2.0"
authors = [{name = "ECMWF", email = "software-support@ecmwf.int"}]
readme = "README.md"
classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]

[project.urls]
Homepage = "https://github.com/ecmwf/tensogram"
Repository = "https://github.com/ecmwf/tensogram"
Documentation = "https://github.com/ecmwf/tensogram/tree/main/docs"
Changelog = "https://github.com/ecmwf/tensogram/blob/main/CHANGELOG.md"
```

Also create `python/bindings/README.md` — PyPI uses this as the long description.

**`python/tensogram-xarray/pyproject.toml`** and
**`python/tensogram-zarr/pyproject.toml`** — add `authors`, `classifiers`,
`[project.urls]` (same pattern), and pin the tensogram dependency:

```toml
dependencies = [
    "tensogram>=0.13.0,<0.14",
    # ... rest unchanged
]
```

**Why pin**: These packages are tightly coupled to the native `tensogram`
extension. A semver-minor bump in 0.x can break the internal API. Pinning to
`>=0.13.0,<0.14` ensures users don't end up with an incompatible combination.

---

## Step 10 — Add docs.rs build hints

**What**: Add `[package.metadata.docs.rs]` to crates that need it and verify
the `documentation` URL is correct for every published crate.

**Why**: docs.rs auto-builds documentation for every crate published to
crates.io. If the build fails (missing system libraries, C++ compilation
timeout), the documentation page is blank and the `documentation` link in the
crate metadata is broken.

Crate-by-crate strategy:

| Crate | docs.rs risk | Action |
|-------|-------------|--------|
| `tensogram-szip` | None (pure Rust) | Default is fine |
| `tensogram-sz3-sys` | C++17 compilation | `all-features = true`, may need extended timeout |
| `tensogram-sz3` | Transitive C++ | Default is fine |
| `tensogram-encodings` | Compiles 6 native codecs | `all-features = true` |
| `tensogram-core` | Transitive native codecs | `all-features = true` |
| `tensogram-grib` | **Will fail** — needs system eccodes | Set `documentation` to repo docs instead of docs.rs |
| `tensogram-netcdf` | **Will fail** — needs system libnetcdf | Set `documentation` to repo docs instead of docs.rs |
| `tensogram-ffi` | Needs cbindgen | May work, monitor |
| `tensogram-cli` | Binary crate, optional grib/netcdf | `default-features = true` (skip grib/netcdf features) |
| `tensogram-wasm` | **Will fail** — wasm-only deps (`js-sys`, `getrandom` with `js`) don't compile on x86_64 | Set `default-target = "wasm32-unknown-unknown"` in `[package.metadata.docs.rs]` |

For crates where docs.rs will fail, set `documentation` to the project
documentation site instead of `https://docs.rs/...`:

```toml
documentation = "https://github.com/ecmwf/tensogram/tree/main/docs"
```

---

## Step 11 — Create the crates.io publish workflow

**What**: Create `.github/workflows/publish-crates.yml` — a single workflow that
publishes all 10 crates in dependency order with index polling between each.

**Why (single workflow, not 10 separate ones)**: The crates have a strict
dependency chain. Publishing them as separate workflows would require manual
coordination. A single workflow with sequential steps and polling ensures each
crate is available on the crates.io index before its dependents try to publish.

**Why (inline, not reusable workflow)**: The ECMWF `publish-rust-crate.yml`
reusable workflow handles one crate at a time. Calling it 10 times via
`workflow_call` jobs doesn't allow inserting index-polling steps between calls.
Inlining the logic (which is simple: checkout, toolchain, `cargo publish`, poll)
gives full control over sequencing. The reusable workflow is only 167 lines and
easy to replicate.

**Why (workflow_dispatch, not on-tag)**: Publishing to crates.io is irreversible.
We want a manual trigger with a protected environment requiring approval, not an
automatic trigger that fires the moment someone pushes a tag.

**Design**:

```
trigger: workflow_dispatch
environment: "crates-io" (protected, requires reviewer approval)
runner: ubuntu-24.04

for each crate in publish order:
  1. check if version already exists on crates.io (skip if yes)
  2. cargo publish -p <crate> --locked
  3. poll crates.io API until version appears in index (10s intervals, 5min timeout)
  4. proceed to next crate
```

The "skip if already published" check makes the workflow **idempotent** — safe to
re-run after partial failures without republishing already-succeeded crates.
Only allow re-runs from the same tag/SHA — any code change requires a new
version.

Publish order (matches dependency graph):

```
 1. tensogram-szip          (leaf — no internal deps)
 2. tensogram-sz3-sys        (leaf — no internal deps)
 3. tensogram-sz3            (depends on 2)
 4. tensogram-encodings      (depends on 1, 3 via features)
 5. tensogram-core           (depends on 4)
 6. tensogram-grib           (depends on 5, 4)    ← --manifest-path, needs eccodes
 7. tensogram-netcdf         (depends on 5, 4)    ← --manifest-path, needs libnetcdf
 8. tensogram-ffi            (depends on 5, 4)
 9. tensogram-cli            (depends on 5, 6 optional, 7 optional)
10. tensogram-wasm           (depends on 5)        ← --manifest-path, wasm target
```

**Excluded crates**: `tensogram-grib`, `tensogram-netcdf`, and `tensogram-wasm`
are in the workspace `exclude` list. They cannot be published with
`cargo publish -p ...` — they need
`cargo publish --manifest-path rust/<crate>/Cargo.toml`.

**System dependencies**: The `cargo publish` verification step compiles the
crate. `tensogram-grib` needs `eccodes` (C library), `tensogram-netcdf` needs
`libnetcdf`. The publish workflow runner must have these installed, or pass
`--no-verify` for these two crates (acceptable since CI already tests them on
macOS). Using `--no-verify` skips the compile check but still validates the
tarball. Preferred: install the libs via apt on the ubuntu runner.

**Wasm target**: `tensogram-wasm` depends on `js-sys`, `getrandom` with `js`
feature, etc. — these only compile for `wasm32-unknown-unknown`. Use
`--no-verify` for this crate (CI tests it separately with `wasm-pack test`),
or `--target wasm32-unknown-unknown` if the runner has the wasm target installed.

---

## Step 12 — Create the PyPI publish workflow for tensogram

**What**: Create `.github/workflows/publish-pypi-tensogram.yml` — builds native
wheels and uploads to PyPI.

**Why (custom, not reusable workflow)**: None of the ECMWF reusable workflows
support maturin. `cd-pypi.yml` uses `python -m build` (pure Python only).
`cd-pypi-binwheel.yml` uses cibuildwheel (not maturin). We need
`PyO3/maturin-action`.

**Why (wheels only, no sdist)**: The `python/bindings/Cargo.toml` references
`../../rust/tensogram-core` via relative path. An sdist would not include the
Rust workspace source tree, so `pip install --no-binary` would fail. Until the
layout is restructured to bundle Rust source in the sdist, ship wheels only.

**Design**:

```
trigger: workflow_dispatch
environment: "pypi" (protected)

build matrix:
  - os: ubuntu-latest,  target: x86_64     (Linux)
  - os: macos-13,       target: x86_64     (Intel Mac)
  - os: macos-14,       target: aarch64    (Apple Silicon)

python versions: 3.9, 3.10, 3.11, 3.12, 3.13

steps per matrix entry:
  1. checkout full repo
  2. PyO3/maturin-action — build wheel in python/bindings/
     use `maturin build --release` (NOT `maturin publish` — that would also
     attempt an sdist)
  3. upload wheel as artifact

final job (after all matrix entries):
  1. download all wheel artifacts
  2. twine check dist/*
  3. twine upload dist/* (PYPI_API_TOKEN)
```

⚠️ **Decision: free-threaded Python 3.13t wheels?** The CI already tests 3.13t
on Linux and the README advertises GIL-free support. If we publish that claim
without shipping cp313t wheels, users on free-threaded Python can't install the
package. Options:
- Add cp313t to the matrix for linux x86_64 (where CI already proves it works)
- Or remove the free-threaded claim from the README until wheels ship

**TestPyPI rehearsal**: Before the real publish, run the same workflow targeting
TestPyPI (`PYPI_TEST_API_TOKEN`) and verify:
```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            tensogram==0.13.0
python -c "import tensogram; print('ok')"
```
The `--extra-index-url` is required because tensogram depends on NumPy, which
is on the real PyPI, not TestPyPI. Without it, pip can't resolve the dependency.
This catches metadata and wheel-tag issues before the irreversible real upload.

---

## Step 13 — Create the PyPI publish workflow for xarray + zarr

**What**: Create `.github/workflows/publish-pypi-extras.yml` using the ECMWF
`cd-pypi.yml` reusable workflow.

**Why (reusable workflow works here)**: Both packages are pure Python with
hatchling. The `cd-pypi.yml` workflow does exactly what's needed: `python -m
build` → `twine check` → `twine upload`.

**Design**:

```yaml
jobs:
  xarray:
    uses: ecmwf/reusable-workflows/.github/workflows/cd-pypi.yml@main
    with:
      working-directory: python/tensogram-xarray/
    secrets: inherit

  zarr:
    uses: ecmwf/reusable-workflows/.github/workflows/cd-pypi.yml@main
    with:
      working-directory: python/tensogram-zarr/
    secrets: inherit
```

**Tag requirement**: `cd-pypi.yml` gates on `github.ref_type == 'tag'` — it
refuses the real PyPI upload if dispatched from a branch. The workflow **must
be dispatched from the release tag** (e.g. `0.13.0`), not from `main`. For
TestPyPI validation during development, use the `testpypi: true` input to
bypass the tag gate.

**Important**: These packages `pip install tensogram` as a dependency during
their own install. If PyPI hasn't fully propagated the `tensogram` wheels yet,
the `pip install` (and therefore the user install of xarray/zarr) will fail.
The release runbook (step 9) includes a propagation wait before triggering this
workflow.

---

## Step 14 — Create the preflight validation workflow

**What**: Create `.github/workflows/release-preflight.yml` — validates
everything is correct before any irreversible publish.

**Why**: crates.io and PyPI publishes cannot be undone. A preflight workflow
catches metadata errors, broken tarballs, and version mismatches *before* we
commit to the registry. It also serves as documentation of what "ready to
release" means.

**Design**:

```
trigger: workflow_dispatch (input: expected version string)
runner: same environment as publish workflow (needs eccodes + libnetcdf)

steps:
  1. checkout
  2. assert VERSION file == input version == every first-party Cargo.toml and
     pyproject.toml version. IMPORTANT: exclude vendored third-party files —
     `rust/tensogram-sz3-sys/SZ3/tools/pysz/pyproject.toml` has version "1.0.3"
     and must NOT be checked or bumped. Scope to the explicit file list from
     Step 1 / make-release.
  3. cargo fmt --check
  4. cargo clippy --workspace --all-targets -- -D warnings
  5. cargo test --workspace
  6. cargo test --manifest-path rust/tensogram-grib/Cargo.toml   (excluded from workspace)
  7. cargo test --manifest-path rust/tensogram-netcdf/Cargo.toml  (excluded from workspace)
  8. for each of the 10 publishable crates:
       cargo package --list (verify tarball contents — no surprises)
       cargo publish --dry-run  (full verification including dependency resolution)
     in publish order, so that each depends on the previous being valid.
     Use --manifest-path for grib, netcdf, and wasm.
     Use --no-verify for wasm (wasm32 target, won't compile on x86_64 runner).
  9. maturin build --release in python/bindings/ (verify Python wheel builds)
 10. python -m build for xarray and zarr
 11. twine check on all Python dist/ artifacts
```

**First-release caveat**: `cargo publish --dry-run` for non-leaf crates (core,
encodings, etc.) **will fail** on the first release because their internal
dependencies don't exist on crates.io yet. This is expected, not a bug. For the
first release, use `cargo package --no-verify` for non-leaf crates instead —
it validates the tarball contents without attempting registry dependency
resolution. On subsequent releases (where all deps are already on crates.io),
the full `--dry-run` will work for all 10 crates. The preflight's primary job
is catching metadata errors and tarball surprises, not compiling from registry.

---

## Step 15 — Extend make-release command

**What**: Update **both** `.claude/commands/make-release.md` and
`.opencode/commands/make-release.md` (they are identical copies — both must stay
in sync, or one must be deleted and replaced with a symlink/include).

**Why**: The project already has a working LLM-driven release command that
handles pre-release checks, version bumping, changelog, commit, tag, push, and
GitHub release creation. We extend it rather than creating a parallel script.

Changes needed to both `make-release.md` files:

1. **Fix missing crates in the version-bump list.** The current command lists
   11 crates but is **missing `tensogram-sz3` and `tensogram-sz3-sys`**. Add
   them.

2. **Add inter-crate dependency version bumping.** After Step 3 adds
   `version = "=X.Y.Z"` to path dependencies, every future version bump must
   also update those pins. The current stale-version grep catches `[package]
   version` fields, but the command must explicitly note that dependency version
   pins need updating too. The locations are:
   - Root `Cargo.toml` workspace deps (szip, sz3, sz3-sys entries)
   - `rust/tensogram-sz3/Cargo.toml` dep on sz3-sys
   - `rust/tensogram-core/Cargo.toml` dep on encodings
   - `rust/tensogram-grib/Cargo.toml` deps on core + encodings
   - `rust/tensogram-netcdf/Cargo.toml` deps on core + encodings
   - `rust/tensogram-cli/Cargo.toml` deps on core + grib + netcdf
   - `rust/tensogram-ffi/Cargo.toml` deps on core + encodings
   - `rust/tensogram-wasm/Cargo.toml` dep on core

3. **Scope the stale-version grep to first-party files only.** The vendored
   `rust/tensogram-sz3-sys/SZ3/tools/pysz/pyproject.toml` contains
   `version = "1.0.3"` — a naive `grep -r 'version = "OLD"' --include='pyproject.toml'`
   will match it and either fail the check or (worse) mutate vendored source.
   Restrict the grep to the three first-party pyproject.toml paths explicitly:
   ```bash
   grep 'version = "OLD"' \
     python/bindings/pyproject.toml \
     python/tensogram-xarray/pyproject.toml \
     python/tensogram-zarr/pyproject.toml
   ```

4. **Reorder: publishing goes BEFORE tag and GitHub release.** The current
   command flow is: version bump → commit → push → tag → GitHub release. The
   new flow must be: version bump → commit → push → preflight → publish →
   smoke tests → tag → GitHub release. This matches the runbook. Do NOT place
   the publish steps after "Create GitHub Release" — that would announce a
   release before artifacts exist.

5. **Fix Python pre-release checks.** The current command runs xarray/zarr
   `pytest` without first installing the packages into the venv. Add:
   ```bash
   uv pip install -e "python/tensogram-xarray/[dask]"
   uv pip install -e python/tensogram-zarr/
   ```
   before the pytest lines.

6. **Add excluded-crate tests** to the pre-release checks section:
   ```bash
   cargo test --manifest-path rust/tensogram-grib/Cargo.toml
   cargo test --manifest-path rust/tensogram-netcdf/Cargo.toml
   ```

7. **Add a "Publish to Registries" section** (after commit+push, before tag):
   - Run `release-preflight.yml` (workflow_dispatch) — must pass
   - Run `publish-crates.yml` (workflow_dispatch) — 10 crates sequentially
   - Smoke-test Rust: `cargo add tensogram-core@X.Y.Z` in a fresh project
   - Run `publish-pypi-tensogram.yml` (workflow_dispatch) — native wheels
   - Wait for PyPI propagation, then smoke-test Python
   - Run `publish-pypi-extras.yml` (dispatch from the **release tag**, since
     `cd-pypi.yml` is tag-gated) — xarray + zarr
   - Wait for PyPI propagation, then smoke-test extras

---

## Release day runbook

Once all code changes from Steps 1–10 are merged to `main`, run the release.
The first 2 steps below are already handled by `/make-release 0.13.0`. The
rest are the new additions.

```
 ── make-release handles these ──────────────────────────────────────────
 1. Pre-release checks                       (fmt, clippy, test, python, docs)
 2. Version bump + changelog + commit + push  (all manifests + dep version pins)
 ── new: preflight + publish ────────────────────────────────────────────
 3. Run release-preflight.yml on main        → must be all green
 4. git tag 0.13.0 on the preflight commit   → git push --tags
 5. Run publish-crates.yml (dispatch on tag) → 10 crates published sequentially
 6. Smoke test Rust (from a fresh dir, not the repo):
      cd $(mktemp -d) && cargo init smoke && cd smoke
      cargo add tensogram-core@0.13.0
      cargo build                            → must compile
 7. Run publish-pypi-tensogram.yml           → wheels on PyPI
 8. Wait for PyPI to serve tensogram==0.13.0 (poll: pip download tensogram==0.13.0)
 9. Smoke test Python (from a fresh venv):
      python -m venv /tmp/tgm && source /tmp/tgm/bin/activate
      pip install tensogram==0.13.0
      python -c "import tensogram; print('ok')"
10. Run publish-pypi-extras.yml (dispatch from the release tag) → xarray + zarr
11. Wait + smoke test extras (same fresh venv):
      pip install tensogram-xarray==0.13.0 tensogram-zarr==0.13.0
      python -c "import tensogram_xarray; import tensogram_zarr; print('ok')"
12. Create GitHub Release (0.13.0)           → public announcement
```

Note: `make-release` currently creates the GitHub release (step 12) and the
tag (step 4) as part of its flow. For the first public release, we insert the
preflight and publishing steps between the commit/push and the tag/release
creation. After the first release, the updated `make-release` command handles
the full flow including publishing.

Every numbered step is a manual gate. Proceed only after the previous step
succeeds. If any step fails, stop and fix — don't push through.

**Why tag AFTER preflight**: If preflight fails, a public tag would point at
unreleased or broken state. By running preflight first, we know the commit is
good before stamping it with a version tag.

**Why smoke tests from outside the repo**: Testing from a checkout doesn't
prove the published artifact works. A fresh `cargo init` or `python -m venv`
with only registry dependencies is the real consumer experience.

**If something fails after partial publish**: crates.io publishes are
irreversible. If crate 5 of 10 fails due to a transient error (network, index
lag), re-run the same workflow at the same version — the skip-if-already-published
check will skip the first 4. If the failure is a real bug requiring a code fix,
bump to 0.13.1 and re-publish all 10 crates at the new version (the skip check
only helps for same-version retries, not cross-version).

---

## Decisions needed before starting

| # | Question | Recommendation | Impact if deferred |
|---|----------|----------------|--------------------|
| 1 | Licence for `tensogram-sz3-sys` — the vendored SZ3 tree has Argonne BSD-like + Boost-1.0 licences, and repo has no MIT text | Drop the `OR MIT` claim, use `license = "Apache-2.0"` for our code, use `license-file` for sz3-sys composite | Blocks publish of sz3-sys, sz3, and all downstream |
| 2 | Free-threaded 3.13t wheels — CI tests it, README claims it | Either ship cp313t wheels or remove the claim | Credibility issue if claim is unsubstantiated |
| 3 | MSRV (`rust-version`) | Set to the oldest toolchain the release workflow actually uses | Non-blocking but looks unprofessional without it |
| 4 | `PYPI_API_TOKEN` — verify it's still valid and scoped (3 years old) | Test with `twine upload --repository testpypi` | Blocks all PyPI uploads |
| 5 | System libs in publish workflow — grib needs eccodes, netcdf needs libnetcdf | Either install via apt or use `--no-verify` for those 2 crates | Blocks publish of grib, netcdf, cli |

---

## Files created or modified (complete list)

**New files** (16):

```
.github/workflows/release-preflight.yml
.github/workflows/publish-crates.yml
.github/workflows/publish-pypi-tensogram.yml
.github/workflows/publish-pypi-extras.yml
rust/tensogram-core/README.md
rust/tensogram-encodings/README.md
rust/tensogram-szip/README.md
rust/tensogram-sz3/README.md
rust/tensogram-sz3-sys/README.md
rust/tensogram-sz3-sys/LICENSES.md           (composite license-file, if Option A)
rust/tensogram-grib/README.md
rust/tensogram-netcdf/README.md
rust/tensogram-ffi/README.md
rust/tensogram-cli/README.md
rust/tensogram-wasm/README.md
python/bindings/README.md
```

**Modified files** (18):

```
VERSION                                     → 0.13.0
Cargo.toml (root)                           → versioned workspace deps
rust/tensogram-core/Cargo.toml              → full metadata + versioned dep
rust/tensogram-encodings/Cargo.toml         → full metadata + description
rust/tensogram-szip/Cargo.toml              → full metadata
rust/tensogram-sz3/Cargo.toml               → fix thiserror, licence, metadata
rust/tensogram-sz3-sys/Cargo.toml           → licence + license-file, metadata, include list
rust/tensogram-grib/Cargo.toml              → full metadata + versioned deps
rust/tensogram-netcdf/Cargo.toml            → full metadata + versioned deps
rust/tensogram-cli/Cargo.toml               → full metadata + versioned deps
rust/tensogram-ffi/Cargo.toml               → full metadata + versioned deps
rust/tensogram-wasm/Cargo.toml              → full metadata + versioned dep + docs.rs target
python/bindings/Cargo.toml                  → publish = false + version bump
python/bindings/pyproject.toml              → full metadata
python/tensogram-xarray/pyproject.toml      → metadata + pin tensogram dep
python/tensogram-zarr/pyproject.toml        → metadata + pin tensogram dep
.claude/commands/make-release.md            → add sz3 crates, dep version bumping, publish steps
.opencode/commands/make-release.md          → keep in sync with .claude copy
```

Plus version bumps in all other crates (benchmarks, examples) to keep
everything in sync.

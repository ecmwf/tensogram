# Contributing to Tensogram

Thank you for your interest in contributing. This guide will get you from zero to a passing test suite in a few minutes.

## Prerequisites

- **Rust 1.87+** (`rustup install stable`)
- **C compiler** (for libaec, zfp, blosc2 FFI dependencies)
- Optional: Python 3.9+ and `uv` for Python bindings (`pip install uv` or https://docs.astral.sh/uv/getting-started/installation/)
- Optional: Node ≥ 20 and `wasm-pack` (`cargo install wasm-pack`) for the TypeScript wrapper
- Optional: mdbook for documentation (`cargo install mdbook`)

## Quick Setup

```bash
git clone https://github.com/ecmwf/tensogram.git
cd tensogram

# Rust-only quick check (needs no optional toolchains)
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The **canonical gate** — what CI runs and what you must pass before committing —
is the top-level Makefile, which covers every language (Rust, Python,
TypeScript, C++, WASM, cargo-c, Fortran):

```bash
make all          # build + test + lint, across all languages
make help         # list every target
```

`make all` needs the optional toolchains listed under Prerequisites (uv, Node +
`wasm-pack`, a C/C++ compiler + CMake, gfortran + pkg-config, cargo-c). When
preparing a release, also run `make release-check` (see
[docs/src/dev/releasing.md](docs/src/dev/releasing.md)).

## Project Structure

```
tensogram/
├── rust/
│   ├── tensogram/         # Core library: wire format, encode/decode,
│   │                           #   file API, iterators, validation, remote
│   ├── tensogram-encodings/    # Encoding pipeline: simple_packing, shuffle,
│   │                           #   compression (szip, zstd, lz4, blosc2, zfp, sz3)
│   ├── tensogram-cli/          # CLI tool: info, ls, dump, get, set, copy,
│   │                           #   merge, split, reshuffle, validate,
│   │                           #   convert-grib (opt), convert-netcdf (opt)
│   ├── tensogram-ffi/          # C FFI layer (generates tensogram.h via cbindgen)
│   ├── tensogram-szip/         # Pure-Rust CCSDS szip codec (for WebAssembly)
│   ├── tensogram-sz3/          # SZ3 Rust API over the clean-room shim
│   ├── tensogram-sz3-sys/      # Clean-room C++ FFI shim for SZ3
│   ├── tensogram-grib/         # GRIB→Tensogram importer (ecCodes; opt-in)
│   ├── tensogram-netcdf/       # NetCDF→Tensogram importer (libnetcdf; opt-in)
│   ├── tensogram-wasm/         # WebAssembly bindings (wasm-pack; opt-in)
│   └── benchmarks/             # Benchmark suite
├── python/
│   ├── bindings/               # Python bindings (PyO3, excluded from default build)
│   ├── tensogram-xarray/       # xarray backend engine
│   ├── tensogram-zarr/         # Zarr v3 store backend
│   ├── tensogram-anemoi/       # anemoi-inference output plugin
│   ├── tensogram-earthkit/     # earthkit-data source + encoder plugin
│   └── tests/                  # Python test suite
├── cpp/
│   ├── include/                # C++ wrapper header + C header (incl. async/)
│   ├── tests/                  # C++ GoogleTest suite
│   └── CMakeLists.txt          # CMake build system
├── fortran/                    # Fortran 2008 binding over the C FFI (CMake + fpm)
├── typescript/                 # TypeScript wrapper over the WASM crate (npm pkg)
├── tensoscope/                 # Browser-based interactive .tgm viewer (React + WASM)
├── examples/
│   ├── rust/                   # Runnable Rust examples (NN_description.rs)
│   ├── cpp/                    # C++ examples using the wrapper
│   ├── python/                 # Python examples using the PyO3 bindings
│   ├── fortran/                # Fortran examples using the binding
│   ├── typescript/             # TypeScript examples using the WASM wrapper
│   └── jupyter/                # Narrative Jupyter notebook walk-throughs
├── docs/                       # mdbook documentation source
├── plans/                      # Design docs, implementation status, TODOs
│   └── ARCHITECTURE.md         # How the crates fit together
├── CHANGELOG.md                # Release history
└── VERSION                     # Current version
```

## Development Workflow

> **Agents:** the full lifecycle (onboard → plan → branch → develop → PR →
> review → release) and the bundled `/`-skill that drives each phase are
> documented in [AGENTS.md](AGENTS.md) under "Development workflow". The bundled
> skills live in `.prompts/` (e.g. `/onboard`, `/prepare-make-pr`,
> `/make-release`). The steps below are the human-facing version.

### Branch naming

Branch off `main`. Names follow `<type>/<kebab-summary>`, where `<type>` is one
of the conventional-commit types also used for commit messages:

```
feat/<summary>      fix/<summary>       docs/<summary>
chore/<summary>     refactor/<summary>  test/<summary>
ci/<summary>        perf/<summary>      build/<summary>
```

Audit / longer-lived work also uses `security/<summary>` and
`hardening/<summary>`. Examples from history: `feat/fortran-interface`,
`fix/python-packages-build-system`, `docs/reorganize-documentation`,
`ci/manylinux-2-28`, `security/hardening-audit`. (`feature/…` appears in older
history; new branches use the short `feat/` form to match the commit types.)

### 1. Make your changes

Work on a branch (see Branch naming above). The codebase follows these conventions:

- **No panics in library code.** All fallible operations return `Result<T, TensogramError>`.
- **Immutability by default.** Use `let` unless mutation is required.
- **Short functions.** 10-50 lines typical, extract helpers when a function grows past that.
- **CBOR determinism.** All metadata encoding goes through the canonicalization step.

See the [Code Style](#code-style) section below for the full style guide.

### 2. Run the gate

Before submitting, run the full multi-language gate:

```bash
make all          # = make check test lint, across all languages
```

It must pass with zero warnings — CI runs the same gate. `make all` wraps the
underlying `cargo` / `ruff` / `vitest` / CMake commands, so there is one source
of truth (see the Makefile or `make help`). For a fast Rust-only inner loop you
can still run:

```bash
cargo fmt && cargo clippy --workspace --all-targets --all-features -- -D warnings && cargo test --workspace
```

### 3. Test optional features

If your change touches file I/O or adds new public API:

```bash
# Test with memory-mapped I/O support
cargo test -p tensogram --features mmap

# Test with async support
cargo test -p tensogram --features async

# Test with both
cargo test -p tensogram --features mmap,async
```

### 4. Update documentation

- **Code changes** should be recorded in `CHANGELOG.md` under the `[Unreleased]` section
- **New public API** should have doc comments and appear in the mdbook guide under `docs/src/`
- **Edge cases** go in `docs/src/edge-cases.md`
- **Wire format changes** go in `docs/src/format/wire-format.md`

Build the docs to check:

```bash
cd docs && mdbook build
cargo doc --workspace --no-deps
```

### 5. Add examples

If you add a new API surface, add a runnable example in `examples/rust/`. The naming convention is `NN_description.rs` (e.g. `11_new_feature.rs`).

### 6. Open a pull request

Use `/prepare-make-pr` (it runs `make all`, commits, pushes, and opens the PR),
or do it by hand:

- Branch name follows **Branch naming** above; commits use conventional-commit
  messages.
- Fill in the PR template (`.github/PULL_REQUEST_TEMPLATE.md`), including the CLA.
- `.github/CODEOWNERS` auto-requests review from the maintainers. A PR needs at
  least one code-owner approval — **two** for the wire format, the C ABI /
  generated header, or security-sensitive code.
- CI must be green. Run `/copilot-review-loop` to convergence and
  `/address-pr-comments` to resolve threads.
- Merge with a **squash merge** and delete the branch. (Mutation testing,
  `make mutants-diff`, is heavy and run only by deliberate human choice.)

The full review / approval / merge policy is in [AGENTS.md](AGENTS.md) under
"Review & merge".

## Code Style

Follow [plans/DESIGN.md](plans/DESIGN.md) principles. When in doubt, read the existing code.

### Principles

1. **Deep modules.** Hide complexity behind an API with a small surface.
2. **Design errors out of existence.** Idempotent ops, never panic, always return Result/error codes.
3. **Stateless.** No global state (thread-local error string in FFI is the sole exception).
4. **Self-contained data objects.** Every data object frame carries all information needed to decode it.

### Rust

- All public functions return `Result<T, TensogramError>` — no `unwrap()` in library code
- No panics in library code — `panic = "abort"` in both release and dev profiles (FFI safety)
- Feature-gate optional dependencies (`#[cfg(feature = "...")]`)
- Prefer `BTreeMap` over `HashMap` for deterministic serialization
- Derive `Debug` on public types

### C FFI

- All functions prefixed `tgm_`
- Opaque handle pattern — callers never see internal structs
- Error codes + `tgm_last_error()` for thread-local error messages
- Every allocation has a matching `_free()` function

### C++ Wrapper

- RAII with `std::unique_ptr` + custom deleters
- Move-only semantics (copy suppressed)
- `[[nodiscard]]` on all accessors
- Typed exception hierarchy mapping C error codes

### Python

- NumPy arrays as the primary data interface
- Context manager for file I/O
- ruff for linting/formatting (E/W/F/I/N/UP/B/SIM/PT/RUF rules)

### Functions

- Prefer short functions (10–50 lines)
- Decompose complex functions into focused helpers

### Comments

- Explain *why*, not *what*
- Document non-obvious design decisions inline
- Doxygen/rustdoc for public API

## Test Structure

The test suite is organised by *shape*, not by fixed count (counts
drift every release). Run `cargo test --workspace` plus the
language-specific runners for the current numbers. For the full map of
what is tested where — unit, integration, adversarial, golden-file,
property-based, remote, importer, WASM, Python, xarray, Zarr, C++, and
mutation testing — see [plans/TEST.md](plans/TEST.md).

Golden binary files in `rust/tensogram/tests/golden/` are checked
into the repo. If the wire format changes, regenerate them by running
`cargo test --test golden_files`.

## Python Bindings

The Python crate is excluded from the default workspace build because it requires a Python interpreter and linker:

```bash
# First time: create a virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install maturin numpy pytest ruff

# Build and install the Rust extension into the active venv
cd python/bindings && maturin develop && cd ../..

# Run core Python tests
python -m pytest python/tests/ -v

# Optional: install and test xarray/zarr backends
uv pip install -e "python/tensogram-xarray/[dask]"
python -m pytest python/tensogram-xarray/tests/ -v
uv pip install -e python/tensogram-zarr/
python -m pytest python/tensogram-zarr/tests/ -v
```

> If `uv` is not available, substitute `python -m venv .venv` and `pip install` for the virtualenv and install steps above.

## C/C++ Bindings

The FFI crate generates `tensogram.h` via cbindgen:

```bash
cargo build -p tensogram-ffi
# Output:
#   rust/tensogram-ffi/tensogram.h     (regenerated each build)
#   target/debug/libtensogram_ffi.a
#   target/debug/libtensogram_ffi.{so,dylib}
```

For the cargo-c flow (versioned shared library, pkg-config descriptor,
installed header) and the binary-tarball release path see the
[C API guide](docs/src/guide/c-api.md).

## TypeScript Wrapper

The TS wrapper lives in `typescript/` and is driven by `wasm-pack` (for the WASM glue)
plus `tsc` + `vitest` (for the TS layer). User-facing docs live in
`docs/src/guide/typescript-api.md`.

```bash
# One-time setup
cargo install wasm-pack        # if not already installed

# Build + test
make ts-build                  # wasm-pack build + tsc
make ts-test                   # vitest
make ts-typecheck              # strict tsc --noEmit across src + tests

# Run an example end-to-end
cd examples/typescript
npm install
npx tsx 01_encode_decode.ts
```

The top-level `make test` / `make lint` now include `ts-test` and `ts-typecheck`.

## Commit Messages

Use conventional commits:

```
feat: add new compression codec
fix: handle zero-length payloads in decode
refactor: decompose encode_message into helpers
docs: update wire format documentation
test: add golden binary test files
chore: update CI configuration
```

## Questions?

Open an issue on GitHub. For design questions, check [plans/DESIGN.md](plans/DESIGN.md) first.

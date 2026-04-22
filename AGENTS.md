# Claude and Other Agents

# Guidelines

- CRITICAL: Always prefer the LSP tool over Grep/Read for code navigation. 
    - Use it to find definitions, references, and workspace symbols.

- CRITICAL: NEVER suppress warnings, lint errors, or test failures with annotations
  (`#[allow(...)]`, `noqa`, `@SuppressWarnings`, etc.) unless the suppression itself
  is the correct semantic choice (e.g., `#[allow(clippy::too_many_arguments)]` on a
  function that genuinely needs many parameters). If a lint fires, fix the underlying
  code. If a test fails, fix the bug. If a warning appears, resolve the root cause.
  Quick workarounds that hide problems are strictly prohibited.

- CRITICAL: Always prefer proper solutions over quick fixes.
  When facing a problem, invest the effort to understand the root cause and fix it
  correctly rather than applying a workaround. Specifically:
  - Do NOT skip platforms or configurations to avoid fixing a build failure.
  - Do NOT add conditional compilation or feature gates to hide broken code.
  - Do NOT remove or weaken tests to make CI pass.
  - Do NOT add TODO/FIXME/HACK comments as a substitute for doing the work now.
  - If a proper fix requires changing multiple files or modules, do it.
  - If a proper fix requires understanding unfamiliar code, read it first.
  - If you are unsure whether a fix is proper, ask before proceeding.

- IMPORTANT: Don't worry about breaking compatibility or being backwards compatible.
    - neither for the API nor for the Wire Format
    - FTM this software has not been make public yet and there is no system using it. 
    - Keep the code simple.

- IMPORTANT: when planing and before you do any work:
  - ALWAYS mention how you would verify and validate that work is correct
  - include TDD tests in your plan
  - take a behaviour driven approach
  - you are very much ENCOURAGED to ask questions to get the design correct
  - ALWAYS seek clarifications to sort out ambiguities
  - ALWAYS provide a summary of the Design and implementation Plan

- IMPORTANT: when you build code and new features:
  - ALWAYS document those features in docs/
  - Remember to add examples (see below)

- IMPORTANT:
  - when you commit your work, make sure it passess all checks, tests and lints -- by running `make all`

# Design & Purpose

- README.md -- entry level generic information
- plans/MOTIVATION.md -- why Tensogram exists and what we're building
- plans/DESIGN.md -- design rationale and key architectural decisions
- plans/STYLE.md -- code style conventions
- plans/WIRE_FORMAT.md -- canonical wire format specification
- plans/DONE.md -- current implementation status (keep updated)
- plans/TODO.md -- features decided to implement (accepted backlog)
- plans/IDEAS.md -- ideas for possible future features (not yet decided)
- plans/TEST.md -- test plan and coverage summary
- CHANGELOG.md -- release history

Follow plans/DESIGN.md principles and plans/STYLE.md conventions in all code.

# Build / lint / test (required before marking done)

## Languages
This project contains Rust, Python, C, C++ and TypeScript code

## Rust
- Build: `cargo build --workspace`
- Format: `cargo fmt`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace`

## Python
- Setup (first time): `uv venv .venv && source .venv/bin/activate && uv pip install maturin numpy pytest ruff`
- Build: `source .venv/bin/activate && cd python/bindings && maturin develop`
- Lint: `ruff check --config python/bindings/pyproject.toml python/tests/`
- Format: `ruff format --config python/bindings/pyproject.toml python/tests/`
- Test: `source .venv/bin/activate && python -m pytest python/tests/ -v`
- xarray tests: `source .venv/bin/activate && uv pip install -e "python/tensogram-xarray/[dask]" && python -m pytest python/tensogram-xarray/tests/ -v`
- zarr tests: `source .venv/bin/activate && uv pip install -e python/tensogram-zarr/ && python -m pytest python/tensogram-zarr/tests/ -v`
- IMPORTANT: ALWAYS run `ruff check` and `ruff format` before committing Python files. CI enforces this.

## TypeScript
- Requires `wasm-pack` (`cargo install wasm-pack`) and Node ≥ 20
- Install deps: `cd typescript && npm install`
- Build: `make ts-build` (runs `wasm-pack build` then `tsc`)
- Typecheck: `make ts-typecheck` (strict, covers src + tests)
- Test: `make ts-test` (vitest)
- Run an example: `cd examples/typescript && npm install && npx tsx 01_encode_decode.ts`
- User-facing docs: `docs/src/guide/typescript-api.md`

## Tensoscope (React SPA)

The interactive web viewer lives at `tensoscope/`.
It depends on the `@ecmwf.int/tensogram` WASM package at `typescript/`.

### Prerequisites
- Build the WASM package first: `cd typescript && make ts-build`

### Dev server
```bash
cd tensoscope && npm install && npm run dev
```
Starts at http://localhost:5173.

### Production build
```bash
cd tensoscope && npm run build
```

### Docker
```bash
cd tensoscope && make build && make run
```
Serves at http://localhost:8000/ (set BASE_PATH env var to deploy under a subpath)

# Version control
- Git project in github.com/ecmwf/tensogram
- IMPORTANT: 
    - versions are tagged using Semantic Versioning form 'MAJOR.MINOR.MICRO'
    - NEVER update MAJOR unless users says so. 
    - Increment MINOR for new features. MICRO for bugfixes and documentation updates.
- NEVER prepend git tag or releases with 'v'
- REMEBER on releases:
    - check all is commited and pushed upstream, otherwise STOP and warn user
    - update the VERSION file
    - git tag with version
    - push and create release in github

- NOTE: SINGLE SOURCE OF TRUTH FOR VERSION — The `VERSION` file at the repo root is the
  canonical version for the ENTIRE project. ALL version strings everywhere MUST match it.
  When bumping the version (e.g. during a release), you MUST update:
    - `VERSION` (the source of truth)
    - The root `Cargo.toml` `[workspace.package]` `version` field — all workspace member
      crates inherit from it via `version.workspace = true`, so a single edit covers:
        `rust/tensogram`, `rust/tensogram-encodings`, `rust/tensogram-sz3`,
        `rust/tensogram-sz3-sys`, `rust/tensogram-szip`, `rust/tensogram-cli`,
        `rust/tensogram-ffi`, `rust/benchmarks`, `examples/rust`.
    - The excluded Cargo.toml files (not workspace members, updated individually):
        `python/bindings/Cargo.toml`
        `rust/tensogram-grib/Cargo.toml`
        `rust/tensogram-netcdf/Cargo.toml`
        `rust/tensogram-wasm/Cargo.toml`
      Also update any pinned workspace dependency version strings in the root `Cargo.toml`
      (e.g. `tensogram-szip = { …, version = "=0.16.1" }`, `tensogram-sz3`, etc.).
    - `pyproject.toml` in EVERY Python package under `python/`, AND
      `examples/jupyter/pyproject.toml` (the Jupyter notebook deps manifest).
    - `package.json` in EVERY JS package — discover them with
      `find . -name package.json -not -path './**/node_modules/*' -not -path './target/*'`
      (currently `typescript/` and `examples/typescript/`; the list grows if
      new JS packages land).
    - `CHANGELOG.md` (new release entry header).
  The provenance encoder in `rust/tensogram/src/encode.rs` reads the version via
  `env!("CARGO_PKG_VERSION")` which comes from Cargo.toml — so keeping Cargo.toml in sync
  with VERSION is critical for correct provenance in encoded messages.
  If ANY of these are out of sync, the release is broken. Always grep for the old version
  string across the repo to catch stragglers.

# Tracking Work Done

Keep track of implementations in plans/DONE.md for all code changes.

# Documentation

Create and maintain documentation under docs/. 
- Easy to follow by average tech person, with well separated topics.
- Use mdbook 
- Add mermaid diagrams when necessary
- Add examples when it becomes hard to follow
- Especially note the edge cases

# Examples

Create and maintain a sub-dir examples/<lang> 
- 1 sub-dir per supported language of the caller Rust, C++, Python
- Populate with examples of caller code showing how to use interfaces
- examplify the most common cases
- show how to use all API functions

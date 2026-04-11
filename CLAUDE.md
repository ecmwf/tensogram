# Claude and Other Agents

# Guidelines

- NOTE: When the user's request matches an available skill:
    - ALWAYS invoke it using the Skill tool as your FIRST action. 
    - Do NOT answer directly, do NOT use other tools first. 
    - The skill has specialized workflows that produce better results than ad-hoc answers.

- CRITICAL: Always prefer the LSP tool over Grep/Read for code navigation. 
    - Use it to find definitions, references, and workspace symbols.

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

- NOTE: When the user asks for "second pass", "third pass" or "N-th pass" perform:
  - simplification opportunities,
  - naming/comments/docs quality review,
  - scan for edge-cases and logical regression,
  - no panics in rust code,
  - all documentation up-to-date with changes,
  - running required formatter/lint/tests

- NOTE: when user asks for 'error handling' checks:
  - verify no panic in rust code
  - verify how errors are handled across-code base, all languages
  - ensure all errors handled and reported correclty with enough information reaching users
  - document all error paths in docs/

- NOTE: when user asks for 'edge cases':
  - look specifically edge cases
  - look for undefined behaviour or ambiguities
  - if necesary, ask the user to clarify 
  - document all those in docs/

- NOTE: when user asks for 'code coverage':
    - explore all the code base looking for code that isn't yet tested. 
    - Look specifically for testing edge cases.
    - Aim to have at least 95% test coverage.

- NOTE: When user asks for 'final prep' make:
    - final check everything builds, all languages and all tests pass
    - all examples in all languages Rust, Python and C++ compile and run
    - all docs build
    - if successful, carefully:
        - select files and contributions to git add
        - ignore the build files and artifacts, don't add hidden directories
        - if not in a branch, create a new properly named branch
        - git commit
        - make a pull request to upstream github project

- NOTE: when user asks to do 'pr reply' or 'pull request reply':
    - check github pull request reviews
    - consider them with respect to the phylosophy and aims of this software
    - if in doubt seek user clarifications
    - fix code and address the raised issues
    - update the docs/
    - make a summary and push your changes to update the PR
    - poll to wait for the CI to finish running
    - continue iterating until all recomentations and issues were addressed

- NOTE: When user asks for 'make release' execute:
    - check all changes are commited and pushed upstream
    - final check everything builds, all languages and all tests pass
    - all examples in all languages Rust, Python and C++ compile and run
    - all docs build
    - if any of the above fails STOP and prompt the user for action
    - otherwise, proceed by check the latest version upstream and in VERSION file
    - if needed, bump in VERSION file, commit and tag then push to upstream
    - make a Github release

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
- plans/PLAN.md -- HISTORICAL: iteration 0 planning decisions (read-only)
- CHANGELOG.md -- release history

Follow plans/DESIGN.md principles and plans/STYLE.md conventions in all code.

# Build / lint / test (required before marking done)

## Languages
This project contains Rust, Python, C and C++ code

## Rust
- Build: `cargo build --workspace`
- Format: `cargo fmt`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace`

## Python
- Setup (first time): `uv venv .venv && source .venv/bin/activate && uv pip install maturin numpy pytest ruff`
- Build: `source .venv/bin/activate && cd crates/tensogram-python && maturin develop`
- Lint: `ruff check --config crates/tensogram-python/pyproject.toml tests/python/`
- Format: `ruff format --config crates/tensogram-python/pyproject.toml tests/python/`
- Test: `source .venv/bin/activate && python -m pytest tests/python/ -v`
- xarray tests: `source .venv/bin/activate && uv pip install -e "tensogram-xarray/[dask]" && python -m pytest tensogram-xarray/tests/ -v`
- zarr tests: `source .venv/bin/activate && uv pip install -e tensogram-zarr/ && python -m pytest tensogram-zarr/tests/ -v`
- IMPORTANT: ALWAYS run `ruff check` and `ruff format` before committing Python files. CI enforces this.

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
  When bumping the version (e.g. during a release), you MUST update ALL of these locations
  to match the VERSION file:
    - `VERSION` (the source of truth)
    - `Cargo.toml` in EVERY crate: tensogram-core, tensogram-encodings, tensogram-cli,
      tensogram-ffi, tensogram-python, tensogram-grib, benchmarks, examples/rust
    - `pyproject.toml` in EVERY Python package: crates/tensogram-python, tensogram-xarray,
      tensogram-zarr
    - `CHANGELOG.md` (new release entry header)
  The provenance encoder in `crates/tensogram-core/src/encode.rs` reads the version via
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

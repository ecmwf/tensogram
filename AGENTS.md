# Claude and Other Agents

This file (`AGENTS.md`) is the canonical agent-instructions document.
`CLAUDE.md` is a symlink to it for cross-tool compatibility — edit `AGENTS.md`
and refer to it by that name.

# Guidelines

- If an LSP / symbol-navigation tool is available, prefer it over Grep/Read for
  code navigation — use it to find definitions, references, and workspace symbols.
  Fall back to Grep/Read when no such tool is exposed.

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
  - when you commit your work, make sure it passes all checks, tests and lints -- by running `make all`

- IMPORTANT: Derived artefacts have one source of truth.
  If a file is produced from another file or tool invocation — codegen output,
  schema-generated bindings, vendored headers, generated docs, lockstep API
  maps, compiled assets — do not maintain an unmanaged copy elsewhere.
  Consumers should read from the canonical generated location, or the
  duplicate must be guarded by a CI step that regenerates from source and
  fails on any drift. The same rule applies to values that are conceptually
  one piece of data but appear in many places (release version, ABI version,
  wire-format version, error code lists, dtype tables): pick one source of
  truth and either generate the others from it or add a consistency check.

- IMPORTANT: Build-system wrappers must not replace the inner tool's
  freshness logic.
  When CMake / Make / Bazel / shell scripts / CI jobs invoke another build
  tool (Cargo, npm, maturin, wasm-pack, Gradle, Docker, another Makefile),
  the wrapper must either invoke the inner tool unconditionally or declare a
  complete dependency graph for it. Do not key freshness solely on "output
  file exists" — the inner tool already knows what is stale, and an mtime
  check from the outer tool can serve a days-old artefact to today's
  consumers with no warning. Prefer always-running wrapper targets
  (e.g. `add_custom_target` over `add_custom_command(OUTPUT ...)` when the
  command is a nested incremental build tool); use output-file rules only
  for truly declarative single-input → single-output generators with all
  inputs listed in `DEPENDS`.

- IMPORTANT: NEVER use process-ephemeral references in code, comments, docstrings,
  commit messages, or planning documents. The code outlives the workflow that
  produced it; references to that workflow age into noise.

  Banned vocabulary (each becomes meaningless once the PR merges):
  - Workflow ordinals: "Phase 2", "Pass 5", "Round 1", "Step 3"
  - Issue-tracker references: "PR-1", "sub-task 4", "issue #94"
  - Review-feedback severity buckets: "Critical #1", "High #2"
  - History phrases inside code: "before this PR", "after review feedback",
    "fixed in commit abc1234"

  Replace each with a name that describes what the thing **is** or **does**:
  - "Round 1 paired fetch" → "the paired postamble fetch"
  - "Phase 2 state refactor" → "extracting the bidirectional state machine"
  - "Pass 5 polish" → "documentation cleanup"
  - "Critical #1 invariant" → "`fwd_terminated` cascades to `disable_backward`"

  Git history records chronology; comment and commit text must record
  substance. A future maintainer reading "Round 2" or "Phase 4" gains
  nothing — those ordinals reference the author's mental sequence at write
  time, which the reader has no access to.

# Design & Purpose

- README.md -- short entry-level overview (also the GitHub landing page)
- plans/MOTIVATION.md -- why Tensogram exists and what we're building (long form)
- plans/DESIGN.md -- design rationale and key design decisions
- plans/ARCHITECTURE.md -- crate structure, module map, feature gates
- plans/WIRE_FORMAT.md -- canonical wire format specification
- plans/TODO.md -- features decided to implement (accepted backlog)
- plans/IDEAS.md -- speculative ideas + long-form "Horizon" brainstorm (not yet decided)
- plans/TEST.md -- test plan and coverage shape
- CONTRIBUTING.md -- contributor setup, workflow, and code style
- CHANGELOG.md -- release history + the [Unreleased] record of merged work

Follow plans/DESIGN.md principles and the Code Style section of CONTRIBUTING.md in all code.

# Build / lint / test (required before marking done)

This project contains Rust, Python, C, C++, Fortran and TypeScript code. The
top-level `Makefile` is the single entry point — run `make help` to list every
target. The full gate before marking work done is:

```bash
make all          # = make check test lint  (the gate; run before committing)
```

Useful focused targets (run `make help` for the complete list):

| Target | What it does |
|--------|--------------|
| `make check` | Compile the Rust workspace (default + `--no-default-features`) |
| `make test` | `rust-test` + `python-test` + `ts-test` |
| `make lint` | `rust-lint` + `python-lint` + `python-fmt` + `ts-typecheck` |
| `make fmt` | Check Rust + Python formatting |
| `make rust-test` / `rust-lint` / `rust-fmt` | Rust only |
| `make python-build` / `python-test` / `python-lint` / `python-fmt` | Python (maturin into `.venv`) |
| `make ts-build` / `ts-test` / `ts-typecheck` | TypeScript (wasm-pack + tsc + vitest) |
| `make cpp-build` / `cpp-test` | C++ wrapper (CMake + GoogleTest) |
| `make fortran-build` / `fortran-test` / `fortran-f2008-check` | Fortran binding |
| `make wasm-test` / `doc-examples` / `docs-build` | WASM, runnable doc examples, mdBook |
| `make version-check` / `bump-version VERSION=X.Y.Z` | Version sync (see *Version control*) |

- IMPORTANT: ALWAYS run `ruff check` and `ruff format` (via `make python-lint`
  / `python-fmt`) before committing Python files. CI enforces this.
- For the raw per-tool commands (and the first-time Python/uv setup), see
  the *Building* and *Testing* sections of `CONTRIBUTING.md` — the Makefile
  targets wrap exactly those commands, so there is one source of truth.

## Tensoscope (React SPA)

The interactive web viewer lives at `tensoscope/`.
It depends on the `@ecmwf.int/tensogram` WASM package at `typescript/`.

### Prerequisites
- Build the WASM package first: `make ts-build` (root Makefile target)

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

# Development workflow

The lifecycle below maps each phase to the bundled slash-command skills in
`.prompts/` (opencode resolves them via `.opencode/commands` → `.prompts`) and,
when your agent harness provides them, to specialised subagents. Humans follow
the same phases, just without the `/`-commands. CONTRIBUTING.md carries the
human-facing version of this list.

1. **Onboard** — `/onboard` reads the docs and surveys the code (use a
   codebase-explorer subagent for the survey).
2. **Pick work** — take an item from `plans/TODO.md` (the accepted backlog).
   `plans/IDEAS.md` is speculative and is NOT committed direction.
3. **Plan** — before writing code, say how you will verify it, plan the TDD
   tests, take a behaviour-driven approach, ask clarifying questions, and
   summarise the design + implementation plan (see Guidelines above). Use a
   planner subagent if you have one; an external-research subagent for
   unfamiliar formats/libraries.
4. **Branch** — branch off `main` using the convention in CONTRIBUTING.md
   ("Branch naming"): `<type>/<kebab-summary>`, where `<type>` matches the
   commit types (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`,
   `perf`, `build`; plus `security` / `hardening` for audit work).
5. **Develop** — follow the conventions here and CONTRIBUTING.md's Code Style.
   Document new features under `docs/` and add a runnable `examples/<lang>/`
   example for any new public API. Record every user-facing change in
   `CHANGELOG.md [Unreleased]` as you go — that is the only work-tracking
   record (no separate status file, no issue ordinals). A domain-specialist
   subagent fits the implementation; a security-auditor subagent for wire
   format / FFI / adversarial-sensitive changes.
6. **Quality passes** — `/make-further-pass` (strictness-graded), plus, as
   needed, `/improve-code-coverage`, `/improve-edge-cases`,
   `/improve-error-handling`, and `/doc-fact-check`. A reviewer/auditor subagent
   fits here. Mutation testing (`make mutants-diff`) is heavy to run locally and
   is ALWAYS a deliberate human decision — it is NOT part of the default gate.
7. **Gate** — `make all` must be green (the single pre-commit gate; see
   Build / lint / test above).
8. **Open the PR** — `/prepare-make-pr` runs `make all`, then commits on a
   feature branch with a conventional-commit message, pushes, and opens the PR
   against `.github/PULL_REQUEST_TEMPLATE.md`. Only commit / push / open a PR
   when the user has asked you to.
9. **Review loop** — `/copilot-review-loop` drives the Copilot reviewer to
   convergence; `/address-pr-comments` resolves human review threads and waits
   on CI (`gh pr checks --watch`). Review ownership is in `.github/CODEOWNERS`.
10. **Release** — `/make-release`; full procedure in `docs/src/dev/releasing.md`
    and "Release process" below.

## Review & merge

- All changes land on `main` through a pull request from a `<type>/…` branch;
  `main` is protected against direct pushes.
- `.github/CODEOWNERS` auto-requests review from the maintainers. A PR needs at
  least one approving review from a code owner — **two** for the wire format
  (`plans/WIRE_FORMAT.md`, `rust/tensogram/src/wire.rs` / `framing.rs`), the C
  ABI / generated header (`rust/tensogram-ffi/`), or security-sensitive code.
- CI (`ci.yml`) must be green before merge; `release-preflight` must be green
  before any release tag.
- Run `/copilot-review-loop` to convergence before requesting human review;
  `/address-pr-comments` resolves threads. Pushing new commits dismisses stale
  approvals, so re-request review after addressing feedback.
- Merge with whichever GitHub strategy fits the change — **squash**, **rebase**,
  or a **merge commit** (all are enabled) — then delete the branch. Release PRs
  default to **rebase** (`/make-release` takes an optional `merge|squash|rebase`
  override).
- Agents NEVER push, merge, or open a PR without explicit user approval.

# Version control
- Git project in github.com/ecmwf/tensogram
- IMPORTANT: 
    - versions are tagged using Semantic Versioning form 'MAJOR.MINOR.MICRO'
    - NEVER update MAJOR unless users says so. 
    - Increment MINOR for new features. MICRO for bugfixes and documentation updates.
- NEVER prepend git tag or releases with 'v'
- Release process (canonical ordered procedure; full rationale + per-step
  detail in `docs/src/dev/releasing.md`):
    1. Ensure `main` is green and every user-facing change is recorded in
       `CHANGELOG.md` under `[Unreleased]`.
    2. `make bump-version VERSION=X.Y.Z` — rewrites every manifest from the
       `VERSION` source of truth (see the list below). By hand, move the
       `[Unreleased]` CHANGELOG entries under a new `## [X.Y.Z] - DATE` heading.
    3. `make all` — the full build/test/lint gate (Rust + Python + TS + C++ +
       WASM + cargo-c + Fortran).
    4. `make release-check` — the release-only gates `make all` does NOT run
       (version-check, crate packaging + leaf-crate dry-run publish, the
       cargo-c header-drift diff, Python wheel + `twine` metadata, npm
       wasm-blob guard). Requires `cargo-c`, `uv`, and Node on PATH.
    5. Commit and push everything. STOP and warn the user if anything is
       uncommitted — a tag must point at a clean tree.
    6. Optionally dispatch the `release-preflight` GitHub workflow — the
       authoritative pre-tag gate, which additionally runs the grib/netcdf +
       macOS matrices and real dry-run publishes on a clean checkout.
    7. `git tag X.Y.Z` (NO `v` prefix), push the tag, and create the GitHub
       release. The tag push triggers the `publish-crates` / `publish-pypi` /
       `publish-npm` / `publish-ffi` workflows.

- NOTE: SINGLE SOURCE OF TRUTH FOR VERSION — The `VERSION` file at the repo root is the
  canonical version for the ENTIRE project. ALL version strings everywhere MUST match it.

  **Use the tooling.** Do not hand-edit each manifest:
    - `make bump-version VERSION=X.Y.Z` rewrites every version string below and
      then greps the tree for stragglers (it wraps `scripts/bump_version.py`).
    - `make version-check` verifies every manifest already matches `VERSION`
      (CI guard; no edits).

  The bump script edits all of these (listed here so the contract is reviewable):
    - `VERSION` (the source of truth).
    - The root `Cargo.toml` `[workspace.package]` `version` field — all workspace member
      crates inherit it via `version.workspace = true` (covers `rust/tensogram`,
      `rust/tensogram-encodings`, `rust/tensogram-sz3`, `rust/tensogram-sz3-sys`,
      `rust/tensogram-szip`, `rust/tensogram-cli`, `rust/tensogram-ffi`,
      `rust/benchmarks`, `examples/rust`).
    - The non-member Cargo.toml files (own `version` field): `python/bindings`,
      `rust/tensogram-grib`, `rust/tensogram-netcdf`, `rust/tensogram-wasm`.
    - Pinned internal workspace dependency strings (`version = "=X.Y.Z"`) wherever
      one crate references a sibling — these live in `rust/tensogram`,
      `rust/tensogram-cli`, `rust/tensogram-ffi`, `rust/tensogram-sz3`,
      `rust/tensogram-grib`, `rust/tensogram-netcdf`, `rust/tensogram-wasm`, and
      the root `Cargo.toml`.
    - `pyproject.toml` in EVERY Python package under `python/` (`bindings`,
      `tensogram-xarray`, `tensogram-zarr`, `tensogram-anemoi`,
      `tensogram-earthkit`) AND `examples/jupyter/pyproject.toml`.
    - Version-controlled `package.json` files (`typescript/`,
      `examples/typescript/`, and `tensoscope/`). The tensoscope viewer app
      follows the project version like every other subpackage; its committed
      `tensoscope/package-lock.json` mirrors that version in two
      self-referential blocks, which the script also rewrites (and only those —
      never a dependency entry). NOTE: `typescript/wasm/package.json` is
      **generated by wasm-pack and git-ignored** — never hand-edit it; it is
      regenerated from the crate version at build time. (The straggler `find`
      also turns up `tests/remote-parity/drivers/` and `.opencode/`, which are
      tooling and NOT on the sync list.)
    - `fortran/fpm.toml` `version` and the `project(... VERSION ...)` line in
      `fortran/CMakeLists.txt`.

  The script does NOT touch (do these by hand): `CHANGELOG.md` (add the new
  release entry header), and Python dependency *constraint ranges*
  (`tensogram>=X.Y.Z,<X.Y+1`) — the `<` ceiling is a deliberate compatibility
  policy, so the script surfaces them for review rather than rewriting them.

  Why it matters: the provenance encoder in `rust/tensogram/src/encode.rs` reads
  the version via `env!("CARGO_PKG_VERSION")` (from Cargo.toml), so a Cargo.toml
  out of sync with `VERSION` stamps wrong provenance into encoded messages. If any
  version string is out of sync the release is broken — `make version-check`
  catches it.

# Tracking Work Done

Record every code change in `CHANGELOG.md` under the `[Unreleased]`
section as it merges — this is the single backward-looking record
(there is no separate status file). Keep entries user-facing and
concise. Durable design decisions and rationale go in
`plans/DESIGN.md`; behaviour and usage go in `docs/`.

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

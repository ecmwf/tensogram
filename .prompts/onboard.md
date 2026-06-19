---
description: Onboard the agent to the Tensogram project by reading the docs and surveying the code
agent: plan
---

# Onboard to the Project

Build a working mental model of this repository so you can act competently
on it. Do NOT rely on any knowledge baked into this command — derive
everything from the repository's own files as they exist right now.

## Arguments

`$ARGUMENTS` — optional focus area (e.g. `wire-format`, `python`, `cli`,
`encodings`, `remote`). If given, go deeper on that area during the code
survey (Step 3) while still doing the full doc pass. If empty, do a balanced
survey across the whole project.

## 1. Read the project-level docs

Read these first, in order. They are the source of truth for *intent*:

- The root markdown files: `README.md`, `AGENTS.md` (canonical agent
  instructions; `CLAUDE.md` is a symlink to it), `CONTRIBUTING.md`,
  `CHANGELOG.md` (read the most recent `[Unreleased]` + last few releases
  for current direction — do not read the whole history).
- Every `plans/*.md` file. Read each one fully. These cover motivation,
  design decisions, architecture, the wire format, the test strategy,
  security analysis, the accepted-work backlog, and speculative ideas.

If a root markdown file or a `plans/*.md` file exists that is not named
above, read it too — discover them with a glob rather than assuming a fixed
list. Treat `IDEAS.md` as speculative/unreviewed (it says so itself), not as
committed direction.

## 2. Map the repository structure

Without reading every file, establish the layout:

- List the workspace layout (the root `Cargo.toml` `[workspace]` members and
  `exclude` list, the `python/` packages, and the other language dirs:
  C/C++, Fortran, TypeScript/WASM, plus any tooling/viewer dirs).
- Identify the build entry point and how the languages fit together
  (start from the `Makefile` — run nothing, just read the target graph).
- Note where tests, examples, and docs live.

## 3. Survey the code (representative, not exhaustive)

Read enough source to understand *how the project achieves its purpose* —
the public surface and the core data flow — NOT every file. Aim for:

- The core crate's public API and primary types (encode / decode / the
  message + frame model / metadata / the on-wire structures).
- The main data flow for the project's central operation, end to end
  (follow it from the entry point through to the bytes / back).
- How the encoding/compression pipeline is structured and dispatched.
- How one binding wraps the core (pick the most relevant to `$ARGUMENTS`,
  or the richest one if no focus given), to learn the FFI/binding pattern.
- The error model and any project-wide invariants the code enforces
  (cross-check what `plans/*.md` claims against what the code actually does).

Prefer LSP / symbol navigation where available; fall back to Grep/Read.
When the docs and the code disagree, trust the code and note the drift.

## 4. Report back

Produce a concise onboarding summary (do not change any files):

1. **Purpose** — what the project is and the problem it solves, in 2-3 lines.
2. **Architecture** — the crates/packages and how they relate (a short list
   or a small diagram).
3. **Core data flow** — the central operation traced end to end.
4. **Conventions that constrain edits** — the non-negotiable rules an agent
   must respect here (pulled from `AGENTS.md` / `CONTRIBUTING.md` /
   `plans/DESIGN.md`), e.g. the version single-source-of-truth, the error/
   panic policy, determinism/cross-language-parity requirements, the
   commit/PR workflow.
5. **How to build, test, and lint** — the canonical commands (the gate to
   run before marking work done).
6. **Current direction & open work** — what's active vs. speculative, from
   `CHANGELOG.md [Unreleased]` and the accepted backlog.
7. **Anything stale or contradictory** you noticed between docs and code.

Keep it tight and skimmable. The goal is a shared mental model, not a
copy of the docs.

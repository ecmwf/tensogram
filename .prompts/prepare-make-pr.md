---
description: Run pre-flight checks then commit, push and open the PR
agent: build
---

# Prepare and Create Pull Request

Final preparation check and PR creation workflow.

## Pre-flight Checks

Run the full gate. If it fails, STOP and report the failure — do not open the PR.

```bash
make all          # build + test + lint across Rust, Python, TS, C++, WASM, cargo-c, Fortran
```

`make all` is the single source of truth for the pre-commit gate; it wraps the
underlying `cargo` / `ruff` / `vitest` / CMake commands (run `make help` for the
individual targets). Do NOT run a narrower subset in its place.

Mutation testing (`make mutants-diff`) is heavy and is a deliberate human
decision — do not run it as part of opening a PR unless explicitly asked.

(Releases have an extra gate, `make release-check`, and their own skill —
see `/make-release` and `docs/src/dev/releasing.md`.)

## PR Creation

If all checks pass:

1. Review what files to include — stage only source, test, doc, and config files
2. Do NOT stage: build artifacts, hidden directories (except `.claude/`, `.github/`), `*.so`, `*.dylib`, `target/`, `.venv/`, `Cargo.lock` inside subcrates
3. If not already on a feature branch, create one with a descriptive name
4. Commit with a clear, conventional-commit-style message
5. Push to origin
6. Create a pull request with:
   - Title: concise summary of the change
   - Body: structured summary with Added/Changed/Fixed sections
   - Link to any related issues
7. Report the PR URL

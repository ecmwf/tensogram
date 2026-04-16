# Documentation Fact-Check

Run code examples in the docs and verify prose claims against the actual codebase.

## Arguments

`$ARGUMENTS` — optional file paths relative to `docs/src/` to narrow the scope.

If no arguments given, check all `.md` files under `docs/src/`. Accepts individual files (`guide/python-api.md`) or directories (`guide/`, `cli/`).

Examples: `/doc-fact-check guide/python-api.md`, `/doc-fact-check cli/`, `/doc-fact-check`

## Setup

```bash
cargo build --workspace
source .venv/bin/activate
cd python/bindings && maturin develop && cd ../..
python -c "import tensogram; print('OK')"
```

If any setup step fails, report it and continue with the languages that work. Blocks that fail due to environment issues (missing tool, unbuilt dependency) are **not** doc bugs — report them separately as setup problems.

## 1. Run Code Examples

For each fenced code block in the selected docs:

### Classify each block
- **Runnable** — self-contained: has its own imports (Python) or `fn main()` (Rust), no `...` ellipsis placeholders
- **Continuation** — follows a runnable block on the same page and reuses its variables; concatenate with its predecessor
- **Skip** — partial snippets, signatures-only, output samples (lines starting with `$`), unlabeled code blocks (no language tag), install/setup commands (`pip install`, `cargo install`, `uv pip install`), or blocks needing user-specific files (`forecast.tgm` etc.)

### Execute runnable blocks

**Python:** write to a temp `.py` file, run `python <file>`. 30s timeout per block.

**Rust:** only `fn main()` blocks with complete `use` statements. Create a temp Cargo project **inside the repo root** (so path dependencies resolve) with `tensogram-core` as a path dependency, `cargo build` it. Clean up after.

**Bash:** only commands that can run without user-specific data. Run and check exit code 0.

**Never** invent missing imports or variables. If a block isn't self-contained, skip it.

Record pass/fail/skip for every block.

## 2. Check Claims Against Code

Read each doc file and find verifiable factual claims — these are inline code spans (`` `EncodeOptions` ``), numbers ("200 tests"), and table cells, not prose paragraphs. For each one, check the source.

**What to check:**
- **API signatures** — struct names, field names, function names, parameter names, return types mentioned in docs → verify they exist and match in `rust/` source (e.g. `rust/tensogram-core/src/`, `rust/tensogram-encodings/src/`) using LSP or reading the source directly
- **Enum/constant values** — dtype names, encoding names, compression names, magic bytes, terminators → verify against actual definitions
- **Default values** — "default hash is xxh3", "row-major by default" etc. → check `Default` impls
- **Numerical claims** — test counts, dtype counts, supported compression count → run the actual count
- **CLI flags and subcommands** — run `tensogram <cmd> --help` and verify documented options exist
- **Feature flags** — verify Cargo feature names mentioned in docs exist in actual `Cargo.toml` files
- **Descriptor key tables** — cross-check documented keys against actual struct/enum fields

**What NOT to check:** subjective claims ("fast", "efficient"), architecture descriptions, explanatory prose that doesn't make a testable assertion.

## 3. Report

For each finding:
```
[ERROR|STALE|DRIFT] file.md:LINE — Summary
  Docs say: <what the docs claim>
  Code says: <what the source actually shows>
```

- **ERROR** — code example fails, API doesn't exist, wrong signature
- **STALE** — outdated value, renamed API, removed feature
- **DRIFT** — minor mismatch (count off by 1, slightly different default)

End with a summary: total files checked, blocks run/passed/failed/skipped, number of claim issues found.

For ERROR findings: propose a fix but do **not** apply it — present for user review.

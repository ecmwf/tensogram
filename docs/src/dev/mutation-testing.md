# Mutation Testing

Mutation testing is a technique for measuring the depth of a test suite.
The tool systematically corrupts the source code — flipping operators,
replacing return values, deleting statements — and then re-runs the
tests. If the tests still pass after a mutation, that *surviving mutant*
marks untested behaviour: a place where the code could silently change
without any test noticing. Survivors are the signal; killed mutants
confirm the suite already covers that logic.

## Why tensogram uses it

Tensogram's critical path is a small set of Rust modules that handle
wire-format layout, bit-flag dispatch, frame boundaries, and hash
verification. These modules share three properties that make them
ideal targets for mutation testing:

- **Bit-flag handling.** Per-frame flags like `HASH_PRESENT` rely on
  bitwise operators (`&`, `|`, `^`). Swapping one for another is the
  exact mutation class `cargo-mutants` generates, and a missed swap
  means silent data corruption.
- **Off-by-one frame-bound bugs.** Boundary arithmetic in the decoder
  (`<` vs `<=`, `+1` vs `-1`) is another mutation sweet spot. The
  Pass-7 frame-bound bug caught during PR #111 review is a textbook
  example.
- **Wire-format byte-equality.** Tensogram guarantees that the same
  input produces bit-identical output across Rust, Python, C++, and
  TypeScript. Golden-file tests enforce this, but mutation testing
  reveals whether those golden files actually exercise every branch
  in the encoder and decoder.

The full rollout plan, including phasing, shard strategy, and triage
SLAs, lives in [`MUTATION_TESTING.md`](../../../MUTATION_TESTING.md) at
the repository root.

## The two regimes

Mutation testing runs in two complementary modes:

### PR-time (`--in-diff`)

Every pull request runs `cargo mutants --in-diff origin/main..HEAD` as
a non-blocking job inside
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml). This
restricts mutation to lines the PR actually touches, keeping wall-clock
time reasonable (typically under five minutes). The job uploads
`mutants.out/` as a CI artifact so reviewers can inspect survivors
without running locally. The plan is to flip this job to
required-for-merge once Phase 1 stabilises.

### Nightly full sweep (sharded)

A scheduled workflow in
[`.github/workflows/mutants-nightly.yml`](../../../.github/workflows/mutants-nightly.yml)
runs the full mutation sweep across all configured modules, split into
eight shards (`--shard 1/8` through `--shard 8/8`) for parallelism.
It triggers on weekdays at 02:00 UTC. Surviving mutants automatically
open GitHub issues tagged `mutation-testing` for triage within seven
days.

> **Note:** The nightly workflow is not yet live — it will be added in a
> subsequent step of the rollout plan.

## Running mutation testing locally

Install the pinned version of `cargo-mutants`:

```bash
cargo install cargo-mutants --version 27.0.0 --locked
```

**Full sweep of a single file** (useful when closing out a Phase-1
step):

```bash
cargo mutants -p tensogram --file rust/tensogram/src/hash.rs --jobs 4
```

**Diff-only** (the most common workflow for PR authors — mutates only
lines you changed):

```bash
cargo mutants --in-diff origin/main..HEAD
```

Both commands write results to `mutants.out/` in the current directory.

## Reading `mutants.out/`

After a run completes, `cargo-mutants` writes four result files into the
`mutants.out/` directory:

- **`caught.txt`** — Mutants that were killed by the test suite. This is
  the happy path: the tests detected the corruption and failed. A long
  `caught.txt` and an empty `missed.txt` is the goal.

- **`missed.txt`** — Surviving mutants. Each entry describes a source
  location and the mutation that was applied. Survivors indicate either
  a genuine coverage gap (write a test) or an equivalent mutant (the
  mutation does not change observable behaviour — add an exemption).
  This is the file you triage.

- **`timeout.txt`** — Mutants that caused the test suite to hang until
  the timeout expired. These are usually equivalent or dead-code
  mutations (e.g. removing a loop-break that the test harness never
  reaches). Worth a quick look, but rarely actionable.

- **`unviable.txt`** — Mutants that did not compile. Expected for
  type-constrained mutations (e.g. replacing a `u32` return with
  `String`). These are noise and can be ignored.

## When to add a test vs. an exemption

When you encounter a surviving mutant, follow this decision tree:

1. **Does the mutation change observable behaviour?** Read the diff
   `cargo-mutants` prints. If flipping that operator or deleting that
   statement would produce incorrect output, a wrong error, or a
   panic in production, the answer is yes — write a test.

2. **Is the mutant equivalent?** Some mutations produce code that is
   functionally identical to the original (e.g. replacing
   `x > 0` with `x >= 1` when `x` is always a positive integer). If
   you can convince yourself (and a reviewer) that no input
   distinguishes the original from the mutant, add an `exclude_re`
   entry to `.cargo/mutants.toml`.

3. **Is the code cosmetic or logging-only?** Display implementations,
   `fmt::Debug` overrides, and log-line formatting are legitimate
   exemption targets. Add an `exclude_re` entry.

Two strict rules apply in all cases:

- **No mass suppression by category.** Do not add broad patterns like
  `"impl Display"` or `"fn fmt"` that suppress entire classes of
  mutants across the codebase. Each exemption must be scoped to a
  specific function or pattern.
- **Every `exclude_re` entry needs a rationale.** Add a one-line
  comment above the entry in `.cargo/mutants.toml` explaining *why*
  the mutant is equivalent or cosmetic.

## Bumping cargo-mutants

The version is pinned at **27.0.0** across local installs, CI workflows,
and this documentation. Version bumps land in dedicated PRs that:

1. Update the version in `.cargo/mutants.toml` comments, CI workflow
   files, and this page.
2. Re-run the full sweep on `rust/tensogram/src/hash.rs` as a smoke
   test to confirm the new version produces comparable results.
3. Document any changed mutant operators or output-format differences
   in the PR description.

## References

- [`MUTATION_TESTING.md`](../../../MUTATION_TESTING.md) — full rollout
  plan at the repository root (phasing, shard strategy, triage SLAs,
  anti-patterns).
- [`plans/TEST.md`](../../../plans/TEST.md) — test plan covering the
  full suite shape, including the mutation testing layer.
- [mutants.rs](https://mutants.rs/) — upstream `cargo-mutants`
  documentation.

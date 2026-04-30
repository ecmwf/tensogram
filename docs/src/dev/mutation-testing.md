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

### Concurrency — three knobs that compound

cargo-mutants concurrency is **not** a single number.  Three independent
layers each fan out by default to one thread per logical CPU; multiplied
together they saturate a 12-core laptop even at `CARGO_MUTANTS_JOBS=2`.
**All three must be clamped** for a comfortable load profile:

| Layer | Knob | Default | Recommended |
|---|---|---|---|
| Mutant workers | `CARGO_MUTANTS_JOBS` env (= `--jobs`) | NCPUS | `2` |
| Build / rustc fan-out | `--jobserver-tasks` flag | NCPUS | `2` |
| Test-binary threads | `cargo test -- --test-threads=N` | NCPUS | `2` (already pinned in `.cargo/mutants.toml`) |
| Library-internal rayon | `TENSOGRAM_THREADS` env | NCPUS | `1` |

**Canonical local invocation** for a Phase-1 file sweep:

```bash
CARGO_MUTANTS_JOBS=2 \
TENSOGRAM_THREADS=1 \
cargo mutants -p tensogram --file rust/tensogram/src/hash.rs --jobserver-tasks 2
```

The test-thread cap is committed to `.cargo/mutants.toml` so it doesn't
need to appear on every command line.

| Use case | Jobs | Jobserver tasks | TENSOGRAM_THREADS |
|---|---|---|---|
| Default (12-core laptop) | `2` | `2` | `1` |
| Strict serial (unattended overnight) | `1` | `1` | `1` |
| Workstation with ≥16 cores + good cooling | `4` | `4` | `1` |

**Why all four clamps?** Without them, `CARGO_MUTANTS_JOBS=2` still
produces ~24 simultaneous rustc threads × ~24 concurrent test threads
on a 12-core box because each layer fans out independently.  Load
average climbs to ~18 (saturating), thermal throttling kicks in, and
laptops have been observed to power down mid-sweep.  The four clamps
together bring sustained load to ~4 on the same hardware.

### Common invocations

**Full sweep of a single file** (useful when closing out a Phase-1
step):

```bash
# Default: 2 mutant workers × 2 jobserver tasks × 2 test threads × 1 rayon
CARGO_MUTANTS_JOBS=2 TENSOGRAM_THREADS=1 \
  cargo mutants -p tensogram --file rust/tensogram/src/hash.rs --jobserver-tasks 2

# Faster on a workstation
CARGO_MUTANTS_JOBS=4 TENSOGRAM_THREADS=1 \
  cargo mutants -p tensogram --file rust/tensogram/src/hash.rs --jobserver-tasks 4
```

**Diff-only** (the most common workflow for PR authors — mutates only
lines you changed):

```bash
CARGO_MUTANTS_JOBS=2 TENSOGRAM_THREADS=1 \
  cargo mutants --in-diff origin/main..HEAD --jobserver-tasks 2
```

Both commands write results to `mutants.out/` in the current directory.

### Resuming an interrupted sweep

Long sweeps (notably the framing-module sweep at ~12 hours with the
default concurrency) can be interrupted by power events, OOM kills, or
laptop sleep. cargo-mutants does not have a native resume flag, but
`--shard N/K` partitions the mutant list deterministically and lets you
re-run a subset:

```bash
# Run quarter 1 of 4 — first ~138 mutants of framing.rs
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 1/4

# Then quarter 2, 3, 4 in subsequent sessions
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 2/4
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 3/4
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 4/4
```

Shard assignment is stable across runs, so re-running a single shard
re-tests the same mutants. `mutants.out/` is overwritten per
invocation; copy or rename between shard runs if you want to preserve
per-shard results.

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

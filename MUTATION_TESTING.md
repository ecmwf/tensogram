# Plan — Mutation Testing for Tensogram

**Tool:** [`cargo-mutants`](https://mutants.rs/) 27.0.0 (pinned in
`.cargo/mutants.toml` and CI workflows).
**Scope:** Rust workspace only — `tensogram`, `tensogram-encodings`, `tensogram-szip`,
`tensogram-ffi`, `tensogram-sz3`. (Other-language bindings keep their existing
coverage stories: golden files for byte parity; see `plans/TEST.md`.)
**Status:** Phase 1 steps 1.1–1.7 complete and merged via PR #114; step 1.8
deferred to Phase 2 weekly machinery (option 3 — see §7).  Phase 2 workflow
(`mutants-weekly.yml`) lives on this PR and goes live the first weekend
after merge.

> **Companion docs:**
> - User-facing reference: [`docs/src/dev/mutation-testing.md`](docs/src/dev/mutation-testing.md)
>   (published in the mdbook).
> - Test-shape map for AI Code Agent sessions: [`plans/TEST.md`](plans/TEST.md)
>   *Mutation testing* section.
> - Per-mutation exemption rationales: [`.cargo/mutants.toml`](.cargo/mutants.toml)
>   `exclude_re` block.
>
> This plan doc captures the **rollout state and open work** — the
> sequencing checklist (§10), risks (§8), and option-3 deferral
> rationale (§7).  When Phase 2 closes, this plan archives into
> `plans/DONE.md` with a short retro and the live references in
> `Cargo.toml` / `.cargo/mutants.toml` / `plans/TEST.md` /
> `docs/src/dev/mutation-testing.md` flip to point at the docs page
> only.

---

## 0. TL;DR for reviewers

Two layered regimes, in this order:

1. **Phase 1 — Incremental critical-path sweeps (manual, file-by-file).**
   Eight files, ordered by signal-per-mutant, each closed before the next
   begins. Goal: zero unexempted survivors per file. New tests written as
   genuine survivors are found; cosmetic / equivalent mutants exempted via
   `.cargo/mutants.toml` with one-line rationale comments. Estimated total
   wallclock across local + CI runs: ~10 days elapsed, ~20 hours
   developer-driven.

2. **Phase 2 — Sharded weekly full sweep (CI-driven, hands-off).**
   Eight-shard parallel matrix triggered on weekday cron. Failures open
   auto-issues; triage SLA 7 days. Lands only after Phase 1 step 1.4 is
   green so the config is stable enough that the weekly sweep is signal-bearing,
   not noise-bearing.

Phase 1 surfaces real coverage gaps fast; Phase 2 keeps them closed.

The plan deliberately does **not** mass-suppress mutants, set a kill-rate
quota, mutate I/O-driven code on the critical path, or extend cargo-mutants
to other languages. See §9.

---

## 1. Goal & rationale

Detect under-tested code on `tensogram`'s **critical paths** — the
modules where a missed mutation translates to silent data corruption or
cross-language wire-format breakage. The codebase has a small number of
load-bearing files (wire layout, framing, hashing, dtype, CBOR canonical
encoding) that are heavily exercised by golden-file tests but have not
been measured against mutation testing.

Why now:

- PR #111 (just merged) added a new per-frame `HASH_PRESENT` flag bit
  and the `MissingHash` error variant. Bit-flag handling is the exact
  mutation class cargo-mutants catches well (`&` ↔ `|` ↔ `^`,
  `<` ↔ `<=` on offset arithmetic). Establishing a baseline now means
  future flag additions arrive with the same testing rigor.
- The Pass-7 frame-bound bug Copilot caught (`verify_data_object_frames`
  walking past `Preamble.total_length`) is a textbook off-by-one — the
  exact shape of mutation cargo-mutants generates by default. We want
  a regression-prevention regime for that class.
- Test suite is already dense (1500+ Rust tests, 500+ Python, 415 TS,
  157 C++) — cargo-mutants is well-positioned to surface the *small
  number* of remaining gaps, not redo work the suite already does.

Why **not** more:

- This plan is about *measurement of existing tests*, not adding new
  test categories. Property-based testing (proptest), fuzzing
  (cargo-fuzz), and differential testing across language bindings are
  separate initiatives that complement mutation testing but live
  outside this plan's scope.

Documentation strategy: this rollout plan
(`MUTATION_TESTING.md`) is the *operational* doc. It is paired with
two derivative artifacts so the regime is visible to humans and AI
agents alike — see §6.

---

## 2. Out of scope

Not mutated in either phase:

| Path | Why skipped |
|---|---|
| `rust/tensogram/src/remote.rs` (~784 mutants) | I/O-driven; `object_store` mocks; mutants would mostly time out without producing actionable signal. Re-evaluate after Phase 2 stabilises. |
| `rust/tensogram/src/doctor/**` | Diagnostic-only print formatting; not a correctness concern. |
| `rust/tensogram-cli/**` | `clap`-driven argument parsing; mutants are noise. |
| `rust/benchmarks/**` | Performance code, not correctness. |
| `rust/tensogram-rust-examples/**` | Illustrative; tests would be tutorial-quality, not coverage. |
| `rust/tensogram-sz3-sys/**` | FFI shim with no public surface; mutate `tensogram-sz3` instead. |
| Other-language bindings (`python/`, `cpp/`, `typescript/`, `rust/tensogram-wasm/`) | cargo-mutants is Rust-only. Wire-format parity is enforced by golden files. |

These exclusions are pinned in the committed `.cargo/mutants.toml` so
sharded weekly runs cannot accidentally pull them in.

---

## 3. Phase 1 — Incremental critical-path sweeps

### 3.0 Tunables: concurrency and resume

#### 3.0.1 Concurrency — seven nested layers

cargo-mutants concurrency is **not** controlled by a single number.
**Seven** independent layers each fan out by default to one thread per
logical CPU; multiplied together they saturate a 12-core laptop and
trigger peak-power events even at `CARGO_MUTANTS_JOBS=2`.  All seven
must be clamped:

| # | Layer | Knob | Default | This plan's value | Where |
|---|---|---|---|---|---|
| 1 | Mutant workers | `CARGO_MUTANTS_JOBS` env (= `--jobs`) | NCPUS | `2` | Canonical invocation |
| 2 | cargo + rustc *invocation* count | `--jobserver-tasks` flag | NCPUS | `2` | Canonical invocation |
| 3 | rustc *codegen* threads inside each rustc | `[profile].codegen-units` | **16** in release | `1` | `[profile.release-mutants]` in `Cargo.toml` |
| 4 | Test-binary parallelism | `cargo test -- --test-threads=N` | NCPUS | `2` | `.cargo/mutants.toml` `additional_cargo_test_args` |
| 5 | cmake build.rs scripts (libaec-sys, blosc2-sys, zfp-sys-cc, tensogram-sz3-sys) | `CMAKE_BUILD_PARALLEL_LEVEL` env | NCPUS | `1` | Canonical invocation |
| 6 | nested make invocations from cmake | `MAKEFLAGS=-j1` env | NCPUS | `-j1` | Canonical invocation |
| 7 | tensogram-internal rayon (axis-A / axis-B pipeline) | `TENSOGRAM_THREADS` env | NCPUS | `1` | Canonical invocation |

**Layer 3 is the killer.**  Without `codegen-units = 1`, each rustc
invocation in release mode spawns 16 codegen threads — and cargo's
jobserver counts the rustc *process* as one slot, not its 16
internal threads.  At `--jobserver-tasks 2` that's 32 simultaneous
codegen threads during peak build, drawing >120W on M-series
silicon for 5–15-second bursts.  The 1-minute load average smooths
those bursts away (steady ~4) but the laptop's power supply
trips during the burst.  This is what caused two observed
mid-sweep power-downs during this rollout despite the four-clamp
config originally landed.

**Canonical local invocation** (use this exact form for every Phase 1
sweep):

```bash
CARGO_MUTANTS_JOBS=2 \
TENSOGRAM_THREADS=1 \
CMAKE_BUILD_PARALLEL_LEVEL=1 \
MAKEFLAGS=-j1 \
cargo mutants -p tensogram --file <path> --jobserver-tasks 2
```

`--test-threads=2` and `--profile release-mutants` are already pinned
in `.cargo/mutants.toml`, so they don't need to appear on every
command line.

| Context | `CARGO_MUTANTS_JOBS` | `--jobserver-tasks` | `TENSOGRAM_THREADS` | `CMAKE_BUILD_PARALLEL_LEVEL` | `MAKEFLAGS` |
|---|---|---|---|---|---|
| Local development | `2` | `2` | `1` | `1` | `-j1` |
| PR-time CI (`mutants-diff`) | `2` | `2` | `1` | `1` | `-j1` |
| Weekend CI shard | `2` | `2` | `1` | `1` | `-j1` |

#### 3.0.1.1 macOS escape hatch — `taskpolicy -b`

On Apple Silicon laptops, layered application-level clamps can still
miss something (an undocumented threadpool in a dependency, a future
crate that fans out without env-var support, etc.).  For unattended
runs where a power event is unacceptable, prefix the canonical
invocation with `taskpolicy -b`:

```bash
taskpolicy -b -- env \
  CARGO_MUTANTS_JOBS=2 \
  TENSOGRAM_THREADS=1 \
  CMAKE_BUILD_PARALLEL_LEVEL=1 \
  MAKEFLAGS=-j1 \
  cargo mutants -p tensogram --file <path> --jobserver-tasks 2
```

`taskpolicy -b` runs the entire process tree under **background QoS**:

- Restricted to E-cores only (the 4 efficiency cores out of 12 on
  M-series).
- Total package power capped at ~30% of system maximum.
- Applies to all child processes — rustc, cc-rs, cmake, make, ld64,
  rayon, zstdmt workers, blosc2 workers — without their cooperation.

It's kernel-level enforcement, not application-level.  No matter how
many threads anything fans out to, they all compete for E-core
time-slices.

**Cost**: 2-3x wallclock (E-cores are slower than P-cores).
framing.rs sweep goes from ~12hr → ~30hr.  Acceptable for unattended
runs; unsuitable for interactive iteration.

To override jobs/jobserver on a beefier machine without taskpolicy:

```bash
# Workstation with ≥16 cores and good cooling
CARGO_MUTANTS_JOBS=4 TENSOGRAM_THREADS=1 \
CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 \
  cargo mutants -p tensogram --file rust/tensogram/src/hash.rs --jobserver-tasks 4

# Strictly serial — safest for unattended overnight runs without taskpolicy
CARGO_MUTANTS_JOBS=1 TENSOGRAM_THREADS=1 \
CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 \
  cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --jobserver-tasks 1
```

#### 3.0.1.2 Why not just lower `CARGO_MUTANTS_JOBS`?

Because the layers *multiply*, not *add*.  Halving jobs from 2 → 1
only divides one of seven nested fan-outs.  All seven clamps
together (or `taskpolicy -b` as the OS-level circuit-breaker) is
what brings load profile down to a sustainable level on a 12-core
laptop.

#### 3.0.2 Resuming an interrupted sweep

Long sweeps (notably step 1.8 `framing.rs` at ~551 mutants) can be
interrupted by power events, OOM kills, or laptop sleep.  cargo-mutants
does not have a native resume flag, but `--shard N/K` partitions the
mutant list deterministically and lets you re-run a subset:

```bash
# Run quarter 1 of 4 — first ~138 mutants of framing.rs
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 1/4

# Then quarter 2, 3, 4 in subsequent sessions
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 2/4
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 3/4
cargo mutants -p tensogram --file rust/tensogram/src/framing.rs --shard 4/4
```

Shard assignment is stable across runs (cargo-mutants hashes mutant
identity), so re-running a single shard re-tests the same mutants.
Note: `mutants.out/` is overwritten per invocation; copy or rename
between shard runs if you want to preserve per-shard results.

For step 1.8 specifically the plan calls for **3 sub-runs** as a
safety budget for interrupts — split as `--shard 1/3`, `2/3`, `3/3`.

### 3.1 Priority order (8 files, sequential)

Order is signal-per-mutant: small files with high consequence first.
Each step closes before the next begins.

| Step | File | Mutants (estimated) | Wallclock target (release, `CARGO_MUTANTS_JOBS=2`) | Why first |
|---|---|---|---|---|
| 1.1 | `rust/tensogram/src/hash.rs` | 32 | ~45 min | xxh3 verification, just extended in PR #111 (`HASH_PRESENT`, `check_frame_hash`, `verify_frame_hash`). Tiny, fast, highest signal. |
| 1.2 | `rust/tensogram/src/error.rs` | ~15 | ~20 min | Newly augmented `MissingHash` + `HashMismatch { object_index }`. Cheap; ensures the new error variants are exercised beyond compile-checking. |
| 1.3 | `rust/tensogram/src/wire.rs` | 68 | ~90 min | Preamble/postamble/frame-header layout + flag bits. The `HASH_PRESENT = 1 << 1` bit-position must be locked in by golden tests. |
| 1.4 | `rust/tensogram/src/dtype.rs` | ~25 | ~50 min | `swap_unit_size()` for complex types (recently added per `plans/TODO.md` *caller-endianess*). Critical for round-trip equality. |
| 1.5 | `rust/tensogram/src/metadata.rs` | 104 | ~150 min | Canonical CBOR ordering — a mutated `<` → `<=` in key sort would silently break cross-language byte-equality goldens. |
| 1.6 | `rust/tensogram/src/validate/integrity.rs` | 61 | ~80 min | Level-3 hash validation, recently rewritten for `object_index` reporting. |
| 1.7 | `rust/tensogram/src/decode.rs` | 83 | ~120 min | The verify-first pre-pass added in PR #111. Pass-7's bound fix (`frame_end <= msg_end`) should resist the `<=` → `<` mutation. |
| 1.8 | `rust/tensogram/src/framing.rs` | 551 | ~12 hr (3 sub-runs at `--shard 1/3`/`2/3`/`3/3`) | Frame ordering enforcement, scan recovery, `decode_message`. The biggest one; split with `--shard` (§3.0.2) to bound interrupt blast radius. |

Total Phase 1 mutants: ~940 across 8 files. Phase 1 deliberately stops
short of `encode.rs` (216 mutants) and `streaming.rs` (76) — those
become the first beneficiaries of Phase 2's weekly sweep when it lands.

### 3.2 Per-step protocol

For each step:

1. Run the sweep with all seven concurrency clamps from §3.0.1:
   ```bash
   CARGO_MUTANTS_JOBS=2 \
   TENSOGRAM_THREADS=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=1 \
   MAKEFLAGS=-j1 \
     cargo mutants -p tensogram --file <path> --jobserver-tasks 2
   ```
   On macOS laptops, prefix with `taskpolicy -b -- env` for
   power-supply safety (§3.0.1.1).  For unattended overnight runs
   without taskpolicy, drop jobs and jobserver-tasks to 1.  For a
   workstation with ≥16 cores, raise jobs / jobserver-tasks /
   `MAKEFLAGS` together (always in lock-step).
2. Read `mutants.out/missed.txt`. For each survivor:
   - **Genuine coverage gap** → write a test, re-run the sweep,
     iterate until killed.
   - **Equivalent / cosmetic mutation** (e.g. trace-field renames,
     `Display` formatting strings, semantically-identical refactors)
     → add an `exclude_re` entry to `.cargo/mutants.toml` with a
     **one-line rationale comment** above the regex. Mass suppression
     by category is forbidden — every entry is explicit.
3. Commit the step's changes (test additions + config exemptions) in
   a single PR with title `test(mutants): cover <module>` (or
   `chore(mutants): exempt cosmetic patterns in <module>` if the
   change is pure exemption — flag for extra scrutiny).
4. Step closes when next sweep on that file reports 0 unexempted
   survivors. Update the sequencing checklist (§10).
5. Advance to the next step.

### 3.3 Initial `.cargo/mutants.toml`

Land **before** step 1.1 begins. Source of truth for skip-globs and
test-runner config.

```toml
# .cargo/mutants.toml
#
# See MUTATION_TESTING.md for the rollout plan.
# See https://mutants.rs/ for cargo-mutants usage.

# Per-mutant test-time bound. tensogram's release-mode test loop runs
# in ~41s; multiplier 3.0 gives a 2-minute kill window per mutant
# before declaring timeout.
minimum_test_timeout = 30
timeout_multiplier = 3.0

# Files / paths excluded from both phases.
exclude_globs = [
    "rust/tensogram/src/remote.rs",          # I/O-driven; revisit post-Phase-2
    "rust/tensogram/src/doctor/**",          # diagnostic prints; not correctness
    "rust/tensogram-cli/**",                 # clap-driven CLI; noise
    "rust/tensogram-rust-examples/**",       # illustrative; not library code
    "rust/benchmarks/**",                    # perf only
    "rust/tensogram-sz3-sys/**",             # FFI shim; mutate tensogram-sz3
    "rust/tensogram/src/lib.rs",             # re-exports only
    "**/build.rs",                           # build scripts
]

# Per-mutation regex exemptions. Each entry MUST have a one-line
# rationale comment explaining why the mutation is equivalent or
# cosmetic. Entries are added by Phase-1 PRs as survivors are
# triaged; never mass-suppressed by category.
exclude_re = [
    # (entries land per-PR during Phase 1)
]

# Run cargo test in release mode — ~8x faster per iteration on this
# codebase (release ~41s vs debug ~6 min for the tensogram crate).
additional_cargo_test_args = ["--release"]
```

### 3.4 PR-time `--in-diff` integration

Wire after step 1.4 is green (config has stabilised on hash + error +
wire + dtype). Initially **non-blocking** (annotations only); flip to
required-for-merge after two clean weeks.

**Runs on ECMWF self-hosted infrastructure** (matches the rest of the
test matrix in `ci.yml`) — *not* GitHub-hosted runners.  Rationale:
GitHub-Actions minutes are scarce; ECMWF maintains its own
`platform-builder-docker-xl` fleet with the project's CI container
(`eccr.ecmwf.int/tensogram/ci:1.3.0`) pre-baked with all native
build dependencies (libaec, libnetcdf, eccodes, blosc2, zfp, sz3).

`.github/workflows/ci.yml` addition:

```yaml
mutants-diff:
  name: Mutants (diff)
  runs-on: [self-hosted, Linux, platform-builder-docker-xl, platform-builder-Ubuntu-22.04]
  container:
    image: eccr.ecmwf.int/tensogram/ci:1.3.0
    credentials:
      username: ${{ secrets.ECMWF_DOCKER_REGISTRY_USERNAME }}
      password: ${{ secrets.ECMWF_DOCKER_REGISTRY_ACCESS_TOKEN }}
  if: github.event_name == 'pull_request'
  timeout-minutes: 30
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # need base branch for --in-diff
    - run: mkdir -p "$TMPDIR"
    - run: cargo install cargo-mutants --version 27.0.0 --locked
    - name: Mutate diff vs base
      env:
        CARGO_MUTANTS_JOBS: 2          # mutant worker count — see §3.0.1
        TENSOGRAM_THREADS: 1           # cap library-internal rayon
        CMAKE_BUILD_PARALLEL_LEVEL: 1  # cmake-driven build.rs in deps
        MAKEFLAGS: -j1                 # nested make invocations from cmake
      run: |
        cargo mutants \
          --in-diff origin/${{ github.base_ref }}..HEAD \
          --no-shuffle \
          --jobserver-tasks 2 \
          --timeout-multiplier 3.0
    - uses: actions/upload-artifact@v4
      if: always()
      with:
        name: mutants-diff-output
        path: mutants.out/
```

Why non-blocking first:

- New PRs surface unexpected mutant patterns; flipping straight to
  required would block merges on equivalent-mutant noise that takes
  a few iterations to exempt cleanly.
- Two-week observation gives a chance to tune `exclude_re` without
  pressure.

---

## 4. Phase 2 — Sharded weekly full sweep

### 4.1 Trigger and matrix

**Runs on ECMWF self-hosted infrastructure**, not GitHub-hosted runners.
GitHub-Actions minutes are scarce; ECMWF maintains its own
`platform-builder-docker-xl` fleet with the project's CI container
(`eccr.ecmwf.int/tensogram/ci:1.3.0`) pre-baked with all native build
dependencies (libaec, libnetcdf, eccodes, blosc2, zfp, sz3).

**Weekend daytime schedule, not nightly.** Saturday + Sunday at
09:00 UTC (10:00-11:00 CET/CEST). Two runs per weekend so the
Saturday run picks up Friday commits and the Sunday run picks up
Saturday commits. Daytime weekend = minimal contention on the
shared builder fleet — we are not competing with weekday-active
developer CI.

**16 shards × `max-parallel: 2`** → typical wallclock per shard
~30 minutes on ECMWF infra (see §4.4 for the math), 16 / 2 = 8
batches of ~30 min each = ~4 hr total weekend wallclock. Safety
cap at 8 hr per shard via GH-Actions `timeout-minutes: 480`, and a
`timeout 7h` wrapper around `cargo mutants` itself so the artifact
upload step has a 1-hr grace window when the inner timeout fires.

`.github/workflows/mutants-weekly.yml` (committed in this branch):

See `.github/workflows/mutants-weekly.yml` for the committed form.
The salient elements:

```yaml
name: mutants-weekly
on:
  schedule:
    # Saturday + Sunday 09:00 UTC = weekend daytime
    - cron: "0 9 * * 6,0"
  workflow_dispatch:
    inputs:
      shards:
        description: "Number of shards (default 16)"
        required: false
        default: "16"

concurrency:
  group: mutants-weekly
  cancel-in-progress: false  # queue, don't kill a running sweep

jobs:
  full-sweep:
    runs-on:
      [self-hosted, Linux, platform-builder-docker-xl, platform-builder-Ubuntu-22.04]
    container:
      image: eccr.ecmwf.int/tensogram/ci:1.3.0
      credentials:
        username: ${{ secrets.ECMWF_DOCKER_REGISTRY_USERNAME }}
        password: ${{ secrets.ECMWF_DOCKER_REGISTRY_ACCESS_TOKEN }}
    timeout-minutes: 480  # 8-hour absolute hard cap per shard
    strategy:
      fail-fast: false
      max-parallel: 2  # polite tenant on the shared builder fleet
      matrix:
        shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    steps:
      - uses: actions/checkout@v4
      - run: cargo install cargo-mutants --version 27.0.0 --locked
      - name: Run shard
        env:
          CARGO_MUTANTS_JOBS: 2
          TENSOGRAM_THREADS: 1
          CMAKE_BUILD_PARALLEL_LEVEL: 1
          MAKEFLAGS: -j1
        run: |
          # `timeout 7h` SIGTERMs cargo mutants 1 hr before the GH-Actions
          # 8-hour kill, leaving the artifact-upload step time to run.
          timeout --signal=TERM --kill-after=60 7h \
            cargo mutants --shard ${{ matrix.shard }}/16 \
              --no-shuffle --jobserver-tasks 2 --timeout-multiplier 3.0 \
            || rc=$?
          # Treat 124/137 (timed out) as "partial results, don't fail matrix"
          if [ "${rc:-0}" = "124" ] || [ "${rc:-0}" = "137" ]; then
            echo "::warning::shard ${{ matrix.shard }}/16 hit 7-hour timeout"
            exit 0
          fi
          exit "${rc:-0}"
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mutants-shard-${{ matrix.shard }}
          path: mutants.out/
```

The collate + open-issue-on-fail jobs that follow are detailed in
`mutants-weekly.yml`.  Key behaviours:

- **Collate writes `$GITHUB_STEP_SUMMARY`** with caught/missed/
  timeout/unviable totals + the first 50 missed mutants — visible
  directly on the workflow run page, no artifact download needed
  for triage.
- **Auto-issue body** has a structured header (commit SHA, date,
  status table, SLA reminder) before the raw `missed.txt`/`timeout.txt`
  dump, plus a link to the workflow-run artifact for full details.
- **`open-issue-on-fail` runs on `ubuntu-latest`** (one short
  GitHub API call) — the only GH-Actions-minute cost in the whole
  workflow, ~10 sec per failed weekend.

### 4.2 Shard mechanics

- `cargo mutants --shard N/16` partitions the global mutant list
  deterministically (cargo-mutants hashes mutant identity for stable
  assignment).  No manual file slicing required; adding a new file
  rebalances naturally.  Re-running a shard re-tests the same
  mutants — handy when one shard times out.
- **Per-shard wallclock estimate (16 shards on ECMWF infra):**
  baseline build (full workspace compile in fresh tmpdir) ~3 min,
  per-mutant ~30s on average, with `--jobserver-tasks 2` running
  two mutants concurrently within a shard.  ~118 mutants/shard
  ÷ 2-way parallel × ~30 s/mutant + 3 min baseline ≈ ~33 min
  typical, ~60 min worst-case.  At `max-parallel: 2` overall
  weekend wallclock is 16/2 × ~33 min ≈ 4 hours.
- **Why 16 shards** (not 8 from the earlier draft): finer sharding
  shrinks per-shard wallclock so the 8-hour `timeout-minutes`
  safety cap is comfortably above the 95th-percentile shard time.
  Cost is one extra baseline build per shard (~3 min × 8 extra
  shards = ~24 min cumulative overhead, negligible relative to
  the ~4 hr wallclock).
- **Configurable** via `workflow_dispatch` input — for ad-hoc
  smaller runs (e.g. a quick smoke after a config bump), drop to
  4 or 2 shards.  The matrix entry list is fixed at 16 in the
  YAML; the `shards` input is documentary today and would require
  a refactor to actually take effect.  See "Future improvements"
  in §4.4.

### 4.3 Failure detection — three independent signals

A failed weekly run surfaces through **three independent channels**
so a quiet email client / disabled notifications cannot hide it.

1. **GitHub Issue auto-created** (`open-issue-on-fail` job).
   - Title: `mutants-weekly: <N> surviving mutants on <sha7> (YYYY-MM-DD)`
   - Labels: `mutation-testing`, `triage`
   - Body has a structured header (status table + SLA reminder)
     followed by `missed.txt` and `timeout.txt` dumps + a link to
     the workflow-run artifact.
   - Repo watchers with "All Activity" or "Issues" notifications
     receive an email.

2. **`$GITHUB_STEP_SUMMARY`** on the workflow run page.
   - At-a-glance status table directly on the run page — no
     artifact download needed.
   - Includes per-status totals (caught / missed / timeout / unviable)
     + first 50 surviving mutants inline.
   - Visible in the GitHub Actions UI under "Summary".

3. **Red badge on the Actions tab + commit status.**
   - Workflow run shows red.
   - Watching the repo's "Actions" or having a CI-status app
     (e.g. GitHub Mobile, GitHub for Slack) surfaces it visually.

### 4.4 Triage policy

- **Surviving mutants → 7-day SLA.**  Each issue closes via
  either:
  - Test addition (preferred), or
  - `exclude_re` PR with comment explaining why the mutation is
    equivalent / cosmetic.
- **3 consecutive red weekends without progress** → escalate.
  Investigate whether the failure is a real coverage gap, a
  test-infrastructure regression (e.g. `cargo install`
  flakiness), or a false-positive class (then tighten `exclude_re`).
- **Release-blocking?** No.  The `release-preflight` workflow
  does not depend on `mutants-weekly`.  Mutation testing is a
  quality metric, not a release gate.

### 4.5 Performance budget

| Resource | Budget | Margin |
|---|---|---|
| Wallclock per shard | 8 hr hard cap (GH-Actions `timeout-minutes: 480`); `timeout 7h` on the cargo-mutants invocation gives a 1-hr grace before the GH kill so artifact upload still runs | typical ~30 min, 95th-percentile ~60 min — the cap is belt-and-braces |
| Total weekend wallclock | 16 shards / `max-parallel: 2` × ~30 min ≈ ~4 hr × 2 weekend days = ~8 hr | weekend daytime window is ~16 hr (09:00 → midnight UTC); 50% margin |
| ECMWF builder occupancy | ~2 builders × ~4 hr per weekend day = ~16 builder-hr/week | very polite tenant |
| GitHub-Actions minutes consumed | only `open-issue-on-fail` (~10 sec per failed weekend) | negligible |
| Artifact storage | 16 shards × ~10 MB = ~160 MB per weekend × 2 weekend days × retain 30 days = ~10 GB rolling | within org artifact storage |
| Issue noise | 1 auto-issue per failed weekend run, deduped by date | acceptable |

### 4.6 Future improvements (not blockers for landing)

- **Make shard count configurable for real**: today the
  `workflow_dispatch.inputs.shards` is documentary; the matrix
  entries are hardcoded `[1..16]/16`.  A small Action like
  `setup-matrix` could expand the matrix dynamically based on
  the input.  Useful when adding ~2000 mutants from a new module
  pushes the per-shard wallclock past comfort.
- **Slack / Teams webhook**: an additional channel beyond email
  + GitHub-Issues notifications.  Trivial to add as another step
  in `open-issue-on-fail`.
- **Resume tracking**: cargo-mutants does not have native resume,
  but a wrapper script could record the previous-run's `mutants.out/`
  and compare deltas.  Useful if shards regularly time out.
  Skip until measurement shows we need it.

---

## 5. Configuration evolution

`.cargo/mutants.toml` is a living document. Lifecycle:

1. **Phase 1.1 birth.** Land the initial config (§3.3) in the same
   PR as the first `hash.rs` sweep.
2. **Per-file exemptions** added during Phase 1 PRs. Each entry has
   a one-line comment; no exemption lands without rationale.
3. **Quarterly review.** Drop exemptions that no longer apply
   (file deleted, mutation no longer generated by tooling upgrade,
   refactor obsoleted the pattern).
4. **cargo-mutants version bumps.** Pinned in CI via
   `--version 27.0.0`. Bumps land in dedicated PRs that re-run
   Phase-1 step 1.1 (`hash.rs`) as a smoke test for new mutant
   classes; if no surprises, the bump merges. If new equivalent-
   mutant classes appear, the PR also updates `exclude_re`.

---

## 6. Documentation deliverables

Two audiences, two artifacts, both landing during Phase 1 — before
the `mutants-diff` workflow flips to required-for-merge, so
contributors hitting the new gate find the docs on the first try.

### 6.1 User & contributor docs (mdbook)

A new mdbook page at **`docs/src/dev/mutation-testing.md`**, linked
from `docs/src/SUMMARY.md` under the existing **Reference** section
(between *Internals* and *Edge Cases*). If proptest / fuzzing pages
land later, this can be promoted into a dedicated *Quality* section;
for now a single page under *Reference* avoids creating a one-item
section.

Page contents, in order:

1. **What mutation testing is** — one paragraph accessible to
   non-experts: "we systematically corrupt the source code and
   confirm tests catch the corruption; survivors mark untested
   behaviour."
2. **Why tensogram uses it** — condensed rationale from §1: bit-flag
   handling, off-by-one frame-bound bugs, wire-format byte-equality
   concerns. Two paragraphs, with a pointer back to this rollout
   plan for the long form.
3. **The two regimes** —
   - PR-time `--in-diff` workflow (link to
     `.github/workflows/ci.yml`),
   - weekly sharded sweep (link to
     `.github/workflows/mutants-weekly.yml`).
4. **Running mutation testing locally** — concrete commands.
   Concurrency is controlled by the `CARGO_MUTANTS_JOBS` environment
   variable (default `2` — see this plan's §3.0.1):
   ```bash
   cargo install cargo-mutants --version 27.0.0 --locked
   # Default concurrency (2 jobs)
   cargo mutants -p tensogram --file rust/tensogram/src/hash.rs
   # Override for serial / faster runs
   CARGO_MUTANTS_JOBS=1 cargo mutants -p tensogram --file rust/tensogram/src/hash.rs
   ```
   Plus the diff-only invocation contributors will use most:
   ```bash
   cargo mutants --in-diff origin/main..HEAD
   ```
   For interrupted long sweeps, see §3.0.2 (`--shard N/K` resume).
5. **Reading `mutants.out/`** — short tour of `outcomes.json`,
   `missed.txt`, `caught.txt`, `timeout.txt`, `unviable.txt`.
   What each file means; when to investigate vs. ignore.
6. **When to add a test vs. an exemption** — decision tree:
   - Survivor changes observable behaviour → write a test.
   - Survivor only changes log / `Display` / trace strings → exempt
     with a one-line rationale comment in `.cargo/mutants.toml`.
   - Survivor is genuinely equivalent (e.g. `+ 0` is dead code) →
     either delete the dead code or exempt.
   - Two strict rules: **no mass suppression by category**;
     **every `exclude_re` entry needs a one-line rationale comment**
     above it.
7. **Bumping cargo-mutants** — protocol from §5: version-pinned in
   CI; bumps land in dedicated PRs that re-run step 1.1 as a smoke
   test.
8. **References** — links to:
   - `MUTATION_TESTING.md` (this rollout plan, repo root),
   - `plans/TEST.md` (canonical test-shape doc — see §6.2 below),
   - <https://mutants.rs/> (upstream documentation).

`CONTRIBUTING.md` gains a single-row addition to its existing
*Test Structure* table (around line 137) so first-time contributors
discover mutation testing alongside unit / integration / property
tests:

```markdown
| Mutation testing | `.cargo/mutants.toml` + `.github/workflows/mutants-weekly.yml` | Measure test depth on critical-path modules — see [docs/src/dev/mutation-testing.md](docs/src/dev/mutation-testing.md) |
```

### 6.2 AI Code Agent context (`plans/TEST.md`)

`plans/TEST.md` is the canonical *test-shape* document for AI Code
Agent sessions: agents read it on cold-start to understand where
tests live and what each covers. Mutation testing must appear there
or future agent sessions will not know about the regime — they will
add tests without checking whether mutants survive, and miss the
PR-time `--in-diff` gate.

Updates land in the same PR as `docs/src/dev/mutation-testing.md`:

1. **New section** after *Coverage shape*, titled `## Mutation
   testing`, covering:
   - Tool: `cargo-mutants` (pinned version).
   - Two regimes: PR-time `--in-diff`, weekly sharded full sweep.
   - Config lives in `.cargo/mutants.toml`; rollout plan in
     `MUTATION_TESTING.md`; user-facing docs in
     `docs/src/dev/mutation-testing.md`.
   - Critical-path priority order — the 8-file table from §3.1
     of this plan, condensed to file + one-line rationale per row.
   - **One-line decision rule** for agents: *if the change touches
     one of the eight critical-path modules, the agent must run
     `cargo mutants --in-diff` and triage survivors before declaring
     the feature covered.*

2. **Coverage-shape table addendum**: a new closing row clarifying
   that mutation testing is a *measurement layer over* the existing
   suite, not an additional source of tests:

   ```markdown
   | Mutation testing | `.cargo/mutants.toml`, `.github/workflows/mutants-weekly.yml` | Measurement of critical-path test depth via cargo-mutants — see `MUTATION_TESTING.md` |
   ```

3. **Edge-cases pointer**: append to the *Edge cases* list:

   > - Per-frame `HASH_PRESENT` flag-bit handling: bitwise operator
   >   flips (`&` ↔ `\|` ↔ `^`) on flag checks must be killed by
   >   the cell-C / cell-D / cell-F matrix tests in every binding.

   This pins a specific mutation class to a specific test, so a
   future agent triaging mutants in `hash.rs` or
   `validate/integrity.rs` finds the existing coverage by name.

### 6.3 Doc-doc consistency

The three docs (`MUTATION_TESTING.md`, `docs/src/dev/mutation-testing.md`,
`plans/TEST.md`) form a small triangle:

- **`MUTATION_TESTING.md`** is the *plan and rollout state*. Lives
  at repo root next to `PLAN_*.md` files. Uncommitted-by-convention
  during active rollout; archived to `plans/DONE.md` on close.
- **`docs/src/dev/mutation-testing.md`** is the *user-facing
  reference*. Committed; published to
  `https://sites.ecmwf.int/docs/tensogram/`. Stable post-Phase-1.
- **`plans/TEST.md`** is the *AI agent cold-start context*.
  Committed; small mutation-testing section pointing at the other
  two.

Cross-links are bidirectional. When any of the three changes
materially, the other two are reviewed in the same PR.

---

## 7. Acceptance criteria

### Phase 1 done when:

- All 8 critical-path files have completed a sweep with **0
  unexempted survivors**, **OR** explicitly deferred to Phase 2's
  weekly machinery (recorded with rationale here).
- `.cargo/mutants.toml` is committed and contains rationale comments
  on every `exclude_re` entry.
- `mutants-diff` workflow is in `.github/workflows/ci.yml` and
  marked **required-for-merge**.
- **Documentation triangle complete (§6)**:
  - `docs/src/dev/mutation-testing.md` published and linked from
    `docs/src/SUMMARY.md` under *Reference*.
  - `CONTRIBUTING.md` *Test Structure* table mentions mutation
    testing with a link to the docs page.
  - `plans/TEST.md` has the new *Mutation testing* section, the
    coverage-shape addendum row, and the *Edge cases* pointer.
- Audit trail of new tests is browseable via
  `git log --grep "test(mutants)" --oneline`.

#### Phase 1 deferral record (option 3, recorded during this rollout)

**Step 1.8 (`framing.rs`, ~551 mutants)** is deferred to Phase 2's
weekly machinery rather than completed locally as a Phase 1 sweep.

Rationale: two repeated power-down events on the dev laptop while
running shard 1/3 of `framing.rs` despite the seven concurrency
clamps from §3.0.1 and the `taskpolicy -b` macOS escape hatch from
§3.0.1.1.  The base wallclock for the file is on the order of
~12 hours under the laptop's clamped configuration, multiplied
~2-3× by `taskpolicy -b` background-QoS throttling — too long for
comfort given the observed power instability.

framing.rs is included in the workspace-wide weekly sweep with
**no special handling** — `cargo mutants` discovers all its mutants
through the same `--shard N/8` partitioning as every other file
not on the §2 exclude list.  The weekly run produces the same
test/exemption signal Phase 1 would have produced, just spread
across more nights instead of one long laptop session.

The 42 survivor + 2 timeout snapshot from the partial shard 1/3
sweep is **not preserved** as committed state — the full weekly
sweep will re-discover them deterministically and triage will
happen via the auto-issue loop (§4.3).

This deferral is in scope of Phase 1 acceptance — the file is not
"unsept", it is "swept by the Phase 2 backstop instead of by Phase 1
local execution".

### Phase 2 done when:

- `mutants-weekly` has run **green for 2 consecutive weeks** on
  `main`, including framing.rs (the deferred file from Phase 1
  step 1.8).
- At least one auto-issue has been triaged through to closure
  (proves the issue-creation + SLA loop works).
- `docs/src/dev/mutation-testing.md` updated with a "Lessons
  learned" section summarising the most common survivor classes
  observed during stabilisation (so future contributors know what
  patterns to watch for).

---

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `framing.rs` step (1.8) blows up runtime budget | High | Stalls Phase 1 for 1-2 days | Pre-shard step 1.8 manually into 3 sub-runs (header / body / footer phases); execute on local hardware overnight, not CI. |
| `--in-diff` flakiness on force-pushed PRs | Medium | Spurious red builds | Pin `fetch-depth: 0`; use `origin/${{ github.base_ref }}..HEAD`. Job is non-blocking initially; only flip to required after stability proven. |
| Equivalent-mutant fatigue | Medium | Reviewer burnout, exemption-list bloat | Strict per-PR review; one rejection of a too-broad regex per PR keeps the bar high. Quarterly exemption-list audit (§5). |
| Weekly run exhausts ECMWF builder fleet capacity | Low | Other CI jobs queue behind | 16 shards × `max-parallel: 2` × ~30 min = ~4 hr per weekend day, deliberately scheduled for low-contention weekend daytime; reduce to single weekend day or fewer shards if pressure appears. |
| cargo-mutants major version upgrade changes mutation set silently | Medium | Phase-1-clean files reappear as red overnight | Pin version explicitly. Bumps land in dedicated PRs (§5.4). |
| Mutation testing surfaces a genuine bug in shipped code | Low | Patch release needed | Treat as any other security/correctness fix: PR + CHANGELOG + version bump. Does not invalidate the plan; validates it. |

---

## 9. What this plan deliberately does NOT do

- **Add property-based testing or fuzzing.** Different tools for
  different failure modes. See `plans/IDEAS.md` for fuzz seeds;
  `tensogram-szip`'s existing `proptest_roundtrip.rs` is the
  template if/when proptest expansion lands as its own initiative.
- **Mutate other-language bindings.** WASM/Python/C++/TS have
  their own coverage stories — golden files for byte parity,
  language-native test suites for behaviour. cargo-mutants is
  Rust-only; we don't extend it artificially.
- **Set hard kill-rate quotas** (e.g. "must reach 95%"). Quotas
  drive busywork on equivalent mutants. The bar is *every survivor
  is justified* — qualitative, not quantitative.
- **Mutate I/O, CLI, or example code in Phase 1.** These are
  excluded by the initial `exclude_globs`. Phase 2 may pull some
  of them in opportunistically (notably encode/streaming) once the
  config has stabilised.
- **Backport the regime to historical PRs.** Mutation testing
  starts with the post-PR-#111 tree. Earlier PRs are not
  retroactively measured.

---

## 10. Sequencing checklist

Phase 1:

- [x] Land `.cargo/mutants.toml` (§3.3).
- [x] Step 1.1 — `hash.rs` sweep + tests/exemptions.
- [x] Step 1.2 — `error.rs` sweep + tests/exemptions.
- [x] Step 1.3 — `wire.rs` sweep + tests/exemptions.
- [x] Step 1.4 — `dtype.rs` sweep + tests/exemptions.
- [x] **Documentation triangle (§6)**:
  - [x] Draft `docs/src/dev/mutation-testing.md` (§6.1) and link
        from `docs/src/SUMMARY.md` under *Reference*.
  - [x] Add `CONTRIBUTING.md` *Test Structure* row pointing at the
        docs page (§6.1, end).
  - [x] Add *Mutation testing* section + coverage-shape addendum
        row + *Edge cases* pointer to `plans/TEST.md` (§6.2).
  - [x] Cross-link review: each of the three docs references the
        other two correctly (§6.3).
- [x] Wire `mutants-diff` into PR CI as **non-required** (§3.4).
- [x] Step 1.5 — `metadata.rs` sweep + tests/exemptions.
- [x] Step 1.6 — `validate/integrity.rs` sweep + tests/exemptions.
- [x] Step 1.7 — `decode.rs` sweep + tests/exemptions.
- [x] Step 1.8 — `framing.rs` sweep — **deferred to Phase 2 weekly
      sweep (option 3, §7)**.  Two laptop power-down events while
      running this 551-mutant file under all seven concurrency
      clamps + `taskpolicy -b` made local execution unsafe; the
      file is in scope of the weekly sharded sweep just like every
      other non-excluded source file.  Recorded in §7.
- [x] Land `.github/workflows/mutants-weekly.yml` (§4.1).
      *(Brought forward from Phase 2 because step 1.8 is deferred to
      it.  Exists on this branch; goes live the first weekend after
      merge.)*
- [ ] Flip `mutants-diff` to **required-for-merge** *(after the
      first weekend run lands cleanly)*.

Phase 2:

- [ ] Two-week observation window starting at first scheduled
      weekend run after merge.  Handle any auto-issues per the §4.4
      SLA.
- [ ] Triage at least one auto-issue end-to-end (proves the loop).
- [ ] Append "Lessons learned" section to
      `docs/src/dev/mutation-testing.md` covering survivor classes
      observed during stabilisation.

Phase 2 close:

- [ ] Mark this plan as **DONE**; archive into `plans/DONE.md` with
      a one-paragraph retro covering: total tests added, total
      exemptions added, runtime patterns, surprises. Reference both
      `docs/src/dev/mutation-testing.md` and `plans/TEST.md` from
      the retro entry so future agents trace the regime back to its
      origin PR.

---

## 11. Estimated total effort

| Phase | Activity | Wallclock | Developer-time |
|---|---|---|---|
| Phase 1.1–1.4 | Small/medium files + initial config + exemption tuning | ~4 days elapsed | ~6 hr |
| **Documentation triangle (§6)** | Draft `docs/src/dev/mutation-testing.md`, update `CONTRIBUTING.md`, update `plans/TEST.md`, cross-link review | ~1 day elapsed | ~3 hr |
| `mutants-diff` PR-time integration | CI wiring + non-required flag | ~1 day elapsed | ~2 hr |
| Phase 1.5–1.7 | Medium files | ~3 days elapsed | ~4 hr |
| ~~Phase 1.8 `framing.rs` (3 sub-runs)~~ → deferred to Phase 2 | (deferred) | ~5 hr (concurrency-clamp investigation, plan + workflow updates, deferral decision) |
| `mutants-weekly` workflow | YAML + plan + docs updates (this commit) | ~1 day elapsed | ~3 hr |
| `mutants-diff` flip to required | Observation + flip after first clean weekend run | ~7-14 days elapsed | ~1 hr |
| Phase 2 stabilisation | Triage + Lessons-learned doc append | ~14 days elapsed | ~3 hr |
| **Total** | | **~5 weeks elapsed** | **~27 hr** |

Most of the elapsed time is overnight CPU on `framing.rs` (step 1.8)
and the two 14-day observation windows. Active developer time is
modest because the workflow is "run sweep → read missed.txt → write
test or exempt → iterate". The documentation triangle adds ~3 hr
spread across one PR — small enough to land alongside step 1.4
without slowing the file-by-file cadence.

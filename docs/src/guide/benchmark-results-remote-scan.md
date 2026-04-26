# Remote-Scan Walker Benchmark Results

Snapshot of the remote-scan walker microbench on a specific machine.
For methodology, walker design, and the verdict that drove the
default choice, see
[`plans/decisions/remote-bidirectional-default-flip.md`](https://github.com/ecmwf/tensogram/blob/main/plans/decisions/remote-bidirectional-default-flip.md).

> **Note:** Wall-clock numbers are machine-specific and noisy.
> HTTP request counts and bytes fetched are deterministic and
> reproducible.  The decision criterion gates only on the
> deterministic metrics; wall-clock is informational.

## Run metadata

| Field | Value |
|-------|-------|
| **Machine** | Linux 7.0.0-1-cachyos x86_64 |
| **Rust toolchain** | rustc 1.95.0 |
| **Python** | 3.14.4 |
| **Node** | 20+ (per `typescript/package.json` engines) |
| **Methodology** | one fresh handle per cell — open + operation + handle dropped before the metrics fetch, deterministic single-pass, no warmup, no repetition (Criterion's iter-count would smear the request counters) |

## Matrix shape

| Tier (N) | 1 | 10 | 100 | 1000 |
|----------|---|----|-----|------|
| **Fixtures** | header-indexed (4 tiers), footer-indexed (4 tiers), streaming-tail (1 fixture, N=10) |
| **Scenarios** | `message_count`, `read_message(0)`, `read_message(N-1)`, `read_message(N/2)`, `iter` |
| **Walkers** | forward-only (`bidirectional=false`) and bidirectional (`bidirectional=true`) |

Total: 9 fixtures × 5 scenarios × 2 walkers = **90 cells per language**.

## Headline cells

Full-file walk (`iter` scenario), header-indexed:

| Lang | N | walker | total_requests | response_body_bytes | wall_ms |
|------|---|--------|---------------:|--------------------:|--------:|
| rust | 100 | forward-only | 201 | 61 600 | 215 |
| rust | 100 | bidirectional | 202 | **1 540 424** | 216 |
| rust | 1000 | forward-only | 2 001 | 616 000 | 2 110 |
| rust | 1000 | bidirectional | 2 002 | **148 604 024** | 2 114 |
| python | 100 | forward-only | 201 | 61 600 | 68 |
| python | 100 | bidirectional | 202 | **1 540 424** | 70 |
| python | 1000 | forward-only | 2 001 | 616 000 | 686 |
| python | 1000 | bidirectional | 2 002 | **148 604 024** | 747 |
| typescript | 100 | forward-only | 201 | 61 600 | 120 |
| typescript | 100 | bidirectional | 251 | 62 800 | 156 |
| typescript | 1000 | forward-only | 2 001 | 616 000 | 1 096 |
| typescript | 1000 | bidirectional | 2 501 | 628 000 | 1 341 |

Two failure modes coexist:

- **Rust + Python** submit each paired round as
  `object_store::get_ranges(&[fwd_preamble, bwd_postamble])`.
  `object_store`'s coalescer (default merge gap 1 MiB) collapses any
  two ranges within 1 MiB of each other into a single contiguous
  HTTP Range covering **the entire span between them**, then slices
  client-side.  Tightly-packed `.tgm` files keep both cursors well
  within 1 MiB for the whole walk, so every paired round fetches the
  slab between cursors even though the walker only uses 48 bytes
  per round.  On `hundred-msg.tgm` round 1 fetches ~59 KB, round 50
  fetches ~1.7 KB; cumulative ≈ 1.48 MB on a 59 KB file.  The
  pathology runs to ~148 MB on the 592 KB `thousand-msg.tgm`.

- **TypeScript** issues paired Range fetches as
  `Promise.allSettled([fetchRange(fwd), fetchRange(bwd)])` — two
  independent HTTP requests, so bytes stay close to forward (+2% at
  N=1000).  Requests, however, scale at +50% because each backward-
  discovered message also triggers a speculative footer-region fetch
  that the gate inside `tryApplyEagerFooter` discards whenever the
  preamble flags lack `FOOTER_METADATA + FOOTER_INDEX`.  The
  discarded bytes cost a few hundred per message; the wasted GET
  round trip is structural.

## Verdict

**FAIL** in every language.  The bidirectional walker is correct
and produces identical message layouts to forward-only on every
fixture; the cost lives in the transport layer.

Defaults remain **forward-only** across Rust, Python, and
TypeScript.  Opt in per call when the workload's network round-trip
cost exceeds the per-fetch byte cost.

## Reproduce

```bash
# Regenerate fixtures (deterministic shapes, fresh timestamps):
python tests/remote-parity/tools/gen_fixtures.py

# Rust:
cargo run --release -p tensogram-benchmarks --bin remote-scan-metrics

# Python (sync mode shown; pass --mode async for the AsyncTensogramFile path):
python rust/benchmarks/python/bench_remote_scan.py

# TypeScript:
npm --prefix typescript run bench:remote-metrics
```

Each run writes its NDJSON sidecar to
`target/remote-scan-bench/{rust,python,typescript}.ndjson`.  The
schema is identical across the three languages so the cells stack
up for cross-language analysis.

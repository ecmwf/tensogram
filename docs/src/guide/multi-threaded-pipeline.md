# Multi-Threaded Coding Pipeline

Since v0.13.0 Tensogram exposes a caller-controlled thread budget that
spreads encoding and decoding work across a scoped pool of workers.
The feature is **off by default** — existing code paths produce
byte-identical output to previous releases until the caller opts in.

This page covers:

- [The `threads` option](#the-threads-option)
- [Axis-A vs axis-B dispatch](#axis-a-vs-axis-b-dispatch)
- [Determinism contract](#determinism-contract)
- [Environment variable override](#environment-variable-override)
- [Interaction with free-threaded Python](#interaction-with-free-threaded-python)
- [Benchmarks and tuning](#benchmarks-and-tuning)

## The `threads` option

All four bindings expose a `threads: u32` option on encode and decode
entry points:

```rust
use tensogram_core::{encode, decode, EncodeOptions, DecodeOptions};

// Encode with a 4-thread pool:
let msg = encode(&meta, &descriptors, &EncodeOptions {
    threads: 4,
    ..Default::default()
})?;

// Decode with an 8-thread pool:
let (meta, objs) = decode(&msg, &DecodeOptions {
    threads: 8,
    ..Default::default()
})?;
```

```python
import tensogram

msg = tensogram.encode(meta, descriptors, threads=4)
decoded = tensogram.decode(msg, threads=8)
```

```cpp
tensogram::encode_options enc{};
enc.threads = 4;
auto bytes = tensogram::encode(meta_json, objects, enc);

tensogram::decode_options dec{};
dec.threads = 8;
auto msg = tensogram::decode(buf, len, dec);
```

```c
tgm_encode(meta_json, data_ptrs, data_lens, num_objects,
           "xxh3", /* threads= */ 4, &out);
tgm_decode(buf, len, /* verify_hash */ 0, /* native_byte_order */ 1,
           /* threads= */ 8, &msg);
```

```bash
tensogram --threads 8 merge -o merged.tgm a.tgm b.tgm
TENSOGRAM_THREADS=4 tensogram split -o 'part_[index].tgm' input.tgm
```

### Value semantics

| `threads` | Behaviour |
|-----------|-----------|
| `0` (default) | Sequential, single-threaded.  Falls back to the `TENSOGRAM_THREADS` env var if set and non-zero. |
| `1` | Build a scoped 1-worker rayon pool.  Useful for testing — everything flows through the parallel code paths but runs deterministically. |
| `N ≥ 2` | Build a scoped `N`-worker rayon pool for the duration of the call.  Pool is dropped when the call returns. |

### Threshold behaviour

For very small payloads the pool-build cost (~10–100 µs) outweighs any
parallelism gain.  The library transparently skips the pool when the
total payload bytes are below a threshold (default **64 KiB**).  The
threshold is tunable:

```rust
EncodeOptions {
    threads: 8,
    parallel_threshold_bytes: Some(0),       // always parallel
    // parallel_threshold_bytes: Some(usize::MAX), // never parallel
    ..Default::default()
}
```

## Axis-A vs axis-B dispatch

The `threads` budget is spent along one of two axes:

- **Axis A — across objects.**  When a message carries multiple data
  objects and none of them uses an axis-B-friendly codec, rayon
  `par_iter()` runs the encode/decode pipeline for each object on a
  worker in parallel.  Output order is preserved exactly.

- **Axis B — inside one codec.**  When any stage is axis-B-friendly
  (`simple_packing` encoding, `shuffle` filter, `blosc2` or `zstd`
  compression), the budget flows into the codec's internal
  parallelism:

  | Stage | How it uses the budget |
  |-------|------------------------|
  | `simple_packing` encode/decode | Chunked `par_iter` with byte-aligned chunk sizes — output bytes remain identical. |
  | `shuffle` / `unshuffle` | Parallelise the outer `byte_idx` loop (shuffle) or output-chunk scatter (unshuffle). |
  | `blosc2` | `CParams::nthreads` / `DParams::nthreads` — decompress path stays single-threaded in v0.13.0. |
  | `zstd` FFI | `NbWorkers` libzstd parameter on compress; decompress is inherently sequential. |

### Policy

Tensogram messages tend to carry a small number of *very large*
objects, so the library prefers axis B when any codec can use it:

| Object count | Any object axis-B friendly? | Behaviour |
|--------------|-----------------------------|-----------|
| 1 | — | Axis B (codec gets the full budget). |
| N ≥ 2 | **yes** | Axis B on each object sequentially.  Avoids `N × N` thread over-subscription. |
| N ≥ 2 | no | Axis A (`par_iter` across objects), each codec single-threaded. |

This decision happens once per encode/decode call based on the
descriptors.  Nothing is configurable beyond `threads` and
`parallel_threshold_bytes` — the policy is deterministic.

## Determinism contract

v0.13.0 makes two different promises depending on which codecs you use.

### Transparent codecs — byte-identical across thread counts

These stages produce the **same encoded bytes** regardless of
`threads`:

- `encoding = "none"`
- `encoding = "simple_packing"` (at any bits-per-value)
- `filter = "none"`
- `filter = "shuffle"`
- `compression ∈ {none, lz4, szip, zfp, sz3}`

Encoded payload bytes are bit-exact identical for `threads ∈ {0, 1, 2,
4, 8, 16, ...}`.  This is exercised by the
`rust/tensogram-core/tests/threads_determinism.rs` integration suite.

### Opaque codecs — lossless round-trip, may differ

`compression ∈ {blosc2, zstd}` hand off work to third-party C
libraries.  When their internal thread pool is asked to run in
parallel, blocks land in the output frame in *worker completion
order*.  The compressed bytes may therefore differ from the
sequential path — but every variant round-trips losslessly:

- Encode with `threads=8`, decode with `threads=0` → same decoded
  values as a pure sequential round-trip.
- Golden files (produced with `threads=0`) are still byte-for-byte
  stable across releases because the default path is unchanged.

### Why this matters

Determinism across thread counts is the core property that lets
Tensogram users turn `threads` on in production without worrying
about cache keys, deduplication hashes, or reproducible builds
breaking.  The invariant is tested at every layer — Rust, Python,
C FFI, C++ wrapper — with a sweep over `{0, 1, 2, 4, 8}`.

## Environment variable override

`TENSOGRAM_THREADS` is consulted only when the caller-provided
`threads` is `0`.  This matches the existing
`TENSOGRAM_COMPRESSION_BACKEND` pattern:

```bash
# One-shot invocation — every library call inherits the budget.
TENSOGRAM_THREADS=4 python my_pipeline.py

# Explicit option still wins.
tensogram.encode(meta, descs, threads=0)   # sequential (env honoured)
tensogram.encode(meta, descs, threads=1)   # single-threaded (env ignored)
tensogram.encode(meta, descs, threads=16)  # 16 workers (env ignored)
```

The env var is parsed once per process (`OnceLock`), so changing it
mid-run has no effect.

## Interaction with free-threaded Python

`threads` is orthogonal to Python threading.  For CPython 3.13+ built
with `--disable-gil`, you can combine:

- **Python threads** — run multiple Tensogram calls concurrently.
- **Tensogram threads** — each call uses rayon internally.

The PyO3 bindings always release the GIL around encode/decode, so
the two dimensions compose cleanly.  Be careful about total thread
count: N Python threads × M Tensogram threads creates N×M workers.
The safest starting point is one dimension at a time.

## Benchmarks and tuning

The `threads-scaling` benchmark measures encode/decode throughput
for 7 representative codec combinations across a sweep of thread
counts:

```bash
cargo build --release -p tensogram-benchmarks
./target/release/threads-scaling \
    --num-points 16000000 \
    --iterations 5 \
    --warmup 2 \
    --threads 0,1,2,4,8,16
```

Output columns (per case × thread count):

- `enc (ms)`, `dec (ms)` — median wall time over `iterations`.
- `enc MB/s`, `dec MB/s` — throughput based on the *original* byte
  size.
- `ratio` — compressed size as a percentage of original.
- `size (MiB)` — compressed size.
- `enc x`, `dec x` — speedup relative to the `threads=0` baseline.

See the [Benchmark Results](benchmark-results.md) page for numbers on
a reference machine.

### Tuning recommendations

1. **Start with `threads=0`.**  The default is deterministic, well
   tested, and fast for small-to-medium payloads.
2. **Turn it on globally via env.**  `TENSOGRAM_THREADS=$(nproc)`
   is a reasonable starting point for CPU-bound data-movement
   pipelines.  Leave the in-process tensogram calls as `threads=0`
   unless you need finer control per call.
3. **Measure before tuning.**  On small payloads the threshold
   keeps you safe, but the sweet spot for large tensors varies by
   codec.  For simple_packing + szip, 2–4 threads already reaches
   diminishing returns; for blosc2 it can scale further.
4. **Do not stack Python threads × Tensogram threads unless you
   know the total fits your CPU budget.**  Over-subscription
   destroys throughput.

# Hash-While-Encoding — Design & Implementation Plan

**Branch**: `feature/hash-while-encoding`
**Status**: approved
**Target**: folded into `plans/DONE.md` on landing; this file removed at that point.

This document is the working design for the one outstanding optimisation item
in `plans/TODO.md` under **Optimisation → hash-while-encoding**:

> explore a possible optimisation to compute the xxhash while the encoding
> is happening — this would save a second pass through the buffer — analyse
> if this makes sense and if it brings a benefit.

## 1. Problem statement

Every encoded object currently walks its payload at least twice when a hash
is requested (the default): once while the pipeline builds the final
`Vec<u8>` and once more for `compute_hash(&encoded_payload)`.  For the
streaming encoder the payload is actually walked *three* times — once for
hashing, once to memcpy into the frame `Vec` built by
`encode_data_object_frame`, and once to push that `Vec` into the underlying
`Write` sink.

xxh3-64 runs at roughly 30 GB/s on modern x86 and roughly 10 GB/s on
Apple-silicon single-thread, so on large payloads the second pass is a
non-trivial fraction of the pipeline:

| Pipeline (128 MiB raw f64) | Approx. pipeline cost | Approx. hash-pass cost | Hash as % of total |
| -------------------------- | --------------------: | ---------------------: | -----------------: |
| `none + none`              |                ~3 ms  |                 ~4 ms  |              ~57 % |
| `none + lz4`               |               ~10 ms  |                 ~3 ms  |              ~21 % |
| `sp(24) + szip`            |              ~100 ms  |                 ~1 ms  |               ~1 % |
| `sp(24) + zstd(3)`         |               ~50 ms  |                ~0.5 ms |               ~1 % |

Eliminating the second pass is therefore a real (but modest) win on the
transparent/light pipelines — and on the streaming encoder it additionally
removes a redundant memcpy.  It is always free for pipelines that already
dominate end-to-end time.

The `xxhash-rust` crate already in the workspace exposes a streaming
hasher (`xxh3::Xxh3Default` with `update(&[u8])` + `digest() -> u64`) that
is *bit-identical* to the one-shot `xxh3_64(data)` call by construction
(both use seed `0` and the default secret).  The streaming hasher is the
building block for this PR.

## 2. Design

Three targeted changes, each independently testable:

### 2A. Pipeline-side — hash on the way out

Add an opt-in hash output to `tensogram-encodings::pipeline`:

```rust
pub struct PipelineConfig {
    // …existing fields…
    /// Compute xxh3-64 over the final encoded bytes inline with encoding.
    /// Set by the `tensogram-core` layer when `EncodeOptions.hash_algorithm
    /// == Some(Xxh3)`.  Leaving this `false` preserves pre-PR behaviour
    /// exactly.
    pub compute_hash: bool,
}

pub struct PipelineResult {
    pub encoded_bytes: Vec<u8>,
    pub block_offsets: Option<Vec<u64>>,
    /// xxh3-64 digest of `encoded_bytes`, produced inline when
    /// `config.compute_hash == true`.  `None` otherwise.
    pub hash: Option<u64>,
}
```

Behaviour in `encode_pipeline` and `encode_pipeline_f64`:

* **Passthrough** (`encoding=None + filter=None + compression=None`) — the
  codec output is `Cow::Borrowed(src)`.  Instead of `filtered.into_owned()`
  (which reads `src` once for the memcpy) followed by `xxh3_64` (which
  reads the just-copied Vec a second time), call a fused
  `copy_and_hash(src, hasher)` helper that walks `src` once, filling the
  destination Vec and updating the hasher in lockstep.
* **Encoded / filtered / compressed** — the final `Vec<u8>` comes out of the
  codec (`simple_packing`, `shuffle`, blosc2, zstd, szip, lz4, zfp, sz3).
  Hash is computed immediately *while the Vec is still cache-hot* — one
  pass through L2/L3, not DRAM.
* **`compute_hash == false`** — hasher is never instantiated.  Zero
  overhead; pre-PR behaviour byte-for-byte.

`copy_and_hash` uses a 64 KiB chunk loop so that each chunk stays in L1
while both the copy and the hasher consume it.  For buffers below that
size the loop degenerates to a single call, equivalent to one-shot
`update + to_vec`.

xxh3 is a strictly serial algorithm; no parallel decomposition is
attempted.  Parallelism in the encoding layers (rayon `par_iter` inside
`simple_packing` / `shuffle`, libblosc2's internal worker pool, libzstd's
`NbWorkers`) has **already joined** by the time we own the final Vec, so
the hasher runs on the calling thread with no synchronisation.

### 2B. Buffered encoder — use pipeline hash

In `tensogram-core::encode::encode_one_object`, the `EncodeMode::Raw`
branch sets `config.compute_hash = options.hash_algorithm.is_some()`
before calling the pipeline, and populates the descriptor from
`PipelineResult.hash` instead of calling `compute_hash` a second time.

The `EncodeMode::PreEncoded` branch is untouched: the caller's bytes
never go through the pipeline, so there is no existing pass to fuse
with.  That path keeps its single hash pass over the input bytes, which
is already optimal for the pre-encoded contract.

The public `compute_hash(&[u8], HashAlgorithm) -> String` and
`verify_hash` functions stay exactly as they are — they are part of the
library's public API, used by validation, FFI, and callers of
`encode_pre_encoded`.

### 2C. Streaming encoder — fused write-and-hash

`StreamingEncoder::write_object_inner` currently reads the encoded
payload three times:

1. `compute_hash(encoded_bytes)` — read 1 (full walk).
2. `encode_data_object_frame(&final_desc, encoded_bytes, false)` — read 2
   (memcpy into a frame `Vec`).
3. `self.writer.write_all(&frame_bytes)` — read 3 (Vec → stream).

The fused path writes frame pieces directly to `W: Write`:

1. Build a **placeholder CBOR descriptor** that carries
   `HashDescriptor { hash_type: "xxh3", value: "0".repeat(16) }`.  CBOR
   encoding is length-prefixed, so this placeholder has the exact byte
   length the final CBOR will have (the `xxh3` algorithm always produces
   a fixed 16-char hex value).  We use the placeholder only to learn
   `cbor_len` — it is never written.
2. Write the `FrameHeader` with the correct
   `total_length = FRAME_HEADER_SIZE + cbor_len + payload.len() + DATA_OBJECT_FOOTER_SIZE`.
3. Write the payload to the sink in 64 KiB chunks, calling
   `hasher.update(chunk)` before each `writer.write_all(chunk)`.  Read 1
   only.
4. Finalise the hash, substitute it into the descriptor, serialise the
   **final** CBOR, assert its length matches the placeholder length
   (debug assert — wire-format regression guard).
5. Write the final CBOR, the `cbor_offset` (8 bytes, big-endian), and
   `FRAME_END` (4 bytes).

The existing `encode_data_object_frame` function is left untouched; it
remains the buffered-path builder.  The streaming path gets a new private
`write_data_object_frame_hashed` helper.

## 3. Interaction with the multi-threaded pipeline

The multi-threaded pipeline (shipped in v0.13.0 as
`multi-threaded-coding-pipeline`) has two parallel axes:

* **Axis A** — `par_iter` across objects.  Each object is encoded by a
  distinct rayon worker; each worker owns its own `Xxh3Default` on its
  own stack.  `Xxh3Default` is `!Sync` and `!Send`-across-scope (it is
  `Send`, but never crosses scopes in this design), so the compiler
  prevents sharing.  Hashes of different objects do not interact.
* **Axis B** — intra-codec parallelism (`simple_packing` chunked,
  `shuffle` chunked, blosc2 `nthreads`, zstd `NbWorkers`).  Parallelism
  lives *inside* the codec call.  The codec returns a single contiguous
  `Vec<u8>` after its internal pool has joined, and only then does
  hashing start.  The hasher runs serially in the calling thread on the
  already-assembled buffer.

**Determinism contract stays identical to 0.13.0.**  Transparent codecs
(`none`, `lz4`, `szip`, `zfp`, `sz3`, `simple_packing`, `shuffle`) are
byte-identical across thread counts, so their hashes are byte-identical
too.  Opaque codecs (`blosc2`, `zstd` with workers) are lossless
round-trip but may reorder compressed blocks by worker-completion order;
their hashes follow the same rule — reproducible within a run, possibly
differing across runs with different `threads` values.

A new test `test_hash_determinism_across_thread_counts` sweeps
`threads ∈ {0, 1, 2, 4, 8}` on each codec combo and verifies the
appropriate invariant:

* Transparent codecs → the hash produced at `threads=N` matches the hash
  produced at `threads=0`.
* Opaque codecs → the hash produced at `threads=N` matches
  `xxh3_64(encoded_bytes_N)` (internal consistency), but may differ from
  the `threads=0` hash.

A one-line comment is added at the hash site in
`tensogram-encodings/src/pipeline.rs` to make the "runs in caller thread,
after all parallel work has joined" invariant visible to future
maintainers.

## 4. Considered alternative — move hash out of CBOR into frame footer

Discussed with the maintainer during plan review.  Decision: **out of
scope for this PR**.  Reasons:

* The optimisation already works with the current wire format — the
  streaming path exploits the fact that `xxh3` produces a fixed-length
  hex string, so CBOR length is deterministic without the hash value.
* A wire-format change would require regenerating all 5 golden `.tgm`
  files, updating every language binding (Rust, Python, C FFI, C++,
  WASM, TypeScript), and re-reviewing the `HashDescriptor` design
  (algorithm byte, 64-vs-128-bit, presence flag).  That deserves its
  own design round.
* Filed as a new entry in `plans/IDEAS.md` under the name
  `hash-in-frame-footer` for separate consideration.

## 5. What changes, what does not

**Changes**:

* `rust/tensogram-encodings/Cargo.toml` — add `xxhash-rust.workspace = true`.
* `rust/tensogram-encodings/src/pipeline.rs` — `PipelineConfig.compute_hash`,
  `PipelineResult.hash`, `copy_and_hash` helper, inline hashing in both
  `encode_pipeline` and `encode_pipeline_f64`.
* `rust/tensogram-core/src/encode.rs` — `encode_one_object` wires
  `compute_hash` through and reads back `PipelineResult.hash`.
* `rust/tensogram-core/src/streaming.rs` — `write_object_inner` refactored
  to use the new fused `write_data_object_frame_hashed` helper.
* `rust/benchmarks/` — new `hash_overhead` criterion benchmark comparing
  the two-pass baseline against the fused path on
  `{ none+none, none+lz4, sp(24)+szip, sp(24)+zstd }`.
* `plans/DONE.md` — new entry describing the optimisation.
* `plans/DESIGN.md` — Integrity Hashing section gets a paragraph on the
  inline-hashing invariant.
* `plans/IDEAS.md` — `hash-in-frame-footer` speculative entry.
* `docs/src/guide/multi-threaded-pipeline.md` — short "Interaction with
  integrity hashing" subsection.

**Unchanged**:

* Public APIs of `tensogram-core`, `tensogram-ffi`, `tensogram-cli`,
  `tensogram-wasm`, Python bindings, C++ wrapper, TypeScript wrapper.
* `EncodeOptions`, `DecodeOptions`, `HashDescriptor`, `compute_hash`,
  `verify_hash`.
* Wire format — every message byte identical to pre-PR output for every
  codec and every thread count.
* Golden `.tgm` fixtures — byte-identical, verified by the existing
  `tests/golden_files.rs` suite.
* `encode_pre_encoded` hashing behaviour — still a single pass over the
  caller's bytes (already optimal).

## 6. Test matrix (TDD, behaviour-driven)

* **Hash equivalence** — `Xxh3Default::new().update(chunks).digest()`
  matches `xxh3_64(concat(chunks))` for sizes 0, 1, 239 (short-input
  boundary), 240 (mid-size boundary), 1 MiB (stripe boundary).  This
  test guards against xxhash-rust API drift.
* **Pipeline hash invariant** — for every codec combo in the existing
  matrix, `PipelineResult.hash == Some(xxh3_64(PipelineResult.encoded_bytes))`
  when `compute_hash = true`, and `None` when `compute_hash = false`.
* **Golden file byte-identity** — existing
  `tests/golden_files.rs` must pass unchanged.  The 5 fixtures encode
  without re-generation.
* **Round-trip + verify_hash** — existing integration tests must pass
  unchanged.
* **Buffered/streaming hash parity** — encode the same object through
  both paths, assert hash strings match.
* **Multi-thread determinism** (new) — for each transparent codec and
  each `threads ∈ {0,1,2,4,8}`, the descriptor hash matches
  `threads=0`.  For opaque codecs, the hash is internally consistent
  with the encoded bytes.
* **Passthrough fused correctness** — `copy_and_hash(src, &mut h)`
  returns `(src.to_vec(), xxh3_64(src))` for sizes 0, 1, 64 KiB − 1,
  64 KiB, 64 KiB + 1, 1 MiB.
* **Zero-overhead when disabled** — pipeline with `compute_hash = false`
  produces `PipelineResult.hash == None` and allocates no hasher
  (structural test: hasher instantiation is behind the flag).

## 7. Benchmark

New criterion benchmark in `rust/benchmarks/benches/hash_overhead.rs`:

* Workload: 16 Mi `f64` values (128 MiB raw), weather-like sinusoidal
  data (matches the existing benchmarks for comparability).
* Pipelines: `none+none`, `none+lz4`, `sp(24)+szip`, `sp(24)+zstd(3)`.
* Groups per pipeline:
  1. `baseline_no_hash` — current behaviour, `hash_algorithm = None`.
  2. `baseline_two_pass` — current behaviour, `hash_algorithm =
     Some(Xxh3)`, separate pass.
  3. `fused_inline` — new behaviour, `compute_hash = true` in
     `PipelineConfig`, hash produced by the pipeline.
* Reports encode-only MB/s throughput for each group.

Results captured into `plans/DONE.md` on landing.

## 8. Verification checklist

```
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo bench -p tensogram-benchmarks --bench hash_overhead  # manual
# Python + C++ byte-for-byte parity (golden files already covered by Rust)
source .venv/bin/activate
(cd python/bindings && maturin develop)
python -m pytest python/tests/ -v
```

On success: record bench numbers in `plans/DONE.md`, delete this
`HASH_WHILE_ENCODING.md` file, and mark the TODO item complete.

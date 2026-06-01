# Features Decided to Implement

Accepted features that are planned but not yet implemented.
They should contain some notes about the intended aims and direction of implementation.
Code Agents are very much encouraged to ask questions to get the design correct, seeking also  clarifications and sorting out ambiguities.

For speculative ideas, see `IDEAS.md`.

## C++ Async API

The core async C++ surface has shipped — callback / `std::future` /
C++20 coroutine frontends over a callback-based FFI core, cancellation
+ timeouts, remote reads via `open_remote`, and a local-file async
streaming encoder.  See the *Asynchronous C++ API* entry in
`../CHANGELOG.md` and the *Asynchronous frontends* section of
`ARCHITECTURE.md`.  Open follow-ups:

- [ ] **cpp-async follow-up: real `tgm_runtime_shutdown_blocking`**
    `tgm_runtime_shutdown_blocking(timeout_ms)` in
    `rust/tensogram-ffi/src/async_core.rs` is a no-op stub: it ignores
    `timeout_ms` and returns `0` (always "zero tasks unfinished").  The
    intended runtime-lifecycle contract is: call
    `Runtime::shutdown_timeout(Duration::from_millis(timeout_ms))`,
    cancel pending tasks, free every outstanding `tgm_async_file_t` /
    `tgm_async_streaming_encoder_t` handle, and **return the count of
    tasks that did not finish within the timeout** so callers can
    log / abort.

    Pickup notes:
    - **Blocker:** the runtime lives in
      `SHARED_RUNTIME: OnceLock<Result<Runtime, String>>` (lazy build,
      build error cached and surfaced as `TGM_ERROR_IO`).
      `Runtime::shutdown_timeout` consumes `self`, which a `OnceLock`
      cannot hand back.  Switch the singleton to an owning container
      that supports `take` (e.g. `Mutex<Option<Result<Runtime,
      String>>>` or `Arc<RwLock<Option<Runtime>>>`) while keeping the
      lazy-build + cached-build-error behaviour.
    - **Task counting:** there is no live-task counter today.  Add one
      at the two spawn choke points — `spawn_task` and
      `spawn_or_set_error` in `async_core.rs` — via an `AtomicUsize`
      (increment on spawn, decrement on completion/drop) or a
      `tokio_util::task::TaskTracker`; `shutdown_timeout` reports the
      residual.
    - **Post-shutdown:** once shut down the runtime is single-shot — no
      rebuild; every subsequent `tgm_async_*` call returns
      `TGM_ERROR_IO` with a descriptive `tgm_last_error()`.
    - **Test to write:** `cpp/tests/test_async_shutdown_during_flight.cpp`
      — fire shutdown while N tasks are mid-flight, assert the returned
      unfinished count, assert subsequent `tgm_async_*` calls return
      `TGM_ERROR_IO`.  Acceptance: "drains in-flight
      tasks within the supplied timeout and reports cleanly on
      overflow."
    - The C ABI (`u64` return) already matches; only the Rust
      implementation behind it changes.

- [ ] **cpp-async follow-up: object-store / remote streaming *writes***
    The streaming-async **write** path shipped as local-file only
    (`tokio::fs::File`).  The object-store producer path (streaming a
    `.tgm` straight to `s3://` / `gs://` / `az://` so an HPC producer's
    network write overlaps the next step's compute) was descoped during
    implementation — documented in `docs/src/guide/cpp-async.md`
    "What's not in scope (v1)".  Remote *reads* (`open_remote`) are
    already shipped; this entry is the write half.

    Pickup notes:
    - **Core is already sink-agnostic.**
      `AsyncStreamingEncoder<W: AsyncWrite + Unpin>` in
      `rust/tensogram/src/streaming_async.rs` is generic over the sink;
      the FFI wrapper in `rust/tensogram-ffi/src/async_streaming.rs`
      hardwires `tokio::fs::File::create`.  Work needed: (a) an
      `AsyncWrite` adapter over `object_store`'s multipart-upload API,
      and (b) URL-scheme dispatch in
      `tgm_async_streaming_encoder_create`, which already accepts
      `path_or_url` + `storage_keys` / `storage_values`.
    - **Part buffering:** `object_store::MultipartUpload` takes
      whole parts; S3 enforces a 5 MiB minimum part size (final part
      exempt).  Maintain an internal part-buffer of
      `multipart_part_size_bytes` — already plumbed through
      `tgm_runtime_configure` (default 8 MiB) but currently stored-and-
      unused on the write side — flush a part at the threshold, upload
      the residual on `finish()`, then `CompleteMultipartUpload`.
    - **Retry / idempotency:** the inline xxh3-64 frame hash is
      computed once per frame in the calling thread *before* upload;
      hold each part-buffer until `put_part` returns `Ok`; never re-hash
      on a transient retry so the landed bytes match the installed hash
      slot bit-for-bit.  Non-transient (4xx) failures surface as
      `TGM_ERROR_REMOTE` and leave the upload unfinalised.
    - **Backfill:** object stores cannot seek, so `backfill=true`
      is silently ignored — the message ships `total_length = 0`
      (forward-scan only) with a one-shot `tracing::warn!`; local-file
      backfill semantics are unchanged.
    - **Tests:** `rust/tensogram/tests/streaming_async_object_store.rs`
      (S3-mock + GCS-mock via `object_store` test fixtures; happy path +
      one transient-retry scenario) and
      `cpp/tests/test_async_streaming_remote.cpp` (in-process HTTP
      fixture, no external network).  Assert remote-written
      bytes round-trip through the sync decoder and are byte-identical
      to local-file output for the same sequence of writes.
    - **Done when:** a producer example writing to `s3://` / `gs://` /
      `az://` round-trips; the "What's not in scope (v1)" remote-write
      caveat is removed from `docs/src/guide/cpp-async.md`; and the
      local-file-only notes in `streaming_async.rs` /
      `async_streaming.rs` are lifted.  Main risk: per-backend
      `MultipartUpload` differences — S3 5 MiB min part, GCS resumable
      upload, Azure block blob.

- [ ] **cpp-async follow-up: close mutation-test gaps in
      `streaming_async.rs` and `tensogram-ffi/src/async_*.rs`**
    The `Mutants (diff)` CI job for PR #115 surfaced ~150 MISSED
    mutants concentrated in:
    - `rust/tensogram/src/streaming_async.rs`: most accessor /
      finish / write helpers can be replaced with `Ok(())` or
      `Default::default()` and the existing round-trip tests still
      pass.  Need targeted unit tests on `object_count`,
      `bytes_written`, `write_padding`, `write_footer_frames_and_postamble`,
      and the `+= -=`/`< >=` arithmetic-flag mutations on the byte
      counter logic.
    - `rust/tensogram/src/streaming.rs:779` (`padding_for`): the
      arithmetic body has 8 missed mutations (% → +/-, padding-for → 0/1,
      etc.).  A direct unit test on `padding_for(n, align)` covering
      `align = 1, 8, 64` and `n = 0, align - 1, align, align + 1`
      would close all of these.
    - `rust/tensogram-ffi/src/async_streaming.rs`: 9 null-arg
      mutations are now killed by tests in
      `async_streaming_tests.rs` (see commit `8a10037`); remaining
      mutations are largely in the descriptor / metadata parsing
      paths.

    Each individual fix is small (5-15 lines of test code per missed
    mutant), but the volume justifies a separate hardening PR rather
    than blocking this one.  The current CI run is also constrained
    by the `Mutants (diff)` job's 30-minute timeout — a large PR
    diff plus serialised-by-design `MUTANTS_JOBS=2` runs through
    only ~30 mutations before the budget elapses, so even if all
    mutations were killed locally, the CI signal would remain
    "cancelled".  Consider either bumping the timeout to 90 min for
    feature-PR scope, or splitting future feature PRs so each
    sits under the typical small-diff window.

- [ ] **cpp-async follow-up: cross-language async parity test**
    The cross-language parity goal: a C++ async **producer**
    feeding a Python async **consumer** through a shared in-process HTTP
    fixture, proving the async streaming encoder's wire output is
    decodable by a different language's async reader.  Only the
    single-process C++ producer/consumer integration test
    (`cpp/tests/test_async_producer_consumer.cpp`) shipped; the
    cross-language half is still open.  The `cpp-async` /
    `cpp-streaming-async` guides and the README / ARCHITECTURE
    entries are already done.

    Pickup notes:
    - **Shape:** C++ side writes a multi-message `.tgm` via the async
      streaming encoder; Python side reads it back with
      `AsyncTensogramFile` and asserts decoded-array + metadata
      equality.  Reuse the in-process Range-capable HTTP fixture from
      the remote-read tests as the transport.
    - **Where it lives:** either a new `tests/cpp_async_parity/`
      directory driven from CI, or fold it into the
      existing `tests/remote-parity/` harness, which already spins up a
      mock HTTP server and cross-language drivers.
    - **Gating:** the producer currently writes local files only, so a
      first version can hand off a local `.tgm` path (or `file://`).
      The full "producer streams to a store, consumer reads from the
      store" shape depends on the object-store-write entry above.

## Multi-Language Support

  - [ ] **fortran-interface — follow-ups (synchronous binding DELIVERED)**
    The shallow `iso_c_binding` binding over the C FFI has **shipped**: the
    `fortran/` Fortran 2008 library (Path A — a hand-written module over the
    existing `libtensogram`, no new C/Rust code), with generic encode/decode,
    the file API, metadata + the encoding pipeline, and the streaming encoder;
    the column-major contract (reversed on-wire shape/strides); examples;
    tests; the user guide (`docs/src/guide/fortran-api.md`); and CI — CMake +
    fpm, the `fortran-f2008-check` conformance gate, bidirectional
    Fortran↔C/C++ and Fortran↔Python parity, and the error-enum↔`tensogram.h`
    consistency check. `fortran/fpm.toml` is on the VERSION-sync list. See
    `PLAN_FORTRAN.md` (§7 milestones, §10 decisions) and the `CHANGELOG`.
    Remaining, demand-driven follow-ups:
    - **Async surface** — the C async path uses completion callbacks (the
      hardest part to bind and the least in demand for blocking NWP codes).
      Bind the blocking `tgm_async_task_join_*` variants before callbacks;
      do not start without a real consumer (PLAN_FORTRAN.md §5.7).
    - **Extra dtypes** — `int8`/`int16`/`complex`/`float16`. Fortran has no
      native unsigned or half/complex-as-pair scalar mapping, so each needs a
      deliberate representation decision.
    - **Zero-copy non-contiguous input** via `CFI_cdesc_t` / TS 29113 — only
      worth it if a concrete need appears; the contiguous-gather path covers
      today's usage (PLAN_FORTRAN.md §4.1, Path C).

  - [ ] **typescript-wrapper (Scope C.3) — distribution & CI maturity**
    Three intertwined tasks that all touch the build, pack, and publish
    pipeline. Best done together so we don't re-open CI config three
    times.
    - **npm publish pipeline.** Choose an npm org (`@ecmwf.int/tensogram`
      already exists in `package.json`), wire a GitHub Actions job that
      publishes on tagged releases and enforces semver lock-step with
      the root `VERSION` file (add `typescript/package.json` to the
      VERSION-sync list in CLAUDE.md — already noted but not wired).
    - **Browser-environment CI.** Today the `typescript` CI job runs
      Vitest in Node only. Add a browser lane (Vitest `browser` mode
      via Playwright, or a dedicated Playwright job) so regressions
      that only surface in a real browser (`import.meta.url` resolution,
      streams API, `fetch` semantics) can't slip through.
    - **Bundle-size budget via `size-limit`.** Track the produced
      `dist/*.js` + `wasm/tensogram_wasm_bg.wasm` size deltas on every
      PR. Fail CI if the growth exceeds a documented threshold
      (e.g. WASM ≤ 1.5 MiB, JS glue ≤ 30 KiB).
    - **Micro-benchmarks on hot paths.** Add `vitest bench` runs for
      `encode`, `decode`, and `decodeStream` so allocation regressions
      can be spotted without running full load-test scenarios.

  - [ ] **typescript-wrapper (Scope C.4) — Zarr.js integration**
    JavaScript mirror of the `tensogram-zarr` package. Lets downstream
    JS tooling (browser-based data explorers, notebook viewers)
    consume Tensogram-backed Zarr stores without re-implementing the
    chunk-store contract. Ships as a separate `@ecmwf.int/tensogram-zarr`
    npm package that depends on `@ecmwf.int/tensogram`.
    - Implement the Zarr v3 chunk-store trait against a `TensogramFile`
      backend.
    - 14 bidirectional dtype mappings to match the Python
      `tensogram-zarr` test matrix.
    - Smoke-test from a notebook-style fetch scenario
      (`zarr-js` reading a `.tgm`-backed store over HTTPS).
    - Standalone CI job; does not block the main `@ecmwf.int/tensogram`
      pipeline.

## Integration with other software

- [ ] **earthkit-data-integration follow-ups** — scope notes carried
    over from PR #88; intentionally deferred so each can land as its
    own small PR.
    - **upstream-readers-relocation** — open a PR against
      [`ecmwf/earthkit-data`](https://github.com/ecmwf/earthkit-data)
      moving the reader callables (`reader` / `memory_reader` /
      `stream_reader`) from
      `python/tensogram-earthkit/src/tensogram_earthkit/readers/`
      into earthkit-data's own `readers/tensogram/` tree.  The
      current layout was deliberately mirrored (one-to-one file
      shape, identical callable signatures) so the upstream change
      is a verbatim directory copy plus an entry-point adjustment.
    - **progressive-stream-reader** — replace the current
      drain-to-bytes path in
      `tensogram_earthkit/readers/stream.py` with a true
      yield-as-each-message-arrives reader.  Today the stream is
      drained into a `bytes` buffer and dispatched through the
      memory path because the xarray backend needs a concrete file
      and the FieldList contract requires `__len__` up-front;
      progressive yields will need either a streaming xarray
      adapter or a two-pass FieldList that lets length resolve
      lazily.
    - **earthkit-encoder-pipelines** — let
      `TensogramEncoder.encode` / `to_target` accept a tuned
      encoding pipeline (`encoding`, `filter`, `compression`,
      `bits_per_value`, …) instead of the current lossless
      pass-through (`encoding=filter=compression="none"`).  The
      Python `tensogram.encode` API already exposes these knobs;
      the earthkit encoder should thread them through end-to-end so
      the earthkit surface is feature-equivalent to the native
      Python API.
- [ ] **torch**
    - convenience methods for tensogram as/from torch, to avoid the numpy intermediary. Wilder ideas and optimizations are additionally given in IDEAS.md
- [ ] **nvidia stack**
    - cuFile or similar interface
- [ ] **arrow/parquet**
    - analyze where we can integrate with arrow/parquet stack and implement: file reading, streaming data conversion (in both directions -- offering tensogram via arrow streaming api, as well as converting arrow streams into tensogram messages)
- [ ] **mlx**
    - similarly to torch, conveninece methods for tensogram as/from mlx frame, to avoid the numpy intermediary


## Documentation

- [ ] interactive-docs:
  - make the docs interactive.
  - since we have WASM, you can embed demos in the docs

## Remote Access

- [ ] **remote 7 — TS lazy scan: 256 KB forward-chunk variant**:
  - during `lazyScanMessages`, fetch one 256 KB chunk per message instead of 24 bytes
  - for header-indexed messages, parse metadata + index inline via `parse_header_chunk`
    (saves the second round trip `#ensureLayout` would otherwise do)
  - gated by a benchmark that shows the round-trip saving outweighs the larger
    per-message fetches on realistic server latencies (RTT ≥ 20 ms)
  - requires the current preamble-only walk to remain as the fallback when any
    chunk-parse fails (bail-to-eager path should stay identical)

## Code Quality

- [ ] **mutation-testing rollout — ongoing**:
    `cargo-mutants` (pinned 27.0.0) measures test depth on critical-path
    Rust modules.  Process reference: `docs/src/dev/mutation-testing.md`;
    per-mutant exemptions live in `.cargo/mutants.toml`; the test-shape
    map is `plans/TEST.md`.  Remaining work:
    - **Weekly sweep triage.** The sharded weekly full sweep
      (`.github/workflows/mutants-weekly.yml`) opens auto-issues tagged
      `mutation-testing` for surviving mutants; triage them (~7-day
      target) — genuine survivors get a test, equivalent/cosmetic ones
      get an `exclude_re` entry with a one-line rationale (no mass
      suppression).
    - **Finish critical-path coverage.** The manual critical-path sweep
      covered `hash.rs`, `error.rs`, `wire.rs`, `metadata.rs`,
      `dtype.rs`, `validate/integrity.rs`, `decode.rs`; `framing.rs`
      (frame ordering, scan recovery, `decode_message`) was deferred to
      the weekly machinery rather than a manual pass — confirm it reaches
      zero unexempted survivors there.
    - **PR-time gate.** The non-blocking `make mutants-diff` CI job can
      flip to required-for-merge once the config has stabilised; the job
      is currently constrained by a 30-minute budget under serialised
      `MUTANTS_JOBS=2`, so bump the timeout or keep feature PRs small.

- [ ] **descriptor ↔ frame-payload consistency checks on decode**:
    - Complementary to the preallocation hardening (already shipped):
      instead of *surviving* a pathological `num_values` via fallible
      allocation after the fact, *reject* malformed descriptors cheaper
      and earlier by cross-checking the descriptor's claimed output
      size against the frame's actual payload length (known from the
      frame header) before any decompression runs.
    - Three tiers of strictness depending on the pipeline:
        - **Exact** for `encoding="none" + compression="none"`:
          `frame_payload_bytes == num_values × dtype_byte_width`
          (and `ceil(num_values / 8)` for the `bitmask` dtype). A
          mismatch in either direction is categorically malformed.
        - **Exact** for `encoding="simple_packing" + compression="none"`:
          `frame_payload_bytes == ceil(num_values × bits_per_value / 8)`.
          The current simple_packing decoder rejects too-small payloads
          with `InsufficientData` but silently ignores too-much data
          — this TODO tightens the too-much-data direction too.
        - **Plausibility ratio** for compressed codecs (`zstd`, `lz4`,
          `szip`, `blosc2`, `zfp`, `sz3`):
          `num_values × dtype_byte_width ≤ frame_payload_bytes ×
          MAX_PLAUSIBLE_RATIO`. Pick a conservative cap (probably
          around `1000×`) that accommodates pathological-but-legitimate
          high-compression inputs (RLE on all-zero bitmasks,
          szip on constant data) while still rejecting claims wildly
          disproportionate to the compressed payload.
    - Fit: `pipeline::decode_pipeline` gains a
      `validate_descriptor_size(encoded.len(), config)` step right
      before the `compressor.decompress(encoded, expected_size)` call.
      `decode_range_pipeline` gets a matching check sized against
      the sliced chunk rather than the full frame.
    - New error: `PipelineError::DescriptorSizeMismatch { claimed_bytes,
      payload_bytes, codec }`, marked `#[non_exhaustive]` consistent
      with the other error enums hardened in PR #90.
    - Why separate from the preallocation-hardening PR:
        - Distinct mechanism (upstream structural validation vs
          downstream fallible allocation).
        - More specific operator-visible errors ("descriptor claims
          4 TiB but frame is 50 bytes") than the generic
          "failed to reserve".
        - Distinct test matrix (one per encoding × compression ×
          {match, too-small, too-big, ratio-plausible, ratio-implausible}).
    - Tests (behaviour-driven):
        - Exact tier: passthrough / bitmask / simple_packing
          round-trips with matched sizes succeed; with off-by-one
          payload lengths surface `DescriptorSizeMismatch`.
        - Ratio tier: hand-craft a `.tgm` with 20-byte compressed
          payload + descriptor claiming 4 TiB decoded size, assert
          the typed error fires before any decompression is
          attempted.
        - Boundary: a descriptor claiming `compressed × 1000` bytes
          is accepted; `compressed × 1001` is rejected (or whatever
          ratio is picked).

## Viewer

- [ ] Wire LevelSelector into the UI for 3D pressure-level fields
- [ ] Cache rendered frames client-side for instant scrubbing through previously viewed steps
- [ ] OffscreenCanvas in worker to avoid main-thread canvas.toDataURL
- [ ] Cache decoded Float32Arrays to skip WASM decode when revisiting fields
- [ ] Pre-fetch next N frames during animation playback
- [ ] URL state persistence (selected file, field, colour scale)
- [ ] Keyboard shortcuts: space play/pause, arrow keys step
- [ ] Resizable sidebar (drag handle)
- [ ] Handle polar stereographic projections and single-point fields

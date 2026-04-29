# Features Decided to Implement

Accepted features that are planned but not yet implemented.
They should contain some notes about the intended aims and direction of implementation.
Code Agents are very much encouraged to ask questions to get the design correct, seeking also  clarifications and sorting out ambiguities.

For speculative ideas, see `IDEAS.md`.

## Multi-Language Support

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

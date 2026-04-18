# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Strict-finite encode checks** — two new `EncodeOptions` flags,
  `reject_nan` and `reject_inf`, that scan float payloads before the
  encoding pipeline runs and bail out on the first NaN / Inf with a
  clean `EncodingError` carrying the element index and dtype. Both
  default to `false` (backwards-compatible). Integer and bitmask
  dtypes skip the scan (zero cost). The guarantee is
  pipeline-independent — applies to `encoding="none"`,
  `"simple_packing"`, and every compressor. Primary motivation: close
  the silent-corruption gotcha where `simple_packing::compute_params`
  accepts `Inf` input and produces numerically-useless parameters
  that decode to NaN everywhere (see `plans/RESEARCH_NAN_HANDLING.md`
  §3.1). Exposed across every language surface (Rust, Python, TS,
  C FFI, C++) with cross-language parity tests. CLI global flags
  `--reject-nan` / `--reject-inf` plus `TENSOGRAM_REJECT_NAN` /
  `TENSOGRAM_REJECT_INF` env vars for ops rollouts. New
  `docs/src/guide/strict-finite.md` covers the full semantics.
- **NaN-handling research memo** —
  `plans/RESEARCH_NAN_HANDLING.md` catalogues every path through the
  library where NaN/Inf can appear on the encode side, documents the
  15 gaps/gotchas (from the newly closed §3.1 silent-Inf-corruption
  to the still-open §3.3 zfp/sz3 undefined behaviour), and proposes
  tiered future work.

### Changed
- **FFI signature extension** — `tgm_encode`, `tgm_file_append`, and
  `tgm_streaming_encoder_create` now take two additional
  `reject_nan` / `reject_inf` bool parameters. `tgm_encode_pre_encoded`
  intentionally does not — pre-encoded bytes are opaque to the library.
  Pre-0.15 C callers will see a compile error from the regenerated
  header; pass `false, false` to preserve old behaviour.

## [0.15.0] - 2026-04-18

### Changed
- **Crate renamed** — `tensogram-core` → `tensogram`. Users now
  `cargo add tensogram` instead of `cargo add tensogram-core`.
  Rust imports change from `use tensogram_core::` to `use tensogram::`.
- **Backwards-compatible redirect** — `tensogram-core@0.15.0` is published
  as a thin re-export crate with all features forwarded; existing users
  can upgrade at their own pace.

### Added
- **Hash-while-encoding** — xxh3-64 digest computed inline during the
  encode pipeline with zero extra passes over the data. New
  `compute_hash` field on `PipelineConfig`.
- **Installation docs** — README and documentation now include install
  instructions for Rust (`cargo add tensogram`), Python
  (`pip install tensogram[all]`), and CLI (`cargo install tensogram-cli`).
- **Registry badges** — crates.io and PyPI badges in the README header.
- **CLI subcommands** — additional CLI commands added.

### Fixed
- Publish workflow: sparse index polling (replaces slow HTTP API polling),
  `--allow-dirty` for excluded crates, `--find-interpreter` for manylinux,
  `pypa/gh-action-pypi-publish` replacing twine (PEP 639 compatibility).
- Python version pins corrected (`>=0.15.0,<0.16`).
- TypeScript golden test paths updated for renamed directory.
- Python bindings crate name ambiguity resolved via dependency rename.
- Benchmarks `PipelineConfig` updated for `compute_hash` field.

## [0.14.0] - 2026-04-17

First public release on [crates.io](https://crates.io/crates/tensogram-core)
and [PyPI](https://pypi.org/project/tensogram/).

### Added
- **crates.io publishing** — 10 Rust crates published with full package
  metadata (license, description, repository, homepage, documentation,
  readme, keywords, categories, authors, rust-version).
- **PyPI publishing** — 14 Python wheels (Linux x86_64, macOS arm64,
  Python 3.9–3.14 including free-threaded 3.13t and 3.14t).
- **Publish workflows** — `publish-crates.yml` (sequential with index
  polling), `publish-pypi-tensogram.yml` (maturin + pypa/gh-action-pypi-publish),
  `publish-pypi-extras.yml` (xarray + zarr via reusable workflow),
  `release-preflight.yml` (validation).
- **Per-crate READMEs** — all 10 Rust crates and the Python bindings
  package have crate-level README files for crates.io/PyPI.
- **Composite LICENSES.md** — `tensogram-sz3-sys` ships a composite
  license file covering Apache-2.0 (wrapper), Argonne BSD (SZ3), and
  Boost-1.0 (ska_hash).
- **Python extras** — `pip install tensogram[xarray]`,
  `pip install tensogram[zarr]`, `pip install tensogram[all]`.
- **Make-release command** extended with registry publishing steps,
  inter-crate dependency version pin bumping, and excluded-crate tests.

### Changed
- **Edition 2024** — workspace migrated from Rust 2021 to 2024 edition
  (resolver 3).
- **MSRV 1.87** — `rust-version` set across all publishable crates
  (edition 2024 floor + `is_multiple_of` stabilisation).
- **thiserror v2** — `tensogram-sz3` migrated from thiserror v1 to v2
  via workspace inheritance, eliminating duplicate versions in the
  dependency tree.
- **Inter-crate version pins** — all path dependencies now carry
  exact version pins (`version = "=X.Y.Z"`) required by `cargo publish`.

### Fixed
- FFI: narrowed unsafe blocks to only raw-pointer operations.
- FFI: fail hard when cbindgen cannot generate the C header.
- sz3-sys: link OpenMP runtime correctly on all supported platforms.
- Core: gate async-only `lock_state` behind `cfg(feature = "async")`.
- Added Apache 2.0 / ECMWF license headers to remaining source files.

## [0.13.0] - 2026-04-17

### Added
- **Multi-threaded coding pipeline** — caller-controlled `threads: u32`
  option on `EncodeOptions` / `DecodeOptions` (default `0` = sequential,
  identical to 0.12.0).  New module
  `tensogram::parallel` wraps a scoped rayon pool, resolves the
  effective budget from the option or the `TENSOGRAM_THREADS`
  environment variable, and dispatches along one of two axes:
  - **Axis B (preferred)** — intra-codec parallelism for `blosc2`
    (`CParams/DParams::nthreads`), `zstd` (`NbWorkers`),
    `simple_packing` (byte-aligned chunked encode/decode, including
    non-byte-aligned bit widths via `lcm(8,bpv)` chunks), and
    `shuffle`/`unshuffle`.
  - **Axis A (fallback)** — `par_iter` across objects via rayon,
    only when every object uses a non-axis-B-friendly codec, so
    the total thread count never exceeds the caller's budget.
  Transparent codecs (`none`, `lz4`, `szip`, `zfp`, `sz3`,
  `simple_packing`, `shuffle`) produce byte-identical encoded
  payloads across all `threads` values; opaque codecs (`blosc2`,
  `zstd` with workers) round-trip losslessly but may reorder
  compressed blocks by completion order.
- **Python/FFI/C++ bindings** gain `threads` parameters on every
  encode/decode entry point plus `TensogramFile.decode_message`,
  async variants, batch variants, and `StreamingEncoder`.  Python
  bindings keep the GIL release behaviour.
- **CLI** — global `--threads N` flag (env `TENSOGRAM_THREADS`
  fallback) on every subcommand; decode-heavy commands
  (`merge`, `split`, `reshuffle`, `convert-grib`,
  `convert-netcdf`) honour it; metadata-only commands
  (`info`, `ls`, `get`, `dump`, `copy`, `set`) ignore it.
- **`threads-scaling` benchmark** binary
  (`rust/benchmarks/src/bin/threads_scaling.rs`) sweeping seven
  representative codec combinations across a user-configurable
  thread budget, reporting speedup vs `threads=0`.
- **New docs page** `docs/src/guide/multi-threaded-pipeline.md`
  covering option semantics, axis-A/B policy, determinism contract,
  env-var precedence, free-threaded Python interaction, and tuning
  recommendations.  Benchmark results page extended with a
  Threading Scaling section.
- **Examples 16** in `examples/rust/src/bin/` and
  `examples/python/` demonstrating both the byte-identity
  invariant for transparent pipelines and the lossless
  round-trip for opaque pipelines across thread counts.
- **Determinism tests** at every layer: Rust integration suite
  (`rust/tensogram/tests/threads_determinism.rs`, 7 tests),
  Python (`python/tests/test_threads.py`, 12 tests), per-codec
  unit tests (`blosc2_nthreads_round_trip_lossless`,
  `zstd_nb_workers_round_trip_lossless`,
  simple_packing aligned + generic byte-identity, shuffle/unshuffle
  byte-identity), and C++ GoogleTest coverage.

### Changed
- **Version bumped to 0.13.0** across all Rust crates, Python
  packages, C++ headers, and the top-level `VERSION` file.
- New cargo feature `threads` (default-on native, off on
  `wasm32`) in both `tensogram` and `tensogram-encodings`
  controlling the rayon dependency.
- `zstd` crate gains the `zstdmt` feature on the workspace
  dependency so libzstd is built with thread support.
- `clap` workspace dependency gains the `env` feature so the CLI
  `--threads` flag can read `TENSOGRAM_THREADS` automatically.
- `PipelineConfig` gains an `intra_codec_threads: u32` field.
- FFI signatures gain a `threads` parameter — this is an ABI
  break from 0.12.0, but downstream code that used option-struct
  defaults will pick it up naturally.
- **TypeScript wrapper** (`@ecmwf/tensogram` under `typescript/`)
  — ergonomic TypeScript bindings over the existing WebAssembly
  crate, shipped as a separate npm package. Includes a full Vitest
  suite (smoke, init, encode, decode, metadata, streaming, errors,
  dtype, file, property-based, and cross-language golden parity
  tests) and the same 0.13.0 version as the rest of the workspace.
- **Remote error class** — `TensogramError::Remote` gets a dedicated
  string in the C FFI error formatter and a corresponding
  `remote_error` exception class in the C++ wrapper. Previously a
  remote I/O failure surfaced as the generic `unknown error`.
- **Documentation reorganisation** —
  - `ARCHITECTURE.md` moved under `plans/`; README and `CONTRIBUTING.md`
    links updated; `plans/ARCHITECTURE.md` rewritten against the current
    11-crate workspace, opt-in crates, separate Python packages, and
    full feature-gate tables.
  - `plans/DONE.md` rewritten as a version-agnostic implementation-path
    log with an explicit instruction that agents must not add version
    numbers or fixed test counts to that file.
  - `plans/TEST.md` replaced with a shape-over-counts coverage
    description.
  - `plans/IDEAS.md` cleaned of items that have already shipped.
  - `plans/BRAINSTORMING.md` added — a long-form brainstorm of
    potential future directions for Tensogram as a general-purpose
    scientific tensor format (protocol extensions, lineage/signing,
    learned compression, conformance test suite, ecosystem bindings,
    and more).
  - `docs/src/introduction.md`, `CLAUDE.md`, `CONTRIBUTING.md`
    refreshed to reflect the current crate and package names.
  - README gains a documentation badge and an online docs link.
  - mdBook pages (wire-format tables, mermaid diagram colours) fixed
    for readability.
  - A docs fact-check pass corrected multiple API signatures
    (`EncodeOptions`, `DecodeOptions`, `decode_range`,
    `simple_packing`, `shuffle`, CLI usage) against the actual source.
- **Apache 2.0 licence metadata** tightened across the repository.

## [0.12.0] - 2026-04-16

### Added
- **Producer metadata dimension hints** — xarray backend resolves dimension
  names from `_extra_["dim_names"]` embedded by the writer. Supports two
  formats: list (axis-ordered, handles same-size axes) and dict (size→name,
  legacy). Resolution priority: user `dim_names` > coord matching > producer
  hints > `dim_N` fallback. (PR #34, @HCookie)
- **Open source preparation** — Apache 2.0 licence headers on all 203 source
  files, `THIRD_PARTY_LICENSES` audit (166/166 clean), `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, PR template with CLA, ECMWF Support Portal link in README,
  branch protection on `main`. (PR #35)
- **Clean-room `tensogram-sz3-sys`** (Apache-2.0 OR MIT) — replaces the
  GPL-licensed `sz3-sys` crate with a thin C++ FFI shim wrapping the
  BSD-licensed SZ3 library. Zero GPL code in the dependency tree.
- **`tensogram-sz3`** (Apache-2.0 OR MIT) — high-level SZ3 API matching the
  published `sz3` crate interface.
- **Docker CI image** — multi-stage Dockerfile with all build deps pre-baked.
  CI split into parallel lint/test/python/C++ jobs. (PR #36)
- **Top-level `Makefile`** — `make rust-test`, `make python-test`,
  `make cpp-test`, `make lint`, `make fmt`, `make docs-build`, `make clean`.
- **30 edge case tests** — NaN/Inf/-0.0 bit-exact float round-trip, bitmask
  validation, 100-object stress, mixed streaming+buffered files, unicode
  metadata (emoji/CJK/Arabic), Python bool→CBOR, xarray single-element and
  all-NaN, decode_range zero-count and overlapping.
- **92 code coverage tests** — tensogram-sz3 (34), validate (10), decode (21),
  file (18), compression (9). Rust coverage 92.6% → 93.4%.

### Changed
- **Repository restructured** — language-grouped layout: `rust/` (all crates
  + benchmarks), `python/` (bindings + xarray + zarr + tests), `cpp/`
  (headers + CMake + tests). Workspace `Cargo.toml` and `Cargo.lock` stay at
  root. (PR #37)
- All copyright headers updated from `2024-` to `2026-`.
- 2 library panics in `tensogram-sz3` replaced with `Result` propagation
  (`UnsupportedAlgorithm`, `UnsupportedErrorBound`).
- Published `sz3`/`sz3-sys` crates removed from dependency tree.

### Fixed
- Bitmask data length now validated at encode time (`ceil(shape_product/8)`).
- `compute_strides` overflow guard in Python bindings (checked_mul).
- FFI `tgm_scan_entry` OOB returns sentinel + error instead of silent zero.
- FFI `collect_data_slices` avoids `from_raw_parts(null, 0)` UB.
- SZ3 FFI shim: `using namespace SZ3` for GCC compatibility.
- Unused `GetOptions`/`GetRange` imports removed from `remote.rs`.
- Dead `get_suffix()` method removed from `remote.rs`.
- Stale hardcoded test counts removed from `CONTRIBUTING.md`.

### Removed
- `ZARR_RESEARCH.md`, `plans/FREE_THREADED_PYTHON.md`,
  `plans/python-async-bindings*.md` (implemented and shipped).
- GPL-licensed `sz3-sys` dependency.

### Stats
- 1,848+ total tests (920 Rust + 423 Python + 201 xarray + 224 zarr +
  80+ C++) — all green
- 0 clippy warnings, 0 ruff issues
- 166/166 third-party dependencies Apache-2.0 compatible

## [0.11.0] - 2026-04-15

### Added
- **Async Python bindings** — new `AsyncTensogramFile` class for use with
  Python `asyncio`. All decode methods return coroutines composable with
  `asyncio.gather()` for true I/O concurrency. Built on
  `pyo3-async-runtimes::tokio` bridging Rust futures to Python coroutines.
- **Batched remote I/O** — `decode_object_batch` and `decode_range_batch`
  (sync and async) combine multiple message decodes into a single HTTP
  request. Achieves O(1) HTTP round-trips for data and O(2) for layout
  discovery regardless of batch size.
- **Layout prefetching** — `prefetch_layouts` / `prefetch_layouts_async`
  pre-warms layout metadata for N messages in ≤2 HTTP calls.
- **Async context manager and iteration** — `async with` and `async for`
  support on `AsyncTensogramFile`.
- **78 async Python tests** — covering all async API methods, sync/async
  parity, error paths, cancellation, iteration, batching, and local-file
  error assertions for batch methods.
- **Async documentation** — new "Async API" section in Python API guide
  with `asyncio.gather` patterns, batch decode examples, sync-vs-async
  decision table.
- **Async example** — `examples/python/15_async_operations.py`.
- **`.claude/commands/` slash commands** — 7 project workflow commands
  extracted from CLAUDE.md: `make-further-pass`, `improve-error-handling`,
  `improve-edge-cases`, `improve-code-coverage`, `prepare-make-pr`,
  `address-pr-comments`, `make-release`.
- **AGENTS.md** symlink to CLAUDE.md for cross-tool compatibility.

### Changed
- All async file methods changed from `&mut self` to `&self`, enabling
  `Arc<TensogramFile>` sharing across concurrent futures.
- `AsyncTensogramFile` uses `Arc<TensogramFile>` internally — no
  Python-level mutex, truly concurrent async operations.
- `__len__` on `AsyncTensogramFile` requires prior `await message_count()`
  call; raises `RuntimeError` if count not yet known (no hidden blocking).
- Removed unnecessary `desc.clone()` in decode paths — move instead of copy.

### Stats
- 1,695+ total tests (870 Rust + 412 Python + 189 xarray + 224 Zarr) — all green
- 0 clippy warnings, 0 ruff warnings

## [0.10.0] - 2026-04-14

### Added
- **Remote object store backend** — read `.tgm` files directly from S3, GCS,
  Azure, and HTTP(S) URLs without downloading the whole file. Range-request
  based message scanning (1 GET per message), layout caching, footer-indexed
  file support, and checked arithmetic for all remote-supplied lengths.
  New `remote` feature gate with `object_store`, `tokio`, `bytes`, `url` deps.
- **Free-threaded Python support (3.13t / 3.14t)** — all hot paths release
  the GIL (`py.allow_threads`). Module declared `gil_used = false`. PyO3
  upgraded 0.25 → 0.28. Buffer handling changed from `&[u8]` to
  `PyBackedBytes` for safe GIL-free access. Passes 23 concurrent thread-safety
  tests.
- **Remote Python API** — `TensogramFile.open_remote(url, storage_options=None)`,
  `file_decode_metadata()`, `file_decode_descriptors()`, `file_decode_object()`,
  `file_decode_range()`, `is_remote_url()`, `is_remote()`, `source()`.
- **Remote xarray backend** — `open_datatree()` / `open_dataset()` accept
  remote URLs via `storage_options`. Lazy chunk reads for remote zarr stores.
  Context-manager file handles with `set_close` callbacks for deterministic
  cleanup.
- **Remote zarr backend** — `TensogramStore` accepts remote URLs with
  `storage_options` and write rejection. Lazy message scanning for remote
  files.
- **Remote documentation** — `docs/src/guide/remote-access.md` (request
  budgets, error handling, limitations) and
  `docs/src/guide/free-threaded-python.md` (benchmark results, usage guide).
- **Remote examples** — `examples/python/14_remote_access.py` and
  `examples/rust/src/bin/14_remote_access.rs`.
- **164 new coverage tests** — 71 FFI round-trip tests, 30 validate
  adversarial tests, 22 remote async parity tests, 17 file API tests,
  11 decode edge-case tests, 13 Python coverage-gap tests.
- **Threading benchmarks** — `bench_threading.py` (multi-threaded scaling)
  and `bench_vs_eccodes.py` (comparison against ecCodes).

### Changed
- **`TensogramFile` is now `&self`** — all read methods changed from
  `&mut self` to `&self` using `OnceLock` for cached offsets (thread safety).
  `Backend` enum (`Local` | `Remote`) replaces raw `PathBuf` + `Option<Mmap>`.
  `path()` returns `Option<&Path>` (remote files have no local path).
- **xarray `BackendArray` refactored** — replaced `xarray._data` private
  traversal with explicit array tracking. `set_close` callbacks for
  deterministic cleanup.
- **CI switched to self-hosted runners** — all jobs use
  `platform-builder-docker-xl` with explicit LLVM install, `noexec /tmp`
  handling, and disk cleanup steps.
- Removed unreachable dead code guards in `xarray/mapping.py` and
  `zarr/mapping.py`.

### Stats
- 1,545+ total tests (870 Rust + 523 Python + 224 Zarr) — all green
- Rust coverage: 90.7% (up from 83.0%)
- Python coverage: 97–99%
- 0 clippy warnings, 0 fmt diffs

## [0.9.1] - 2026-04-11

### Added
- **Python validation bindings** — `tensogram.validate(buf, level, check_canonical)`
  and `tensogram.validate_file(path, level, check_canonical)` return plain dicts
  with issues, object count, and hash verification status. Four validation levels:
  quick (structure), default (integrity), checksum (hash-only), full (fidelity + NaN/Inf).
- **C FFI validation bindings** — `tgm_validate()` and `tgm_validate_file()` return
  JSON via `tgm_bytes_t` out-parameter, matching the existing ABI pattern. NULL level
  defaults to "default"; NULL buf with buf_len=0 is valid for empty-buffer validation.
- **C++ validation wrapper** — `tensogram::validate()` and `tensogram::validate_file()`
  return JSON strings with typed exception mapping (`invalid_arg_error`, `io_error`,
  `encoding_error`).
- **Python API guide** — new `docs/src/guide/python-api.md` mdBook page covering
  the full encoding pipeline (all compressors, filters, simple packing), decoding
  (full, selective, range, scan, iter), file API, streaming encoder, validation,
  error handling, and dtype reference.
- **Validation examples** — `examples/python/13_validate.py` and
  `examples/rust/src/bin/13_validate.rs`.
- **34 Python validation tests** — all levels, canonical combos, hash/no-hash,
  NaN/Inf detection, file edge cases (garbage, truncation, gaps).
- **12 FFI validation tests** — option parsing + end-to-end (null guards, empty
  buffer, invalid level, missing file).
- **11 C++ GoogleTest validation tests** — wrapper→FFI→Rust chain, exception
  mapping, all levels, file validation.

### Changed
- C header doc comments now use C enum names (`TGM_ERROR_OK` not `TgmError::Ok`).
- `tgm_bytes_t` output documented as not NUL-terminated (use `out->len`).
- `PYTHON_API.md` removed from repo root; replaced by mdBook guide page.
- `bfloat16` doc updated: returned as `ml_dtypes.bfloat16` when available,
  `np.uint16` fallback.

## [0.9.0] - 2026-04-10

### Added
- **Native byte-order decode** — decoded payloads are now returned in the
  caller's native byte order by default.  `DecodeOptions.native_byte_order`
  (default `true`) controls this across all interfaces: Rust, Python, C FFI,
  C++.  Users never need to inspect `byte_order` or manually byteswap; a
  simple `from_ne_bytes()` or `data_as<T>()` is always correct.
  Set `native_byte_order=false` to get the raw wire-order bytes.
- `Dtype::swap_unit_size()` — returns the swap granularity for each dtype
  (handles complex64/complex128 correctly by swapping each scalar component).
- `ByteOrder::native()` — compile-time detection of the platform's byte order.
- `byteswap()` utility — public in-place byte reversal by element width.
- **`tensogram validate`** — CLI command and library API for checking `.tgm`
  file correctness (3 validation levels: structure, metadata, integrity).
  Includes `--quick`, `--checksum`, `--canonical`, `--json` modes and
  ~40 stable `IssueCode` variants with serde serialization.
- **`tensogram-wasm`** — browser decoder via wasm-bindgen with zero-copy
  TypedArray views, streaming decoder, and full encode/decode API.
  Supports lz4, szip (pure-Rust), and zstd (pure-Rust) codecs.
- **`tensogram-szip`** — pure-Rust CCSDS 121.0-B-3 AEC/SZIP codec
  (encode, decode, range-decode). Drop-in replacement for libaec in
  environments without C FFI (e.g., WebAssembly).
- **Runtime compression backend dispatch** — szip and zstd can have both
  FFI and pure-Rust backends compiled simultaneously. Selection via
  `TENSOGRAM_COMPRESSION_BACKEND=pure` env var; WASM defaults to pure.
- **WASM CI job** — `wasm-pack test --node` runs 134 wasm-bindgen tests
  on every PR.
- Comprehensive szip test suite: 154 tests including stress tests (all
  bit widths 1-32), property-based tests (proptest), error path tests,
  libaec parity tests, and FFI cross-checks.

### Changed
- ZFP and SZ3 lossy compressors are now byte-order-aware: decompressed output
  is written in the wire byte order declared in the descriptor, making the
  pipeline's native-endian conversion step uniform across all codecs.
- Python `byte_order` default changed from `"little"` to native (compile-time).
- C FFI decode functions (`tgm_decode`, `tgm_decode_object`, `tgm_decode_range`,
  `tgm_file_decode_message`, `tgm_object_iter_create`) gain a
  `native_byte_order` parameter.
- CLI `reshuffle`, `merge`, `split`, `set` commands use wire byte order
  on decode to preserve byte layout when re-encoding.

### Removed
- Zarr store read-path manual byteswap workaround — no longer needed.

## [0.8.0] - 2026-04-08

### Added
- **`tensogram-netcdf` crate** — NetCDF → Tensogram converter supporting
  NetCDF-3 classic and NetCDF-4 files via `libnetcdf`. Preserves all
  variable and file attributes, unpacks `scale_factor` / `add_offset`,
  handles fill values, and skips unsupported types with warnings.
  Excluded from the default workspace build because it requires
  `libnetcdf` at the OS level.
- **`tensogram convert-netcdf` CLI subcommand** — gated behind a new
  `netcdf` feature of `tensogram-cli`. Flags: `--output`,
  `--split-by {file,variable,record}`, `--cf`, plus the shared
  encoding pipeline flags. `--split-by=record` walks the unlimited
  dimension and replicates static variables into every record message.
- **Shared `tensogram::pipeline` module** — single source of truth
  for `DataPipeline` and `apply_pipeline()`. Both `tensogram-grib` and
  `tensogram-netcdf` re-export `DataPipeline` and delegate to the same
  helper, so the `--encoding/--bits/--filter/--compression/--compression-level`
  flags produce byte-identical descriptor fields on both converters.
  Supported values: `simple_packing` + `shuffle` +
  `zstd`/`lz4`/`blosc2`/`szip`.
- **CF metadata mapping behind `--cf`** — curated 16-attribute
  allow-list (`standard_name`, `long_name`, `units`, `calendar`,
  `cell_methods`, `coordinates`, `axis`, `positive`, `valid_min`,
  `valid_max`, `valid_range`, `bounds`, `grid_mapping`,
  `ancillary_variables`, `flag_values`, `flag_meanings`) stored under
  `base[i]["cf"]`. Full verbose attribute dump still available under
  `base[i]["netcdf"]` regardless.
- **`convert-grib` now accepts the shared pipeline flags** —
  previously they were parsed by clap but discarded. The new path
  wires them through `build_data_object()` via the shared
  `apply_pipeline` helper.
- **mdBook docs** — new `docs/src/guide/convert-netcdf.md` user guide
  and `docs/src/reference/netcdf-cf-mapping.md` CF attribute reference;
  full converter error taxonomy added to
  `docs/src/guide/error-handling.md`.
- **Examples** — `examples/python/12_convert_netcdf.py` (CLI via
  `subprocess`, the v1 pattern since the Python bindings do not
  expose `convert_netcdf_file` directly) and
  `examples/rust/src/bin/12_convert_netcdf.rs` (direct library API,
  gated behind a new `netcdf` feature on the examples crate).
- **Python end-to-end tests** — `tests/python/test_convert_netcdf.py`
  with 8 round-trip tests covering simple f64, packed int16, CF
  lifting, split modes, zstd compression, and the record-split error
  path.
- **CI `netcdf` job** — Ubuntu + macOS matrix running clippy + netcdf
  crate tests + CLI tests + example build. `grib` job extended to the
  same matrix for symmetry.
- **Clap `PossibleValuesParser`** on `--encoding`, `--filter`, and
  `--compression` — invalid values fail fast at arg-parse time with a
  "did you mean?" suggestion instead of propagating into the
  converter as an `InvalidData` error at run time.
- **17 new `tensogram::pipeline` unit tests** covering every
  encoding/filter/compression stage, default pass-through, NaN skip,
  non-f64 skip, unknown-codec errors, and shuffle element-size
  derivation for both raw and simple-packed payloads.
- **Metadata module unit tests** — 24 unit tests in
  `tensogram-netcdf/src/metadata.rs` exhaustively covering every
  `AttributeValue` → `CborValue` mapping, including a regression
  test for `u64` values above `i64::MAX` (ciborium's native
  `From<u64>` path, avoiding wrap-around).

### Changed
- **`convert-grib` pipeline flags are now honoured** — before this
  release the `--encoding`/`--bits`/`--filter`/`--compression` flags
  were parsed and silently dropped. Default remains `none/none/none`
  so existing `convert-grib` invocations produce byte-identical output.
- **`DataPipeline` now lives in `tensogram::pipeline`** — re-exported from `tensogram_grib` and `tensogram_netcdf` so existing `use tensogram_{grib,netcdf}::DataPipeline` callers keep
  compiling. The ~150 lines of previously-duplicated `apply_pipeline`
  logic in the two converters are now a single helper.
- **`tensogram-netcdf` panic-free audit** — zero `unwrap`/`expect`/
  `panic!` in library code. `metadata::attr_value_to_cbor` gained
  an exhaustive match over all 22 `netcdf::AttributeValue` variants
  (no `Option` wrapper; match exhaustiveness catches upstream drift
  at compile time).
- **Warning-on-drop metadata reads** — `extract_var_attrs` /
  `extract_cf_attrs` / `extract_global_attrs` now emit stderr
  warnings when an attribute can't be read, instead of silently
  dropping it.
- **CI `python` job** — now installs libnetcdf/hdf5 + netCDF4 and
  runs the new Python e2e tests against a feature-gated `tensogram`
  binary.

### Fixed
- `tensogram-grib/tests/integration.rs` — replaced
  `.expect(&format!(...))` with `.unwrap_or_else(|| panic!(...))` to
  clear a pre-existing `clippy::expect_fun_call` lint that surfaced
  under stricter review.

### Stats
- 69 `tensogram-netcdf` tests (44 integration + 24 unit + 1 doctest)
- 124 `tensogram-cli` tests with `--features netcdf`
- 17 new `tensogram::pipeline` unit tests
- 8 Python end-to-end round-trip tests
- 630 workspace tests, 271 Python tests, 124 C++ tests — all green
- 95.54% region / 94.80% line coverage on `tensogram-netcdf`
- 0 clippy warnings, 0 fmt diffs

## [0.7.0] - 2026-04-08

### Fixed
- **szip 24-bit data corruption** — `AEC_DATA_3BYTE` is now auto-set in `effective_flags()` for 17-24 bit samples, so libaec reads 3-byte-packed data correctly. Decoded values previously had ±60 max error; now match quantization step (~1.9×10⁻⁶ at 24 bits).
- **szip byte-order mismatch** — `AEC_DATA_MSB` is now set when the upstream encoding is `SimplePacking` (which produces MSB-first bytes). libaec's predictor now sees the correct byte significance order; compression ratio on 24-bit GRIB data now matches ecCodes (~27%).
- **Benchmark `AEC_DATA_PREPROCESS` constant** — was 1 (`AEC_DATA_SIGNED`), now correctly 8. Benchmarks were running without the preprocessor step.

### Added
- `simple_packing::encode_pipeline_f64()` — typed-input variant that avoids the bytes→f64 round-trip allocation
- Benchmark fidelity validation: lossless paths checked for exact round-trip; lossy paths report Linf, L1, and L2 (RMSE) norms
- Benchmark structured error handling: `BenchmarkError` enum, `BenchmarkRun` struct, non-zero exit on failures
- `--warmup` flag (default 3) and raised default iteration count from 5 to 10
- Throughput (MB/s) reporting and compressed-size variability detection in benchmarks
- Rewritten benchmark documentation with split tables (lossless / SimplePacking / lossy), human-readable method names, sizes in MiB, and fidelity norms explained

### Changed
- `simple_packing` encode is ~2.5× faster for typical SimplePacking cases:
  - Fused NaN + min + max into a single pass in `compute_params` (was 3 passes)
  - Precomputed `scale = 10^D × 2^(-E)` — eliminates per-value f64 division
  - Specialized `encode_aligned<N>` / `decode_aligned<N>` loops for 8/16/24/32-bit widths
  - Removed redundant NaN check from `encode()`
- Benchmark methodology page cleaned of internal jargon — no Rust API names, no C function signatures
- GRIB comparison timing is now symmetric end-to-end (both ecCodes and Tensogram include parameter setup)

## [0.6.0] - 2026-04-06

### Changed (BREAKING)
- **Metadata major refactor** — `GlobalMetadata` fields `common` and `payload` removed; replaced by `base` (per-object metadata array where each entry is independent and self-contained)
- **CBOR key renames** — `reserved` → `_reserved_`, `extra` → `_extra_` on the wire
- **Python API** — `meta.common` and `meta.payload` replaced by `meta.base` and `meta.reserved`; `meta.extra` now maps to `_extra_` in CBOR
- Auto-populated tensor metadata (ndim/shape/strides/dtype) now lives under `base[i]["_reserved_"]["tensor"]`

### Added
- `encode_pre_encoded()` API for advanced callers (e.g., GPU pipelines) across Rust, Python, C FFI, and C++ — bypasses the encoding pipeline for already-encoded payloads
- `StreamingEncoder::write_object_pre_encoded()` for streaming pre-encoded objects
- `compute_common()` utility for extracting shared keys from `base` entries in software
- Encoder validates that client code does not write to `_reserved_` at any level
- Preceder Metadata Frames (type 8) for streaming per-object metadata

### Stats
- 1008 total tests (283 Rust + 226 Python + 181 xarray + 204 Zarr + 105 C++ + 7 GRIB new + 2 streamer)

## [0.5.0] - 2026-04-06

### Added
- **Project logo** — centered at top of README with badge
- **82 new Python coverage tests** — metadata properties, error paths, descriptor coverage,
  decode_range across all 10 dtypes, file slice edge cases, concurrent iterators,
  iter_messages edges, Message unpacking across all 5 decode paths, scan edges,
  big-endian round-trips

### Fixed
- Replaced `unwrap()` in FFI decode macro with `Result` propagation (panic=abort safety)
- Fixed TOCTOU race condition in Python file iterator initialization
- Added buffer copy warning to `iter_messages()` docstring
- Python examples 01-03, 05-06 rewritten to use real API
- Removed binary `.so` build artifact, added `*.so` to `.gitignore`

### Stats
- 888 total tests (283 Rust + 200 Python + 124 xarray + 172 Zarr + 109 C++)

## [0.4.0] - 2026-04-05

### Added
- **tensogram-xarray** — xarray backend engine for `.tgm` files (`engine="tensogram"`)
  - Lazy loading via `BackendArray` with N-D slice-to-flat-range mapping
  - Coordinate auto-detection by name matching (13 known names)
  - User-specified dimension mapping (`dim_names`) and variable naming (`variable_key`)
  - Multi-message auto-merge via `open_datasets()` with hypercube stacking
  - `StackedBackendArray` for lazy hypercube composition
  - Ratio-based `range_threshold` heuristic (default 0.5) for partial vs full decode
  - 113 tests, 97% line coverage
- **tensogram-zarr** — Zarr v3 Store backend for `.tgm` files
  - Read/write/append modes via `TensogramStore`
  - `zarr.open_group(store=TensogramStore.open_tgm("file.tgm"), mode="r")` for standard Zarr API access
  - 14 numeric dtypes mapped bidirectionally (TGM <-> Zarr v3 <-> NumPy)
  - Variable naming from MARS metadata with dedup and sanitization
  - Byte-range support (RangeByteRequest, OffsetByteRequest, SuffixByteRequest)
  - 172 tests
- **Python iterator protocol** — `TensogramFile` now supports standard Python iteration
  - `for msg in file:` iterates all messages (owns independent file handle, free-threaded safe)
  - `file[i]`, `file[-1]` — index by position (negative indexing)
  - `file[1:10:2]` — slice returns list of Message namedtuples
  - `iter_messages(buf)` — iterate decoded messages from a byte buffer
  - `Message` namedtuple — `.metadata` and `.objects` fields, tuple unpacking
- **`decode_descriptors(buf)`** — parse metadata + per-object descriptors without decoding payloads (Rust, Python, C, C++)
- **`meta.base`**, **`meta.reserved`**, and **`meta.extra`** getters in Python bindings
- **float16/bfloat16/complex** Python support — proper typed numpy arrays (`ml_dtypes.bfloat16` if installed)

### Changed
- **`decode_range` API (BREAKING)** — now returns split results by default (one buffer per range). `join` parameter opts into concatenated output. Affects Rust, Python, C, and C++ APIs.
- **`decode()` and `decode_message()`** now return `Message` namedtuple (supports both attribute access and tuple unpacking — backward compatible)

### Fixed
- Removed binary `.so` build artifact from Python package
- Replaced `unwrap()` in FFI decode macro with proper error propagation
- Fixed TOCTOU race condition in Python file iterator initialization
- Python examples 01-03, 05-06 rewritten to use real API (were placeholder code)

### Stats
- 888 total tests (283 Rust + 200 Python + 124 xarray + 172 Zarr + 109 C++)

## [0.3.0] - 2026-04-04

### Added
- **Python bindings quality overhaul** — all 10 numeric numpy dtypes natively accepted/decoded
- **`TensogramFile` Python** — context manager (`with ... as f:`), `len(f)`, `"key" in meta`
- **95 Python tests** — parametrized dtype round-trips, multi-object, multi-range, big-endian, wire determinism
- **ruff** configured as Python linter/formatter (0 warnings)

### Fixed
- `decode_range` validates byte count vs expected elements (was silently truncating)
- Bitmask dtype fallback no longer produces empty array
- `byte_order` rejects unknown values (was silently defaulting)
- Safe i128-to-i64 bounds check for CBOR integers
- Flaky golden file test race condition fixed (tests now read-only)
- `cargo doc` and `cargo fmt` CI failures resolved

### Performance
- `from_slice` zero-copy for u8/i8 numpy decode (eliminates allocation)

### Stats
- 262 total tests (167 Rust + 95 Python), 0 clippy warnings

## [0.2.0] - 2026-04-04

### Added
- **Streaming API** — `StreamingEncoder<W: Write>` for progressive encode without buffering
- **Metadata structure** — `GlobalMetadata` now has `common`, `payload`, `reserved` CBOR sections (backwards-compatible)
- **CLI merge** — `tensogram merge` combines messages from multiple files
- **CLI split** — `tensogram split` separates multi-object messages into individual files
- **CLI reshuffle** — `tensogram reshuffle` converts streaming-mode to random-access-mode
- **GRIB converter** — `tensogram-grib` crate with ecCodes FFI for GRIB-to-Tensogram conversion
- **CLI convert-grib** — `tensogram convert-grib` subcommand (feature-gated behind `grib`)
- **Feature-gated compression** — all 6 codecs (szip, zstd, lz4, blosc2, zfp, sz3) are optional features (default on)
- **Streaming example** — `examples/rust/src/bin/11_streaming.rs`
- **GRIB docs** and **CLI docs** — mdbook pages for conversion and new commands

### Changed
- `README.md` shortened from 302 to 100 lines; detailed content moved to mdbook docs

### Removed
- **md5 and sha1 hash support** — only xxh3 is supported; unknown hash types return a clear error

### Stats
- 170 Rust tests, 0 clippy warnings

## [0.1.0] - 2026-04-04

Initial release of Tensogram, a binary N-Tensor message format library for scientific data.

### Core
- Encode and decode N-dimensional tensors with self-describing CBOR metadata
- Pack multiple tensors per message, each with own shape, dtype, and encoding pipeline
- 6 compression codecs per data object: szip (CCSDS), zstd, lz4, blosc2, zfp (lossy float), sz3 (error-bounded lossy)
- GRIB-style simple packing for lossy quantization (0-64 bits per value)
- xxh3 per-object payload hashing for integrity verification
- 15 data types: float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask

### File I/O
- `TensogramFile` with lazy scanning, O(1) random-access, and message append
- Memory-mapped I/O via `mmap` feature (zero-copy reads)
- Async file operations via `async` feature (tokio)

### Language Bindings
- Rust native API
- C FFI layer with auto-generated `tensogram.h` header (62 functions)
- Python bindings via PyO3 with NumPy integration

### CLI
- `tensogram info`, `ls`, `dump`, `get`, `set`, `copy` subcommands
- Where-clause filtering (`-w`), key selection (`-p`), JSON output (`-j`)

### Wire Format (v2)
- Frame-based message structure: Preamble + typed frames + Postamble
- Streaming support with `total_length=0` and footer-based index
- Deterministic CBOR encoding (RFC 8949 section 4.2 canonical key ordering)
- Corruption recovery via magic boundary detection

### Quality
- 157 tests across 5 workspace crates, 0 clippy warnings
- Golden binary test files for cross-language verification

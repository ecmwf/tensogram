# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed — PyPI Linux wheels are installable on glibc ≥ 2.28

Linux wheels published to PyPI are tagged `manylinux_2_28` for both
x86_64 and aarch64 and are installable on distributions with
glibc ≥ 2.28 (RHEL/AlmaLinux 8+, Debian 10+, Ubuntu 18.10+, Fedora 29+).
The publish workflow builds each wheel inside the corresponding
`quay.io/pypa/manylinux_2_28_<arch>` image via `PyO3/maturin-action`
for all six supported interpreters: `cp311`, `cp312`, `cp313`,
`cp314`, `cp313t`, `cp314t`.  A tag-verification step in the workflow
confirms the produced wheels carry the expected platform tag before
artifact upload.

### Fixed — `regular_ll` longitude convention

GRIB-derived `.tgm` files from ECMWF open-data (where the grid scans
from the dateline eastwards, `longitudeOfFirstGridPointInDegrees =
180`) rendered 180° off in Tensoscope — Sahara appeared where the
Pacific should be and vice versa.  Root cause: the converter lifted
only the ecCodes `mars` namespace, which does not include the grid's
corner coordinates, so the viewer had to guess the longitude
convention and guessed the wrong one.

- **`tensogram-grib`** now lifts the four `regular_ll` corner-point
  keys from the `geography` namespace and emits a canonical
  `mars.area = [N, W, S, E]` alongside `mars.grid`.  A new pure
  helper `compute_regular_ll_area` handles the one unambiguous
  normalisation (full-global dateline-first → `west = -180`) and
  refuses everything else it cannot express as a monotone
  `[N, W, S, E]` tuple (non-standard scan modes,
  dateline-crossing regional subdomains).  Non-`regular_ll` grids
  (`reduced_gg`, octahedral `O*`, Gaussian `N*`) are unchanged.
- **Tensoscope** reads `mars.area` when present.  For legacy files
  without it, the inference fallback is now formalised as a named
  `DEFAULT_REGULAR_LL_AREA` constant (the `[-180, 180]` dateline-
  first default already landed in #86) and emits a one-shot
  `console.warn` the first time the fallback fires, explaining
  how to re-convert for accurate geometry.

## [0.18.1] - 2026-04-23

### Added — Tensoscope: render GRIB-derived `.tgm` files without preprocessing

ECMWF-operational GRIB-derived `.tgm` files (converted via
`tensogram convert-grib`) now open directly in Tensoscope without any
Python-side coordinate preprocessing.  All changes are confined to the
browser layer — no wire-format, CLI, or core-library changes.

- **Auto-expand 1-D lat/lon axes** — files that ship 1-D latitude /
  longitude axes plus 2-D `[nLat, nLon]` data (the common GRIB /
  NetCDF layout) now auto-meshgrid in the viewer instead of requiring
  a Python pre-pass.  Already-meshgridded files pass through
  unchanged.
- **Infer lat/lon axes from `mars.grid`** — files with no explicit
  coordinate objects but `mars.grid = "regular_ll"` now have their
  axes synthesised from the data shape plus `mars.area` (default
  `[90, 0, -90, 360]`), with endpoint-exclusive longitude on full
  360° wrap.  Reduced, octahedral, and Gaussian grids (`reduced_gg`,
  `O*`, `N*`) return `null` and will need per-point coords emitted
  by `tensogram-grib` in a future release.
- **Per-message coordinate cache** — `fetchCoordinates(msgIdx)` is
  now keyed per message, so heterogeneous multi-message files with
  different grids per message each render against the right mesh
  instead of silently reusing message 0's coords.

### Fixed

- **Integer `mars.param` codes** — sidebar and field-selection on
  GRIB-derived files that use ECMWF integer param codes
  (`167`, `130`, …) no longer crash.  `decodeFieldSlice` and
  `groupByParam` coerce `bigint` param values via a safe `toNumber`
  helper.
- **Slicing on truly 2-D gridded data** — `selectField` no longer
  level-slices `[721, 1440]` meshed fields and renders them as a
  "donut in the north"; `decideSliceDim` returns `-1` when
  `total === coordLength`.
- **Slicing on packed-level 3-D fields** — `[N_lev, nLat, nLon]`
  fields with meshed coords slice on dim 0 as expected.
  `decideSliceDim` handles "one or more leading level-like dims
  above the grid" via integer-multiple detection instead of
  per-dim matching.
- **Stale msg-0 coordinates** — `useAppStore.selectField` fetches
  per-message coords and commits them atomically with `fieldData`
  in the same store update; `initViewer` no longer eager-fetches
  msg 0.  Multi-message files with different grids per message now
  render consistently.
- **`mars.area` infinity guard** — `toNumber()` rejects `bigint`
  values that overflow to `Infinity` via `Number(v)`; pathological
  areas fall back to safe defaults instead of propagating
  `Infinity` into generated lat/lon arrays.
- **Map flash / reset on file open** — eliminated the visible
  reset of camera and styling when loading a new file (#84).

### CI / Infrastructure

- **Tensoscope CI** — `.github/workflows/ci.yml` now runs
  `make ts-install` + `make ts-build` before `tensoscope/npm ci`, so
  vitest can resolve `@ecmwf.int/tensogram` through
  `file:../typescript`.  Pure-helper tests (`expandAxes`,
  `inferAxesFromMars`, `decideSliceDim`) no longer fail to load.
- **PyPI Linux aarch64 wheels** — `publish-pypi.yml` now builds
  Linux `aarch64` wheels alongside `x86_64` (#83).

### Stats

- Rust workspace: 1505 passed, 5 ignored, 0 failed
- `tensogram` with `remote,async` features: 824 passed, 2 ignored
- `tensogram-grib`: 36 passed
- `tensogram-netcdf`: 69 passed
- Python `tensogram`: 535 passed, 40 skipped
- Python `tensogram-xarray`: 242 passed
- Python `tensogram-zarr`: 235 passed
- Tensoscope (vitest): 54 passed across 7 files
- `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `mdbook build docs/`: clean

## [0.18.0] - 2026-04-23

### Added — TypeScript / WASM / Tensoscope: browser-usable remote + async parity

`@ecmwf.int/tensogram` is now feature-comparable with the Rust core's
`object_store` integration over HTTP(S) and AWS-signed HTTPS.  The
TypeScript wrapper, the WASM crate, and the Tensoscope viewer all
move from "download the whole message" to "fetch only the bytes you
ask for", with bounded-concurrency fan-out tuned to typical browser
per-host limits.

- **Layout-aware per-object access on `TensogramFile.fromUrl`** over
  HTTP Range:
  - `messageMetadata(i)` — 256 KB header chunk (or footer suffix),
    instead of full-message download.
  - `messageDescriptors(i)` — header + footer + CBOR-prefix
    optimisation; full frame for frames ≤ 64 KB, mirrors Rust
    `read_descriptor_only`.
  - `messageObject(i, j)` — exactly one `Range: bytes=…` GET for the
    target object's frame.
  - `messageObjectRange(i, j, ranges)` — partial sub-tensor decode
    against a single fetched frame.
  - `messageObjectBatch` / `messageObjectRangeBatch` — parallel
    fan-out with bounded concurrency.
  - `prefetchLayouts(msgIndices)` — pre-warm the per-message layout
    cache so subsequent metadata / descriptor / per-object reads
    make zero extra round trips.
- **Bounded-concurrency pool** (default 6, configurable via
  `FromUrlOptions.concurrency` and per-call `opts.concurrency`).  The
  outer batch limiter and the shared per-host pool are independent
  so the nested-pool deadlock path stays closed (regression-tested).
- **AWS SigV4 helper**:
  - `signAwsV4Request(input, creds)` — pure signer.  Byte-for-byte
    against AWS `sig-v4-test-suite` vectors (get-vanilla,
    duplicate-query-key ordering, header-value trim, session token,
    pre-encoded path round-trip).  Web Crypto only — no transitive
    deps.
  - `createAwsSigV4Fetch(creds, opts?)` — `fetch`-compatible wrapper
    pluggable directly into `FromUrlOptions.fetch`.  Read-only
    (`UNSIGNED-PAYLOAD` semantics over an empty body); for write
    paths use a presigned URL.
- **WASM additions**: nine new `#[wasm_bindgen]` exports in a new
  `layout.rs` module (`read_preamble_info`, `read_postamble_info`,
  `parse_header_chunk`, `parse_footer_chunk`,
  `read_data_object_frame_header`, `read_data_object_frame_footer`,
  `parse_descriptor_cbor`, `decode_object_from_frame`,
  `decode_range_from_frame`).  These let the TS wrapper compose
  per-object Range reads without re-implementing wire-format
  parsing.  Header / footer chunk parsers route metadata through
  `metadata_to_js` so the lazy backend's `metadata.version` is the
  same wire-format integer the eager backend has always returned.
- **Rust core addition**: one new public function,
  `decode::decode_object_from_frame` (plus
  `decode_range_from_frame`) — both take one frame's bytes in
  isolation, mirroring `decode_object` / `decode_range`.  Used by
  the WASM single-frame helpers and by the `remote` backend when it
  has fetched one indexed frame via Range.
- **Tensoscope** uses `prefetchLayouts` + `messageMetadata` for
  `buildIndex` (chunked at 64 indices to keep the per-host
  concurrency limit honoured) and `messageObject` for `decodeField`
  (one Range GET per displayed field instead of a whole-message
  download).  Misleading `s3://` placeholders in the file-open
  dialogs are replaced with accurate "https:// URL (use a presigned
  URL for private S3/GCS/Azure)" guidance.

### Fixed

- **TypeScript `POSTAMBLE_BYTES`**: `typescript/src/file.ts` had the
  v2 value (16); fixed to v3's 24.  Fixes a silent off-by-8 in the
  lazy-scan `total_length` minimum-size check.
- **TypeScript `messageDescriptors` eager path**: removed a
  `parse_footer_chunk(slice)` call against the *full* message bytes
  whose result was discarded.  The call could throw on valid
  messages where stray `FR` byte sequences appeared inside payloads
  (notably compressed streams), failing `messageDescriptors`
  entirely.  Eager fallback now goes straight to the standard
  `decode(slice)` route.
- **TypeScript `wrapWbgDecodedMessage` lifecycle**: the lazy
  backend's `messageObject` returned `{ ...wrapped, metadata }` —
  spreading into a fresh object that bypassed the
  `FinalizationRegistry` registration on the wrapped object,
  silently leaking the WASM handle whenever the caller forgot to
  call `close()`.  `wrapWbgDecodedMessage` now accepts an optional
  `metadataOverride` argument that's applied in place; the returned
  identity stays stable so GC finalisation still runs.
- **TypeScript AWS SigV4 fetch**: when `input` is a `Request`,
  `createAwsSigV4Fetch` now merges `init.headers` over
  `input.headers` before signing — matching standard `fetch`
  semantics.  Previously a caller passing
  `(request, { headers: { Range: '…' } })` silently lost the Range
  header (and the resulting signature was for a Range-less request).
- **TypeScript descriptor-fetch concurrency cap**: the inner
  `fetchRange` calls inside `#fetchOneDescriptor` (header + footer
  parallel reads, CBOR-region fetch, and the prefix-doubling loop)
  now route through the backend's bounded-concurrency limiter.
  Previously each descriptor task held an outer slot via
  `b.limit(t)` and then fired multiple ungoverned leaf fetches —
  burst exceeded the documented per-host cap.  The outer wrap is
  removed (it would deadlock at low caps when leaves also acquire
  slots) so the limiter applies once at the leaf level.

### Changed — BREAKING (free-form CBOR metadata)

The CBOR metadata frame is now **fully free-form**.  The library-
interpreted top-level keys are just `base`, `_reserved_`, and
`_extra_`; anything else the caller supplies flows into `_extra_`
on decode.  The wire-format version lives exclusively in the
preamble (see `plans/WIRE_FORMAT.md` §3) and is never written to
CBOR.

- **`GlobalMetadata.version` field removed** from the public Rust
  struct.  Constructors that used `GlobalMetadata { version: 3, … }`
  must drop the `version: 3` line; `GlobalMetadata::default()`
  keeps working.  Decoded-side accessors stay:
  - Rust: new `tensogram::WIRE_VERSION` crate-root constant.
  - Python: `tensogram.WIRE_VERSION` module constant; `Metadata.version`
    property still works and now sources from the preamble.
  - TypeScript: new `WIRE_VERSION` export from
    `@ecmwf.int/tensogram`; `metadata.version` on decoded messages
    is synthesised from the preamble.
  - C FFI: `tgm_message_version` / `tgm_metadata_version` retain
    their signatures and now return the preamble-sourced
    `WIRE_VERSION`.
  - C++: `message::version()` / `metadata::version()` unchanged at
    the ABI level; now preamble-sourced.
- **Free-form top-level CBOR keys** — `tensogram.encode({}, …)` and
  `tensogram.encode({"foo": "bar"}, …)` are now valid (previously
  rejected with `ValueError("missing 'version'")`).  A caller who
  still supplies `{"version": N, …}` sees `N` land in
  `_extra_["version"]`; the wire-format version remains 3.
- **Python/TS/FFI input validation relaxed.**  Dropped the
  "metadata must contain a non-negative integer `version`" guard in
  all three bindings.  `_reserved_` protection and `base[i]` shape
  checks are unchanged.
- **CLI pseudo-key preserved.** `tensogram ls -p version`,
  `tensogram get -p version`, and the default `ls` column continue
  to return `"3"` — they now source from `WIRE_VERSION` rather than
  from any CBOR key.
- **Zarr / xarray attribute renamed.**  The Zarr group attribute
  `_tensogram_version` becomes `_tensogram_wire_version`; the
  xarray dataset attribute `tensogram_version` becomes
  `tensogram_wire_version`.  Both still source from the preamble
  (always `3` in the current library).
- **Golden fixtures regenerated.**  `rust/tensogram/tests/golden/*.tgm`
  were rewritten under the new schema.  `simple_f32.tgm`,
  `multi_object.tgm`, `multi_message.tgm`, and `hash_xxh3.tgm` now
  carry a generic `{"product": "efi", "parameter": "pressure"}`
  vocabulary; `mars_metadata.tgm` keeps its MARS namespace as the
  canonical MARS-regression fixture.
- **Anemoi plugin** stops seeding `{"version": 3}` in its output
  metadata — the key was redundant.
- **NetCDF / GRIB converters** stop emitting a `"version"` top-level
  CBOR key; output bytes shrink by ~7 bytes per message.
- **Documentation and examples** refreshed across `docs/src/**/*.md`,
  `plans/WIRE_FORMAT.md` §6.1, `plans/DESIGN.md`, `plans/ARCHITECTURE.md`,
  and every `examples/{rust,python,cpp,typescript,jupyter}` file.

### Stats

- Rust workspace: 1505 tests passing (824 with `tensogram --features
  remote,async`).  Excluded-crate suites: 36 across `tensogram-grib`
  + `tensogram-netcdf`.
- Python: 530 + 242 + 235 tests across `python/tests/`,
  `python/tensogram-xarray/tests/`, and `python/tensogram-zarr/tests/`
  (45 skipped on CPython 3.14).
- WASM: 172 tests (161 lib + 11 new `layout_tests` suite).
- TypeScript (vitest): 376 tests across 27 files.
- Tensoscope (vitest): 12 tests.

## [0.17.0] - 2026-04-22

### Fixed

- `tensogram-zarr`: `TensogramStore` now falls back to descriptor-level
  keys (`name`, `param`, `shortName`, ...) for variable naming when
  `meta.base[i]` does not supply one, matching the long-standing xarray
  backend behaviour.  This rescues files where application metadata
  was accidentally placed in the descriptor dict (a common Python-API
  mistake, since unknown keys are silently routed to
  `DataObjectDescriptor.params`).  Per-key precedence across sources
  follows the existing name chain; `meta.base[i]` still wins for the
  same key.  ([#67](https://github.com/ecmwf/tensogram/issues/67))

- `tensogram-xarray`: `xr.open_dataset(engine="tensogram")` no longer
  raises `ValueError: conflicting sizes for dimension 'dim_0'` on
  messages that mix objects of different ranks without CF-recognised
  coordinate names.  Generic `dim_N` fallback names that would collide
  across variables of different shapes are renamed to
  `obj_{i}_dim_{axis}` so the Dataset opens cleanly.  Coordinate dim
  names are never auto-renamed; hinted names that claim conflicting
  sizes across objects emit a warning and are renamed to the same
  per-object form as generic fallbacks so the Dataset still opens.
  ([#66](https://github.com/ecmwf/tensogram/issues/66))

- `tensogram-xarray`: `open_datasets()` and `merge_objects=True` now
  honour `_extra_["dim_names"]` (list and size-to-name dict) with the
  same priority chain as single-message `open_dataset`.  Previously
  only the single-message path read the hint.

### Changed

- `tensogram` (Python): `encode()`, `encode_pre_encoded()`,
  `TensogramFile.append()`, `StreamingEncoder.write_object()`, and
  `StreamingEncoder.write_object_pre_encoded()` now emit a
  `UserWarning` when a descriptor dict contains keys that look like
  application metadata (`name`, `param`, `shortName`, `long_name`,
  `description`, `units`, `dim_names`, `mars`, `cf`, `product`,
  `instrument`), pointing callers at `meta["base"][i]` as the
  canonical location.  The keys are still captured into
  `DataObjectDescriptor.params` for wire compatibility, so existing
  files keep round-tripping; the warning is suppressible via standard
  `warnings.filterwarnings`.  One aggregated warning per descriptor
  rather than one per key.
  ([#67](https://github.com/ecmwf/tensogram/issues/67))

### Added

- `tensogram-xarray`: new opt-in reader convention — producers may
  embed `base[i]["dim_names"]` (axis-ordered list) per object.  The
  backend validates the hint (length equal to ndim, all non-empty
  distinct strings) and uses it in the dim resolution chain:
  user kwarg > coord match > per-object hint > `_extra_` hint >
  generic fallback.  Malformed hints are logged at DEBUG and ignored
  so files remain openable.  Inconsistent hints across a
  merge/hypercube group trigger a warning and fall back to lower
  priority sources.  ([#66](https://github.com/ecmwf/tensogram/issues/66))

- `tensogram-anemoi`: the output plugin now writes the per-object
  `dim_names` list on each data field entry (`["values"]` for flat
  fields, `["values", "level"]` for stacked pressure-level fields)
  alongside the existing message-level `_extra_["dim_names"]` hint,
  opting into the new reader convention while staying
  backward-compatible with older readers.

### Wire format v3 — BREAKING

This release is a clean break from v2.  There is no backward
compatibility; v2 messages are rejected at preamble read with a
clear error pointing at re-encoding.  The v3 spec lives at
[`plans/WIRE_FORMAT.md`](plans/WIRE_FORMAT.md).

#### Preamble / postamble

- Preamble `version` bumped `2 → 3`.  `Preamble::read_from`
  requires `version == 3` exactly.  `pub const WIRE_VERSION = 3`
  in `rust/tensogram/src/wire.rs` is the single source of truth.
- Postamble grew `16 → 24 B`.  New layout
  `[first_footer_offset u64][total_length u64][END_MAGIC 8]`.
  The mirrored `total_length` makes the postamble self-locating
  from any byte position inside a message, enabling backward and
  bidirectional scan.
- New preamble flag `HASHES_PRESENT` (bit 7) signals whether the
  inline per-frame hash slots are populated.

#### Frame registry

- Type 4 (obsolete v2 `NTensorFrame`) — removed from the enum.
  Any decoder that reads a type-4 frame hard-fails with a
  reserved-type error.
- Type 9 renamed from `NTensorMaskedFrame` → `NTensorFrame` and
  is the only concrete data-object type in v3.  The body phase is
  explicitly designed to accommodate future non-tensor
  data-object types at fresh unused type numbers without a
  wire-format version bump.

#### Hashing

- Every frame now ends with a common 12-byte tail
  `[hash u64][ENDF 4]`.  Data-object frames have a 20 B footer
  `[cbor_offset u64][hash u64][ENDF 4]`.
- Hash scope is the frame *body* only — `bytes[16 ..
  frame_end − footer_size(type))`.  Neither the header nor any
  byte of the footer (including `cbor_offset`) is covered.
- Hashing is message-wide: `HASHES_PRESENT = 1` populates every
  frame's slot, `= 0` leaves them zero.  No per-frame flag.
- `DataObjectDescriptor.hash: Option<HashDescriptor>` — REMOVED.
  The inline slot is the single integrity source.
- `HashFrame` CBOR schema: `hash_type → algorithm`, `object_count`
  removed (derived from `hashes.len()`).
- `IndexFrame` CBOR schema: `object_count` removed (derived from
  `offsets.len()`).
- New `EncodeOptions.create_header_hashes: bool` (default `true`
  buffered) and `.create_footer_hashes: bool` (default `false`
  buffered; streaming folds both into the footer).
- `validate --checksum` rewired to recompute per-frame body
  hashes and compare to the inline slot — no CBOR parse on the
  fast path.

#### Compression codecs

- `rle` and `roaring` promoted from mask-companion-only to
  first-class `DataObjectDescriptor.compression` values.
  Bitmask-only: attempting either codec on a non-`Bitmask` dtype
  is an encode-time `EncodingError`.  `decompress_range` returns
  `CompressionError::RangeNotSupported`.

#### Scan

- `scan` / `scan_file` default to bidirectional (meet-in-the-middle)
  walking.  New `ScanOptions { bidirectional, max_message_size }`
  lets callers opt out.  Falls back to pure forward scan when the
  backward walker hits a `total_length = 0` (streaming from a
  non-seekable sink) or any I/O / postamble anomaly.

### Added

- **NaN / Inf bitmask companion frame** (wire type 9
  `NTensorMaskedFrame`) — encoders that pass the new `allow_nan` /
  `allow_inf` opt-in now substitute non-finite values with `0.0` on
  the wire and record their positions in up to three compressed
  bitmasks stored alongside the payload.  Decoders restore canonical
  NaN / ±Inf at the marked positions.  See
  `plans/WIRE_FORMAT.md` §6.5 and
  `docs/src/guide/nan-inf-handling.md` for the design and usage.
- Per-kind mask compression methods:
  `nan_mask_method` / `pos_inf_mask_method` / `neg_inf_mask_method`,
  accepting `"none"` | `"rle"` | `"roaring"` (default) | `"blosc2"` |
  `"zstd"` | `"lz4"`.  Small-mask auto-fallback to `"none"` below
  `small_mask_threshold_bytes` (default 128).
- Decode-side `restore_non_finite` flag (default `true`) and the
  advanced `decode_with_masks` API (Rust + Python) that return raw
  decompressed bitmasks alongside the substituted payload.
- `tensogram validate --full` is now mask-aware: NaN / ±Inf at a
  masked position is expected; at any other position it still
  fails as `NanDetected` / `InfDetected`.
- CLI: `--allow-nan` / `--allow-inf` global flags with matching
  `TENSOGRAM_ALLOW_NAN` / `TENSOGRAM_ALLOW_INF` env vars.
  Per-kind `--nan-mask-method` / `--pos-inf-mask-method` /
  `--neg-inf-mask-method` and `--small-mask-threshold` flags with
  matching `TENSOGRAM_*` env vars.

### Changed — BREAKING (default-behaviour flip, 0.17 pre-release)

- **Non-finite float values now error by default at encode.** Pre-0.17
  the library passed NaN / ±Inf bit patterns through
  `encoding="none"` pipelines verbatim; opt-in `reject_nan` /
  `reject_inf` flags upgraded rejection to pipeline-independent.
  0.17+ inverts that policy: rejection is the default across every
  encode path (`tensogram::encode`, `file.append`, streaming,
  converters, CLI), and the former opt-in flags are **removed**.
  Callers who intentionally shipped NaN-bearing data through
  `encoding="none"` must now either pre-process the data or opt in
  to the new `allow_nan` / `allow_inf` bitmask companion.
- **Frame-type rename**: pre-0.17 `DataObject` = wire type 4
  becomes `NTensorFrame`.  New encoders emit `NTensorMaskedFrame`
  (wire type 9) for every data object; decoders accept both.  See
  `docs/src/format/wire-format.md` for the registry.
- **`DataObjectDescriptor` gains an optional `masks` field** of
  type `Option<MasksMetadata>`.  Absent by default, serialises as
  no CBOR key — byte-compatible with pre-0.17 descriptors when no
  masks are present.
- **Removed API surface** (hard break — no deprecation shim):
  - Rust: `EncodeOptions::reject_nan`, `EncodeOptions::reject_inf`,
    and the `tensogram::strict_finite` module; non-finite handling
    now uses the default-reject policy plus the
    `tensogram::substitute_and_mask` / `allow_nan` / `allow_inf`
    opt-in path where callers want mask-based preservation.
  - Python: `reject_nan` / `reject_inf` kwargs on `tensogram.encode`,
    `TensogramFile.append`, `StreamingEncoder`, `convert_grib`,
    `convert_grib_buffer`, `convert_netcdf`.
  - TypeScript: `EncodeOptions.rejectNan`, `EncodeOptions.rejectInf`,
    `StreamingEncoderOptions.rejectNan`, `StreamingEncoderOptions.rejectInf`.
  - C FFI: removed `bool reject_nan, bool reject_inf` parameters from
    `tgm_encode`, `tgm_file_append`, `tgm_streaming_encoder_create`.
    Headers regenerated; pre-0.17 C callers get compile errors.
  - C++: removed `reject_nan` / `reject_inf` fields from
    `encode_options`.
  - CLI: removed `--reject-nan` / `--reject-inf` global flags and
    `TENSOGRAM_REJECT_NAN` / `TENSOGRAM_REJECT_INF` env vars (clap
    reports "unknown argument" if passed).
  - Docs: replaced `docs/src/guide/strict-finite.md` with
    `docs/src/guide/nan-inf-handling.md`, which covers both the
    default-reject policy and the `allow_nan` / `allow_inf` bitmask
    opt-in.  Cross-references from `edge-cases.md`, `error-handling.md`,
    `python-api.md`, `simple-packing.md`, `convert-netcdf.md`
    updated.
- **`encode_pre_encoded` no longer runs the finite check.** The
  pre-encoded API was previously a source of conflicting semantics
  (the flags errored if set on this path); it now unconditionally
  treats bytes as opaque.

### Fixed

- **`simple_packing::encode` safety net handles `i32::MIN` correctly.**
  Before: `params.binary_scale_factor.abs()` panics on debug builds
  (integer-overflow) and silently returns `i32::MIN` (still negative)
  on release, so the comparison `> 256` falsely accepts the
  worst-case value, and the subsequent encode silently corrupts.
  After: uses `saturating_abs()` which clamps `i32::MIN` to
  `i32::MAX`, correctly rejecting it via
  `PackingError::InvalidParams`.  Regression test added.

### Added

- **Strict-finite kwargs on Python converter functions.**
  `tensogram.convert_grib`, `tensogram.convert_grib_buffer`, and
  `tensogram.convert_netcdf` now accept `reject_nan: bool = False`
  and `reject_inf: bool = False` kwargs that forward to the shared
  `EncodeOptions`.  CLI parity (`tensogram convert-grib --reject-nan`)
  already existed; this closes the corresponding Python gap.

### Changed — BREAKING (converter behaviour)

- **`tensogram convert-grib` / `convert-netcdf` now hard-fail when
  `--encoding simple_packing` is requested on data containing NaN or
  Inf.** The previous behaviour (stderr warning + silent downgrade to
  `encoding="none"`) hid real data-quality problems; the new behaviour
  fails cleanly with an error that names the offending variable and
  the sample index. Recovery options: (a) pick a non-simple_packing
  encoding up front, (b) pre-process NaN/Inf out of the data, or (c)
  use `tensogram.encode(..., reject_nan=True)` which surfaces the same
  failure at encode time.  The non-f64-payload branch (structural
  mismatch, not a data-quality problem) keeps its stderr-warning +
  fallback behaviour unchanged.

### Added — simple_packing standalone-API safety net

- **`simple_packing::encode_with_threads` now validates its
  `SimplePackingParams` input** with an always-on safety net.  Catches
  hand-crafted or mutated params that would otherwise produce
  silently-wrong output:
  - Non-finite `reference_value` (NaN / ±Inf) → new
    `PackingError::InvalidParams { field: "reference_value", .. }`.
  - `|binary_scale_factor| > 256` → new
    `PackingError::InvalidParams { field: "binary_scale_factor", .. }`.
    The threshold catches the `i32::MAX` fingerprint that results from
    feeding Inf through `compute_params`.  Real-world weather data
    (`|bsf| ≤ 60`) comfortably fits.  Exposed as the public constant
    `MAX_REASONABLE_BINARY_SCALE = 256`.
  - `bits_per_value = 0` remains valid (legitimate constant-field
    encoding); `> 64` continues to be caught by the pre-existing
    `BitsPerValueTooLarge`.
  The validation fires on every encode path — direct Rust calls via
  `simple_packing::encode()`, via the high-level `tensogram::encode()`,
  via PyO3, WASM, C FFI, and C++ wrapper.  Cross-language parity tests
  pin the behaviour.

### Removed — BREAKING

- **`tensogram-core` redirect crate removed.** The
  `rust/tensogram-core-redirect/` directory has been deleted and is no
  longer part of the workspace; no new versions of `tensogram-core` will
  be published to crates.io. The redirect was introduced in 0.15 (when
  the crate was renamed `tensogram-core` → `tensogram`) as a thin
  `pub use tensogram::*;` shim to give downstream users a grace period
  to migrate; three minor versions later we are retiring it.
  Users still depending on `tensogram-core` should switch to
  `cargo add tensogram` directly. The previously published
  `tensogram-core@0.14.0` through `tensogram-core@0.16.1` remain
  available on crates.io as (now-frozen) re-exports.

### Stats

- Rust workspace: 1487 tests passing (1593 with `remote` + `async`
  feature coverage).  Excluded-crate suites run separately: 387 tests
  across `tensogram-grib`, `tensogram-netcdf`, and `tensogram-cli` built
  with `grib` + `netcdf`.
- Python: 526 + 242 + 235 tests across `python/tests/`,
  `python/tensogram-xarray/tests/`, and `python/tensogram-zarr/tests/`
  (29 skipped on CPython 3.13 and 3.13t free-threaded).  Jupyter
  notebooks: 32 structural-guard + 46 nbval-lax tests.
- C++: 143 tests (Linux + macOS).  WASM: 161 tests.
  TypeScript (vitest): 319 tests.

## [0.16.1] - 2026-04-19

### Fixed
- **Jupyter notebook CI failures** — `examples/jupyter/pyproject.toml` pinned
  `tensogram>=0.15.0,<0.16` but the repo had moved to 0.16.0, causing
  `uv pip install` to silently replace the locally-built bindings with an
  older PyPI wheel lacking grib/netcdf features. Bumped all three version
  references (package version, `tensogram` dep, `tensogram[xarray]` dep) to
  `>=0.16.0,<0.17`.
- **`make-release` command** — added `examples/jupyter/pyproject.toml` to the
  version bump checklist, dependency pins list, and stale-version grep check
  so future releases won't miss it.

### Stats
- Rust tests: 513 passed
- Python tests: 513 passed, 40 skipped
- xarray tests: 201 passed
- zarr tests: 224 passed

## [0.16.0] - 2026-04-18

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
  accepted `Inf` input and produced numerically-useless parameters
  that decoded to NaN everywhere; also now independently guarded at
  the simple_packing layer. Exposed across every language surface
  (Rust, Python, TS, C FFI, C++) with cross-language parity tests.
  CLI global flags `--reject-nan` / `--reject-inf` plus
  `TENSOGRAM_REJECT_NAN` / `TENSOGRAM_REJECT_INF` env vars for ops
  rollouts. Env-var values are bool-ish: `1`/`true`/`yes`/`on`
  enable, `0`/`false`/`no`/`off` disable. New
  `docs/src/guide/strict-finite.md` covered the full semantics
  (superseded in 0.17 by `docs/src/guide/nan-inf-handling.md`).
- `docs/src/guide/vocabularies.md` — a developer-guide page listing example
  application vocabularies (MARS, CF, BIDS, DICOM, custom) and conventions
  for wiring them into Tensogram metadata.
- `examples/{rust,python,cpp,typescript}/02b_generic_metadata.*` — sibling
  examples to the MARS-based `02_mars_metadata.*` set, showing that the
  per-object metadata mechanism works with any application namespace.

### Added — TypeScript wrapper: streaming `StreamingEncoder`

- **`StreamingEncoderOptions.onBytes`** — an optional synchronous
  `(chunk: Uint8Array) => void` callback.  When supplied, every chunk
  of wire-format bytes the encoder produces is forwarded to the
  callback as it is produced; no internal buffering is performed and
  `finish()` returns an empty `Uint8Array`.  Closes the "true no-
  buffering stream" gap flagged in the Pass-4 focus list — useful for
  browser uploads, WebSocket pushes, or any sink that needs bytes
  incrementally.
- **`StreamingEncoder#streaming` getter** reports whether an `onBytes`
  callback was supplied (so call sites that need to branch on mode
  don't have to remember their own options).
- **WASM**: new `JsCallbackWriter` (`std::io::Write` into a
  `js_sys::Function`) plus an `Inner::Buffered` / `Inner::Streaming`
  enum dispatcher inside the exported `StreamingEncoder` class.  The
  constructor gained a third optional argument `on_bytes:
  Option<js_sys::Function>`; buffered-mode callers pass `None` /
  omit it and see no behavioural change.
- 9 new TS tests and 5 new wasm-bindgen tests covering construction-
  time delivery, multi-object chunking, bytes-written parity,
  callback-throw propagation, non-function rejection, buffered-vs-
  streaming mode detection, and `hash: false` compatibility.
- New example `examples/typescript/14_streaming_callback.ts`.

### Added — Python bindings

- `tensogram.compute_hash(data, algo="xxh3")` — hex digest over
  arbitrary bytes.  Closes the cross-language parity gap vs Rust,
  WASM, C FFI, and C++ (all of which already exposed an equivalent).
  Accepts `bytes` and `bytearray` zero-copy via `PyBackedBytes`;
  other buffer-protocol objects (`memoryview`, `numpy.ndarray`, …)
  must be converted via `bytes(obj)` / `obj.tobytes()` first.
  Default algorithm `"xxh3"`; unknown names raise `ValueError`.

### Added — TypeScript wrapper Scope C.1 (API-surface parity)

- `decodeRange(buf, objIndex, ranges, opts?)` — partial sub-tensor
  decode mirroring Rust `decode_range`, Python `file_decode_range`,
  and `tgm_decode_range`.  `ranges` is an array of `[offset, count]`
  pairs (numbers or bigints); the result carries one dtype-typed view
  per range, or a single concatenated view when `join: true`.
- `computeHash(bytes, algo?)` — standalone hex-digest computation,
  matching the digest stamped by `encode()` on the same bytes.
- `simplePackingComputeParams(values, bits, decScale?)` — GRIB-style
  simple-packing parameter computation.  Returns snake-case keys so
  the result spreads directly into a descriptor.
- `encodePreEncoded(meta, objects, opts?)` — wrap already-encoded
  payloads into a wire-format message without re-running the
  encoding pipeline.  The library recomputes the payload hash.
- `validate(buf, opts?)` / `validateBuffer(buf, opts?)` / `validateFile(path, opts?)` —
  structural / metadata / integrity / fidelity validation with modes
  `quick`, `default`, `checksum`, `full`.  Returns a typed
  `ValidationReport` / `FileValidationReport`; never throws on bad
  input.
- `StreamingEncoder` class — frame-at-a-time message construction
  (`writeObject`, `writePreceder`, `writeObjectPreEncoded`, `finish`).
  Backed by an in-memory `Vec<u8>` sink on the WASM side, matching
  Python's `StreamingEncoder`.
- `TensogramFile#append(meta, objects, opts?)` — Node-only local-
  file-path append.  Rejects `fromBytes` / `fromUrl`-backed files
  with `InvalidArgumentError` to match the Rust / Python / FFI / C++
  contract.
- **Lazy `TensogramFile.fromUrl`** — a `HEAD` probe detects
  `Accept-Ranges: bytes` + `Content-Length`; when present, the file
  lazily scans preambles and fetches message payloads on demand via
  HTTP `Range` requests (small LRU cache).  Falls back transparently
  to a single eager GET when Range isn't supported.

### Added — TypeScript wrapper Scope C.2 (first-class dtypes)

- `Float16Polyfill` — TC39-Stage-3-accurate `Float16Array`-shaped
  polyfill.  Round-ties-to-even narrow, NaN / ±Inf / ±0 / subnormal
  preservation.  Used when the host runtime lacks
  `globalThis.Float16Array`.
- `Bfloat16Array` — view class for the 1-8-7 brain-float layout.
- `ComplexArray` — view class over interleaved Float32 / Float64
  storage with `.real(i)`, `.imag(i)`, `.get(i) → {re, im}`, and
  iteration.
- `hasNativeFloat16Array()`, `getFloat16ArrayCtor()`,
  `float16FromBytes()`, `bfloat16FromBytes()`, `complexFromBytes()`
  factories for zero-copy construction.
- `typedArrayFor('float16')` now returns a native `Float16Array` or
  the polyfill.  `typedArrayFor('bfloat16')` returns
  `Bfloat16Array`.  `typedArrayFor('complex64' | 'complex128')`
  returns `ComplexArray`.

### Changed

- **Documentation reframed for the broader scientific-computing community.**
  The motivation, introduction, README, and concept pages now present
  Tensogram as a general-purpose N-tensor message format for scientific data
  at scale, with ECMWF weather-forecasting workloads as one well-validated
  use case rather than the founding motivation. GRIB and NetCDF remain
  first-class importers; MARS vocabulary remains a fully supported example.
  No API, wire-format, or behavioural changes.
- `tensogram-grib` and `tensogram-netcdf` are now framed as *importers* (one-way
  GRIB → Tensogram / NetCDF → Tensogram), not "converters". CLI subcommands
  `convert-grib` and `convert-netcdf` keep the same names and flags.
- `tensogram-benchmarks`: `datagen::generate_weather_field` renamed to
  `generate_smooth_field` to reflect that the generator is domain-neutral
  (its output matches the statistical profile of any smooth bounded-range
  scientific field — temperature, pressure, intensity, density). Callers
  inside the benchmarks crate updated accordingly; benchmark outputs and
  numerical results are unchanged.
- **FFI signature extension** — `tgm_encode`, `tgm_file_append`, and
  `tgm_streaming_encoder_create` now take two additional
  `reject_nan` / `reject_inf` bool parameters. `tgm_encode_pre_encoded`
  intentionally does not — pre-encoded bytes are opaque to the library.
  Pre-0.15 C callers will see a compile error from the regenerated
  header; pass `false, false` to preserve old behaviour.
- **Pre-encoded APIs reject the strict-finite flags loudly.**
  Setting `reject_nan` or `reject_inf` on `encode_pre_encoded` or
  `StreamingEncoder::write_object_pre_encoded` now returns an
  `EncodingError` rather than silently discarding the flag. Pre-encoded
  bytes are opaque to the library and cannot be meaningfully scanned;
  failing loudly brings Rust and C++ behaviour in line with Python
  (which raised `TypeError` from day one). Cross-language uniformity
  achieved.

### Changed — Rust core (affects all language bindings)

- **`tensogram_encodings::simple_packing` now rejects ±Infinity values
  alongside NaN.**  Simple-packing's `binary_scale_factor` is computed
  from `(max − min) / max_packed` — an infinite range produced a
  nonsensical `i32::MAX`-saturated scale factor and garbage output.
  The guard was previously only in the TS wrapper (Pass 3); it is now
  in the Rust core so Python, C FFI, C++, and WASM all benefit
  simultaneously.  New `PackingError::InfiniteValue(usize)` variant
  reports the index of the first offending sample (first-offender
  guarantee in sequential mode; non-deterministic choice in parallel
  mode, matching the existing NaN contract).  The previous TS-side
  `Number.isFinite` check in `simplePackingComputeParams` is kept as
  defence-in-depth so callers still see a clean
  `InvalidArgumentError` without a WASM round-trip.  This is now
  defence-in-depth alongside the new pipeline-independent
  `reject_inf` `EncodeOptions` flag — both guards fire for the same
  input but through different code paths.

### Changed — TypeScript wrapper

- **BREAKING: `TensogramFile#rawMessage(index)`** is now `async`
  (returns `Promise<Uint8Array>`).  Needed so the lazy HTTP backend
  can issue a `Range` GET on first access.  Existing call sites add
  `await`.
- **BREAKING: `typedArrayFor` for half-precision and complex dtypes**
  returns view classes, not raw `Uint16Array` / interleaved
  `Float32Array`.  Consumers that need the raw bits reach them via
  `.bits` (half-precision) or `.data` (complex).

### Added — WASM

- `tensogram-wasm` exports `decode_range`, `encode_pre_encoded`,
  `compute_hash`, `simple_packing_compute_params`, `validate_buffer`,
  and the `StreamingEncoder` class.

### Stats

- Rust workspace: 1324 tests passing. `tensogram-grib` 17 tests,
  `tensogram-netcdf` 44 tests (feature-gated suites run separately).
- Python: 538 + 201 + 224 tests across `python/tests/`,
  `python/tensogram-xarray/tests/`, and `python/tensogram-zarr/tests/`.
- C++: 154 tests. WASM: 161 tests. TypeScript (vitest): 326 tests.
- All examples build cleanly (`cargo build --workspace`); mdbook docs
  build cleanly.
- `cargo fmt --check`, `cargo clippy --workspace --all-targets
  --all-features -- -D warnings`, and `ruff format --check` all green.

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
- **TypeScript wrapper** (`@ecmwf.int/tensogram` under `typescript/`)
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

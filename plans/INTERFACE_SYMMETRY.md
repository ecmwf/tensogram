# Cross-Language Interface Symmetry — Gap Analysis & Action Plan

> **Status: ACTIVE.** Authoritative record of the 0.24.0 cross-language symmetry
> audit and the roadmap to close the gaps. The summary matrix + accepted
> exceptions live in `plans/DESIGN.md` § *Cross-Language Interface Symmetry*;
> the rule for contributors is in `AGENTS.md` § *Cross-language interface
> symmetry*. This doc holds the **detailed per-binding inventory** and the
> **dependency-ordered action plan**.

## 1. Goal

Every binding (Rust, C, C++, Python, TypeScript, Fortran) exposes the **same
user-facing capabilities**, differing only in idiom. **Rust core
(`rust/tensogram/src/lib.rs`) is the single reference.** Any genuine gap must be
explicit and classified — never accidental.

## 2. Method

Six concurrent code-analysis passes, one per surface, each reading the actual
interface code, the backend it binds (Rust core for Py/TS; the C ABI
`rust/tensogram-ffi/tensogram.h` for C++/Fortran), and the per-language
`examples/`. Findings were spot-verified against source (not docs). Totals:
the C ABI declares **135** `tgm_*` functions; C++ wraps ~124; Fortran binds
**49** (~36%).

## 3. Why the surface diverged

1. **Bindings grew around their consumers** with no enforced symmetry contract:
   Python (data science) and TypeScript (web/remote) got the deepest
   investment; C/C++ a broad sync+async core; Fortran a minimal sync core.
2. **The C ABI is the bottleneck for the C-family.** Python/TS bind the Rust
   **core** directly (PyO3 / wasm-bindgen) and raced ahead; C, C++, and Fortran
   can only be as complete as `tensogram-ffi`, and many capabilities were never
   lowered into it.

Gap taxonomy (used throughout): **[O]** omission (backend has it, binding
doesn't — fix in binding), **[B]** backend gap (the C ABI/core lacks it — widen
upstream first), **[L]** language limit (accepted; listed in DESIGN.md).

## 4. Feature × language matrix

See `plans/DESIGN.md` § *Cross-Language Interface Symmetry* for the 24×6 status
matrix. Headline: **Python & TS most complete; C/C++ broad but stringly-typed
and C-ABI-bound; Fortran smallest (mostly [O] omissions).**

## 5. Detailed per-binding gap inventory

### 5.1 C FFI (`rust/tensogram-ffi`) — 135 functions, stringly-typed

- **[B] Not lowered into the C ABI at all:** GRIB/NetCDF convert; mask
  introspection (`decode_with_masks`, `DecodedMaskSet`, `MaskDescriptor`,
  `MasksMetadata`); `decode_descriptors`, `decode_range_from_payload`;
  `scan_with_options`, `scan_file`, `scan_file_with_options`, `ScanOptions`,
  `data_object_inline_hashes`; `validate_buffer` + typed validation model
  (report/issue/severity/level are JSON-only); `compute_common`,
  `verify_canonical_cbor`; `objects_metadata` iterator; sync remote +
  `is_remote_url` (only async remote *open* exists); typed wire introspection
  (`FrameType`, `MessageFlags`, `MessageLayout`); typed `Dtype`/`ByteOrder`/
  `AggregateHashPolicy`/`CompressionBackend`; `EncodeOptions.compression_backend`
  / `parallel_threshold_bytes` / `aggregate_hash` setters.
- **[BUG] Documented-but-missing** `tgm_last_error_object_index()` (referenced in
  the `TGM_ERROR_MISSING_HASH` note, `tensogram.h:61` / `lib.rs:110`) — the
  offending object index is unreachable from C.
- **Awkward/incomplete:** sync file API lacks file-level `decode_metadata`/
  `decode_object`/`decode_range` (async has them); compression/filter/encoding/
  dtype/byte-order are JSON strings; `tgm_async_streaming_encoder_object_count`
  overloads `usize::MAX` for 3 failure modes; `tgm_message_num_decoded`
  duplicates `num_objects`.
- **Dead/untested surface (10 C fns unreferenced by any C++ wrapper/test/example):**
  `tgm_doctor_to_json`, `tgm_message_num_decoded`, `tgm_async_file_decode_object`,
  `tgm_async_file_decode_range`, `tgm_async_task_join_multi_bytes`,
  `tgm_multi_bytes_free`, `tgm_async_file_path`,
  `tgm_async_streaming_encoder_path`, `tgm_async_task_cancel`,
  `tgm_async_task_is_ready`.

### 5.2 C++ (`cpp/include/tensogram.hpp`)

- **[O] C ABI provides it, C++ does not wrap it (11 fns — highest-value, cheap):**
  `tgm_doctor_to_json` → whole `doctor()` category missing; **
  `tgm_simple_packing_compute_params`**; async `tgm_async_file_decode_object`,
  `tgm_async_file_decode_range` (+ `tgm_async_task_join_multi_bytes`,
  `tgm_multi_bytes_free`); `tgm_async_file_path`,
  `tgm_async_streaming_encoder_path`; `tgm_async_task_cancel`,
  `tgm_async_task_is_ready`; `tgm_message_num_decoded`.
- **[B] Shared with the C ABI (fix upstream):** `decode_descriptors`,
  `decode_with_masks`; scan options/file; typed validation; `compute_common`,
  `verify_canonical_cbor`; typed `Dtype`/`ByteOrder`/`AggregateHashPolicy`/
  `CompressionBackend`; `FrameType`/`MessageFlags`/`MessageLayout`; sync remote;
  `parse_hash_name`/`HASH_ALGORITHM_NAME`; package version.
- **Example holes:** the **entire precise metadata cursor** (`get`/`get_at`/
  `try_get_*`/`meta_value`/`object`/`extra`/`reserved`) and masks are exercised
  by **no** `examples/cpp/*`; `strides`/`object_type`/`byte_order`/`filter`/
  `compression`/`hash_*`, `compute_hash()`, `streaming_encoder::write_object`/
  `write_preceder` also uncovered.

### 5.3 Python (`python/bindings`) — most complete

- **[O] Rust core has it, Python doesn't:** **`AsyncStreamingEncoder`** (async
  streaming *encoder*) entirely absent; **`compute_common`** (jup01 reimplements
  it in pure Python); lazy `objects`/`objects_metadata` iterators; scan family
  (`scan_file`/`scan_with_options`/`ScanOptions`); `validate_buffer`; streaming
  introspection (`write_preceder`/`object_count`/`bytes_written`); file
  `message_layouts()`/`open_mmap`; `verify_canonical_cbor`,
  `HASH_ALGORITHM_NAME`, `parse_hash_name`, `RESERVED_KEY`; `EncodeOptions.
  compression_backend`/`parallel_threshold` kwargs.
- **[B]:** `decode_range_from_payload`; typed wire introspection.
- **[BUG] Inline-hash read dead-end:** `DataObjectDescriptor.hash` is a permanent
  `None` stub whose docstring points to `Message.object_inline_hashes()` /
  `Message.object_hash(i)`, but `Message` is a bare 2-field `namedtuple`
  (`__init__.py:30`) with no such methods → reading v3 inline hashes is
  impossible from Python.
- **Uniquely present:** GRIB/NetCDF convert, `decode_with_masks`, full remote +
  batch + prefetch, async decode, feature probes, exotic-dtype numpy bridging.
- **Example holes:** `doctor`, `decode_with_masks`, `to_grib`/`to_netcdf`, the
  three integrity exception classes, most `Metadata` methods, exotic dtypes.

### 5.4 TypeScript / WASM (`typescript/src`, `rust/tensogram-wasm`)

- **[O]/[B] Rust core has it, TS doesn't:** `decode_with_masks` + mask sets
  (needs wasm widen — binds core); lazy `objects()`/`objects_metadata()`
  iterators; `TensogramFile.create` (empty-file factory); `decode_range_from_
  payload`; scan options/file/inline-hashes; decode-option surface
  (`native_byte_order`); async streaming *encoder* (async is native on the read
  side); public wire introspection is `@internal` only.
- **[BUG] `TensogramFile.append`** declares `allowNan`/`allowInf`/`*MaskMethod`/
  `smallMaskThresholdBytes` in `AppendOptions` but `file.ts:1110` forwards
  **only `hash`** to `encode()` — mask options silently dropped.
- **Minor:** `DecodeStreamOptions` type not re-exported from `index.ts`.
- **Uniquely present / exceeds:** richest remote (lazy HTTP Range, LRU,
  bidirectional scan, AWS SigV4 signer), first-class exotic-dtype view classes
  (`Float16`/`Bfloat16`/`ComplexArray`), typed validation model, extra error
  classes.
- **Example holes:** all dtype utilities + exotic views, `doctor`,
  `validateBuffer`, `simplePackingComputeParams`, most metadata helpers, masks.

### 5.5 Fortran (`fortran/src/tensogram.F90`) — smallest, mostly [O]

- **[O] plain omissions (C ABI has it, trivially bindable):** `encode_pre_encoded`;
  decode variants (`tgm_decode_object`/`_range`/`_metadata`/`_with_options`);
  masks (`*_with_options` + `TgmEncode/DecodeMaskOptions` PODs); `scan`
  (`tgm_scan*`, `tgm_scan_entry_t` is a POD like the already-bound `tgm_bytes_t`);
  `tgm_compute_hash` + inline-hash accessors (`tgm_object_hash_type`/`_value`/
  `tgm_payload_has_hash`/`_encoding`); iterators (10 `tgm_*_iter_*`); `validate`
  + `validate_file` (return JSON bytes); `doctor` (`tgm_doctor_to_json`); object
  accessors (`tgm_object_byte_order`/`_type`/`_filter`/`_compression`/`_strides`);
  version (`TGM_WIRE_VERSION` param + `tgm_message_version`/`_metadata_version`);
  the `threads` argument (hard-coded `0` everywhere); `tgm_file_path`/
  `_append_raw`; streaming `write_preceder`/`write_pre_encoded`/`create_with_
  options`; `tgm_metadata_to_json`/`_object_to_json` + `_get_*_at`.
- **[L] genuine language limits (accepted):** async surface (no runtime/futures/
  closures); `float16`/`bfloat16` (no native half); unsigned dtypes (no unsigned
  kind); metadata builder beyond `base` (no stdlib JSON).
- **[B] backend gaps (cannot fix in Fortran):** convert; `decode_descriptors`/
  `decode_range_from_payload`; `scan_file`/`scan_with_options`; sync remote/
  `is_remote_url`; frame/flags/layout introspection.

## 6. Confirmed interface defects (bugs)

| ID | Binding | Defect |
|----|---------|--------|
| BUG-FFI | C ABI | `tgm_last_error_object_index()` documented but undefined — object index unreachable from C |
| BUG-PY | Python | `DataObjectDescriptor.hash` `None` stub → docstring points to non-existent `Message.object_inline_hashes()`/`object_hash(i)` (Message is a 2-field namedtuple) |
| BUG-TS | TypeScript | `TensogramFile.append` drops all `AppendOptions` except `hash` |
| BUG-CPP-DOCTOR | C++ | `tgm_doctor_to_json` exists but no C++ wrapper (whole doctor category absent) |

## 7. Accepted exceptions

The **[L]** language limits enumerated in `plans/DESIGN.md` § *Documented
exceptions* are the only sanctioned asymmetries. Adding a new one requires
documenting it there with a concrete reason.

## 8. Action plan

### 8.1 Dependency model

Two structural constraints drive scheduling:

- **Disjoint by binding.** Each binding lives in its own tree (`cpp/`,
  `python/`, `typescript/` + `rust/tensogram-wasm/`, `fortran/`), so binding
  tasks for *different* languages never touch the same files and can run fully
  in parallel.
- **The C ABI is a shared, single-file chokepoint.** All C-ABI work lands in
  `rust/tensogram-ffi/src/lib.rs` (+ regenerated `tensogram.h`), so C-ABI tasks
  are **FFI-serial** (one owner at a time). C++ and Fortran *build* the C ABI, so
  their agents must not run while the FFI source is mid-change, and the two of
  them can contend on the cargo-c install prefix — **run C++ and Fortran build
  phases sequentially, not concurrently.** Python (maturin → `python/bindings/
  target`) and TypeScript (wasm-pack → wasm crate target) build in isolated
  target dirs and are safe to parallelize with everything.

Rule of thumb: **Python & TypeScript bind the Rust core directly**, so their
gaps are almost always **[O]** — fixable immediately, in parallel. **C++ &
Fortran are gated on the C ABI** — a **[B]** capability must be lowered into the
FFI first (Wave B) before they can wrap it (Wave C).

### 8.2 Task list

Group = concurrency lane (tasks in different lanes run in parallel; tasks in the
same lane serialize). Lanes: **FFI** (serial, chokepoint), **CPP**, **PY**,
**TS**, **FTN**.

| ID | Scope | Files | Backend change | Depends on | Lane |
|----|-------|-------|----------------|------------|------|
| **Wave A — bugs + cheap [O] (parallel across bindings)** |
| BUG-FFI | Implement `tgm_last_error_object_index()` (thread-local object index for MissingHash/HashMismatch) + regen header | `tensogram-ffi` | +1 fn | — | FFI |
| BUG-PY | Fix inline-hash dead-end: real per-object inline-hash accessor + fix `descriptor.hash` docstring | `python/` | no | — | PY |
| BUG-TS | `TensogramFile.append` forwards all `AppendOptions` (masks) to `encode` | `typescript/` | no | — | TS |
| O-CPP-1 | `doctor()` + wrap unwrapped-existing C fns (`simple_packing_compute_params`, async `decode_object`/`decode_range` + `join_multi_bytes`/`multi_bytes_free`, async `path()`, `task_cancel`/`is_ready`) + examples + tests | `cpp/` | no | BUG-FFI (ordering only) | CPP |
| O-PY-1 | `AsyncStreamingEncoder`; streaming introspection (`object_count`/`bytes_written`/`write_preceder`); lazy `objects`/`objects_metadata`; `compute_common`; missing option kwargs | `python/` | no | — | PY |
| O-TS-1 | Lazy `objects()`/`objects_metadata()`; `TensogramFile.create`; export `DecodeStreamOptions`; `native_byte_order` decode option | `typescript/` (+wasm) | maybe | — | TS |
| O-FTN-1 | Batch: version consts+getters, `encode_pre_encoded`, decode variants, object accessors, scan, `compute_hash`+inline, iterators, validate, doctor, masks `*_with_options`, `threads` arg, file `path`/`append_raw`, streaming `write_preceder`/`write_pre_encoded` + examples + tests | `fortran/` | no | — | FTN |
| **Wave B — C ABI widening [B] (FFI-serial; unblocks C++/Fortran)** |
| W-DECODE | `decode_descriptors`, `decode_with_masks` (+ mask-set out-structs), `decode_range_from_payload` | `tensogram-ffi` | yes | — | FFI |
| W-SCAN | `scan_file`, `scan_with_options`, `ScanOptions`, `data_object_inline_hashes` | `tensogram-ffi` | yes | — | FFI |
| W-VALIDATE | `validate_buffer` | `tensogram-ffi` | yes | — | FFI |
| W-META | `compute_common`, `verify_canonical_cbor` | `tensogram-ffi` | yes | — | FFI |
| W-ENUMS | typed `Dtype`/`ByteOrder`/`AggregateHashPolicy`/`CompressionBackend` + `EncodeOptions` knobs (`aggregate_hash`, `compression_backend`, `parallel_threshold`) | `tensogram-ffi` | yes | — | FFI |
| W-WIRE | typed `FrameType`/`MessageFlags`/`MessageLayout` | `tensogram-ffi` | yes | — | FFI |
| W-REMOTE | sync remote + `is_remote_url` + `RemoteScanOptions` (larger) | `tensogram-ffi` | yes | — | FFI |
| **Wave C — wrap the widened ABI (parallel by binding, after its W-*)** |
| C-CPP-{DECODE,SCAN,VALIDATE,META,ENUMS,WIRE,REMOTE} | C++ wrappers + tests + examples | `cpp/` | no | matching W-* | CPP |
| C-FTN-{DECODE,SCAN,VALIDATE,META,WIRE,REMOTE} | Fortran wrappers (skip [L]) + tests + examples | `fortran/` | no | matching W-* | FTN |
| **Wave D — reach the core directly (parallel, isolated)** |
| D-TS-MASKS | TS `decode_with_masks` via a wasm-crate widen (binds core) | `typescript/`+wasm | wasm | — | TS |
| D-PY-2 | Python `scan_file`/`scan_with_options`, `validate_buffer`, `message_layouts`/`open_mmap` | `python/` | no | — | PY |
| **Wave E — convert: DECIDED not to lower into the core C ABI (accepted `[A]` exception)** |
| E-DECISION | Convert (GRIB/NetCDF) stays **CLI + Python**. Lowering `libeccodes`/`libnetcdf`/`libhdf5` into the lean C ABI is deliberately avoided — it would break the self-contained C-API tarball and force native-lib linkage on every C-family build. See `DESIGN.md` § *Documented exceptions* `[A]`. Escalation path (a separate opt-in `tensogram-convert-ffi`) is tracked in `plans/TODO.md`, not scheduled. | docs | — | — | — |
| **Wave F — example/test coverage (per binding, ongoing)** |
| F-{CPP,PY,TS,RS}-COVER | Add examples exercising the precise metadata cursor, exotic dtypes, masks, doctor where missing | per lang | no | features exist | each |

### 8.3 Dispatch order (concurrency)

1. **Now — Wave A:** land **BUG-FFI** first (settles the C ABI), then dispatch
   **{O-CPP-1, O-PY-1+BUG-PY, O-TS-1+BUG-TS, O-FTN-1}** in parallel. Constraint:
   O-CPP-1 and O-FTN-1 both build the C API — run their *build/test* phases
   sequentially (PY/TS are isolated and fully concurrent). D-PY-2 / D-TS-MASKS
   can also run in this wave (isolated).
2. **Wave B (FFI-serial):** one owner lands W-DECODE → W-SCAN → W-VALIDATE →
   W-META → W-ENUMS → W-WIRE → W-REMOTE, one at a time, while **no** C-family
   agent is building. Each W is independently shippable.
3. **Wave C:** after each W lands, dispatch its C++ and Fortran wrappers
   (sequentially between CPP and FTN; parallel with PY/TS work).
4. **Wave E — not scheduled:** convert is an accepted native-library exception
   (DESIGN.md § *Documented exceptions* `[A]`); a separate opt-in
   `tensogram-convert-ffi` is the escalation path, tracked in `plans/TODO.md`.
5. **Wave F** interleaved — every feature landed must gain an example in each
   language that supports it (symmetry contract).

### 8.4 Done criteria

- The DESIGN.md matrix cell moves to ● (or documented [L]/—) for the touched
  capability in every applicable binding, with a test and an example.
- The accepted-exceptions list in DESIGN.md is the only place a non-● cell is
  allowed to remain.
- Re-audit the full matrix at the next release.

## 9. Deferred Wave B — detailed implementation plan

> **Status: PLANNED, not started.** One combined PR (`feat/symmetry-wave-b-deferred`).
> Design settled by the questions in the 0.24.x planning round; recorded here so
> the ABI decisions are unambiguous before any code lands.

### 9.1 Settled design decisions

| # | Decision |
|---|----------|
| Frame walker — scope | **Core-first**: add a public `frames()` iterator to the Rust core, then mirror in **all six** bindings (no new asymmetry). |
| Frame walker — granularity | **Per-message**: the walker iterates the frames of a single message. Callers get message byte-ranges from the existing `scan` (`tgm_scan`), then frame-walk each. |
| Frame walker — traversal | **Lazy iterator** handle (`create` / `next` / `free`), not a pre-collected index. |
| Frame entry — contents | **Structural + borrowed payload**: `{ frame_type, version, flags, offset, length }` **and** a borrowed `payload` (content bytes, i.e. header + type-specific footer stripped). `offset`/`length` give the whole-frame span; `payload` gives the content. |
| Frame walk — envelope | **Only the type 1–9 `FR` frames.** The preamble/postamble are exposed via a **separate message-header accessor** (below), not as frame entries. |
| Message flags | **Typed booleans** (not a raw bitset): a `MessageHeader` value with named boolean members (`has_header_metadata`, …) plus `version` and `total_length`. |
| Sync remote — storage_options | **Parallel arrays** `keys[]` / `values[]` + `count` → Rust `BTreeMap<String,String>`. |
| Sync remote — open API | **Separate** `is_remote_url` + `open_remote`; the local `open` stays local-only. |
| Sync remote — gating | **Opt-in `remote` feature**, *not* in the default published FFI tarball (same model as `async`). Header still declares the symbols (cbindgen). |
| Encode setters | **New `TgmEncodeOptions` struct** (hash algo + `aggregate_hash` + `compression_backend` + the existing mask fields) + `*_with_encode_options` entry points; keep the mask-only entry points. |
| Typed dtype/byte-order | **Include** `tgm_dtype` / `tgm_byte_order` enums + `_enum` accessors (alongside the string getters), each with an enum-consistency guard. |
| Delivery | **One combined PR**, grouped commits per component. |

**Sub-decisions implied by the above (specified so nothing is ambiguous):**
- **Iterator input.** `frames(msg: &[u8])` / `tgm_frame_iter_create(const uint8_t* msg, size_t msg_len)` where `msg` points at the message preamble (the slice a caller obtains from `scan`). Frame `offset` is **relative to `msg`** (the message start), `payload`/frame pointers borrow `msg`.
- **Lifetime (C).** The iterator borrows `msg` (which must outlive the iterator). `tgm_frame_iter_next(it, TgmFrame* out) -> bool` fills a caller-provided `TgmFrame`; `out->payload` / whole-frame bytes point **into the caller's `msg`** and are valid for as long as `msg` lives (independent of `next`/`free`). `free` releases only the cursor.
- **Message header (C).** `tgm_message_header(const uint8_t* msg, size_t msg_len, TgmMessageHeader* out) -> tgm_error`, filling a POD with the eight `bool has_*` fields + `uint16_t version` + `uint64_t total_length`. The "typed booleans" are struct members (one parse), the cross-language-friendly form.

### 9.2 Component A — Frame walker + message-header introspection

**A0 · Rust core (foundational — everything else depends on it).**
- New public `FrameInfo<'a> { frame_type: FrameType, version: u16, flags: FrameFlags, offset: usize, length: usize, payload: &'a [u8] }` and `FrameIter<'a>` (a lazy `Iterator<Item = Result<FrameInfo>>`) over one message's bytes, built on the existing `Preamble::read_from` + `FrameHeader::read_from` + `read_frame` (make `read_frame` reusable). `pub fn frames(message: &[u8]) -> Result<FrameIter<'_>>`.
- Expose a public `MessageHeader { flags: MessageFlags, version: u16, total_length: u64 }` (or reuse `Preamble` + `MessageFlags`, which are already `pub`) with typed predicates. Re-export from `lib.rs`.
- Tests: golden multi-frame message → assert the exact frame-type sequence, offsets/lengths tile the message, payload slices match, header booleans match the encode mode (streaming vs random-access).

**A1 · C ABI** (`tensogram-ffi`):
- `tgm_frame_type` enum (`TGM_FRAME_TYPE_HEADER_METADATA=1 … _NTENSOR=9`) — cbindgen `[export.rename]`; **enum-consistency guard** (shell check like `check_value_type_enum.sh`).
- `TgmFrame { tgm_frame_type frame_type; uint16_t version; uint16_t flags; size_t offset; size_t length; const uint8_t* payload; size_t payload_len; }`.
- `tgm_frame_iter_create(const uint8_t* msg, size_t msg_len) -> tgm_frame_iter_t*` (NULL on bad preamble), `tgm_frame_iter_next(it, TgmFrame* out) -> bool`, `tgm_frame_iter_free(it)`.
- `TgmMessageHeader` POD + `tgm_message_header(msg, msg_len, &out)`; plus `tgm_frame_has_hash(const TgmFrame*)` convenience over the per-frame `HASH_PRESENT` bit.
- FFI tests (golden frame sequence, lifetime, null-safety); regen header; header-drift guard.

**A2 · C++**: `frame` struct + `frame_iterator` (input-iterator; `frames(std::span<const uint8_t>)` range-for) and a `message_header` struct with `[[nodiscard]] bool has_*()` + `version()`/`total_length()`. Tests + example.

**A3 · Fortran**: `type(tensogram_frame_iterator)` (`%next(frame, found)` where `frame` is a `type(tensogram_frame)` with `%frame_type()`, `%offset()`, `%length()`, `%payload(bytes)`, `%version()`, `%flags()`), plus `tensogram_message_header(msg, header)` → `type(tensogram_message_header)` with `%has_header_metadata()` … type-bound logicals. Public `TGM_FRAME_TYPE_*` params + f2008 clean + enum guard. Tests + example.

**A4 · Python**: `frames(buf) -> Iterator[Frame]` (`Frame` with `.frame_type` (enum/int), `.offset`, `.length`, `.flags`, `.payload` `bytes`/memoryview) and `message_header(buf) -> MessageHeader` (dataclass/NamedTuple with the boolean fields). pytest + example.

**A5 · TypeScript**: `frames(buf) -> Frame[]` (or a generator) via a small wasm walker, each `Frame` `{ frameType, version, flags, offset, length, payload: Uint8Array }`; `messageHeader(buf) -> MessageHeader` (`{ hasHeaderMetadata, …, version, totalLength }`). vitest + example.

### 9.3 Component B — Synchronous remote (opt-in `remote` feature)

- **C ABI** (`#[cfg(feature = "remote")]`, cbindgen emits unconditionally): `bool tgm_is_remote_url(const char* source)`; `TgmRemoteScanOptions { bool bidirectional; }`; `tgm_error tgm_file_open_remote(const char* source, const char* const* keys, const char* const* values, size_t n, const TgmRemoteScanOptions* opts, tgm_file_t** out)` — returns a normal sync `tgm_file_t` so the existing file API works on remote sources. Error mapping → `TGM_ERROR_REMOTE`. New `remote` feature on `tensogram-ffi` enabling `tensogram/remote` (sync, **without** `async`).
- **C++**: `bool is_remote_url(std::string_view)`; `file file::open_remote(std::string_view source, const std::map<std::string,std::string>& storage_options = {}, remote_scan_options = {})`. Guarded so a build without the feature degrades to a clear error.
- **Fortran**: `tensogram_is_remote_url(url)` + `tensogram_file_open_remote(source, keys, values, [bidirectional], file, err)`.
- **Python / TypeScript**: already have full sync/lazy remote — **no work** beyond confirming parity in the matrix.
- **Distribution note** (docs): the default `tensogram-ffi` tarball stays remote-free; a `remote`-enabled build is opt-in (mirrors async). Record in `docs/src/guide/c-api.md`.

### 9.4 Component C — Typed enums + encode-option setters

- **Enums** (C ABI, each cbindgen-renamed + an enum-consistency guard): `tgm_dtype` (15 variants), `tgm_byte_order` (`LITTLE`/`BIG`), `tgm_aggregate_hash_policy` (`AUTO`/`NONE`/`HEADER`/`FOOTER`/`BOTH`), `tgm_compression_backend` (`AUTO`/`FFI`/`PURE`).
- **Accessors**: `tgm_object_dtype_enum(msg,i) -> tgm_dtype`, `tgm_object_byte_order_enum(msg,i) -> tgm_byte_order` (the string getters stay).
- **`TgmEncodeOptions`** POD: `{ const char* hash; tgm_aggregate_hash_policy aggregate_hash; tgm_compression_backend compression_backend; bool allow_nan; bool allow_inf; tgm_mask_method nan_mask_method; … small_mask_threshold_bytes; }` (supersedes `TgmEncodeMaskOptions`; keep the old struct + entry points). New `tgm_encode_with_encode_options` / `tgm_file_append_with_encode_options` / `tgm_streaming_encoder_create_with_encode_options`.
- **Cross-language**: expose the four enums + the two new encode knobs in C++ (`enum class`), Fortran (integer params + option fields), Python (accept `aggregate_hash=`/`compression_backend=` kwargs + a `Dtype`/`ByteOrder` enum), TS (union types / enums + option fields). Some already have partial coverage (e.g. Python already surfaces most `EncodeOptions`); fill the two missing knobs everywhere.

### 9.5 Dependency order & concurrency

1. **A0 (core frame iterator + message header)** — foundational, land + commit first.
2. Then **parallel** (disjoint crates): **A1+B+C on the C ABI** (one FFI owner, serial within the FFI), **A4 Python** (binds core), **A5 TS** (binds core via a small wasm walker). Python/TS also pick up the Component-C enum/setters.
3. After the C ABI lands: **A2/B/C C++**, then **A3/B/C Fortran** (sequential — shared C-API build).
4. Enum-consistency guards (frame-type, dtype, byte-order) added with each enum; header-drift guard must stay green.

### 9.6 Testing & examples

- Golden **frame-sequence** fixtures (random-access vs streaming message) shared across bindings, asserting identical frame types/offsets/flags and header booleans.
- Remote: reuse the existing remote test harness; add a sync `open_remote` round-trip (local `file://` + a mock/object-store path) in C/C++/Fortran.
- Encode setters: assert `aggregate_hash = Both` places hashes in header *and* footer (inspectable via the new frame walker — nice closed loop), and `compression_backend = Pure` selects the pure codec.
- One example per binding for the frame walker + message header; extend an encode example for the new knobs.

### 9.7 Risks / notes

- **Frame-walker lifetime** is the sharpest edge (borrowed payload into the caller's buffer). Document it prominently in every binding; the C++ `frame_iterator` and Python `Frame.payload` must not outlive the source buffer.
- **Enum drift**: three new C enums ⇒ three new consistency guards (Fortran already has the pattern). Non-negotiable to avoid silent skew.
- **Combined-PR size**: this is large (frame walker × 6 bindings + sync remote × 3 + enums/setters × bindings). If review becomes unwieldy, it can still be *reviewed* commit-by-commit along the A/B/C grouping without splitting the branch.

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

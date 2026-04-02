# Tensogram — Code Quality Improvements

> Generated from deep code review (2026-04-02). Covers all 29 .rs files across 5 crates.
> Baseline: 0 clippy warnings, 52 tests passing, 0 compiler errors.

## Summary

The architecture is clean and the module split is sound. The problems are **semantic correctness risks**, not style issues. The byte-order gap, missing validation, and panic paths are the kind of bugs that silently produce wrong results in production.

**Priority: fix correctness before adding features.**

---

## PR 1: Eliminate panic paths from public APIs

**Severity: 🔴 Critical — library code must not crash on untrusted input**
**Effort: ~2h**

- [ ] `shuffle.rs:8-14` — Replace `assert_eq!` with `Result<Vec<u8>, ShuffleError>` in `shuffle()`
- [ ] `shuffle.rs:31-37` — Replace `assert_eq!` with `Result<Vec<u8>, ShuffleError>` in `unshuffle()`
- [ ] Add `ShuffleError` enum (InvalidElementSize, Misaligned)
- [ ] Update `pipeline.rs` to propagate shuffle errors through `PipelineError`
- [ ] `framing.rs:111-125` — Validate monotonicity of object offsets before slicing
- [ ] `framing.rs:145-155` — Bounds-check object regions to prevent slicing panics from crafted headers
- [ ] `file.rs:63,87,104` — Replace `.unwrap()` with `.expect("message_offsets set by ensure_scanned")` or propagate error

---

## PR 2: Add metadata/payload validation boundary

**Severity: 🔴 Critical — no invariants enforced, trivially mis-constructable metadata**
**Effort: ~4h**

- [ ] Add `validate_object(desc, data_len)` function that checks:
  - `ndim == shape.len()`
  - `strides.len() == shape.len()`
  - `shape.product() × dtype.byte_width() == payload.len()` (for unencoded data, non-bitmask)
  - All shape/stride dimensions fit in `usize` (checked arithmetic)
  - Element count doesn't overflow `usize`
- [ ] Call `validate_object()` in `encode()` before encoding each object
- [ ] Call `validate_object()` in `decode()` / `decode_object()` after parsing metadata
- [ ] Validate `version` field is a known value (currently ignored)
- [ ] Validate `obj_type` is a known value or at least non-empty
- [ ] Validate `encoding`, `filter`, `compression` strings are known values in `build_pipeline_config` (already done, keep)
- [ ] Validate numeric params are in-range before casting (bits_per_value ≤ 64, scale factors fit i32, etc.)

---

## PR 3: Fix byte order and dtype handling in pipeline

**Severity: 🔴 Critical — silent data corruption across platforms**
**Effort: ~6h**

- [ ] `pipeline.rs:138-145` — `bytes_to_f64()` uses `from_ne_bytes`; must respect declared `byte_order`
- [ ] `pipeline.rs:144-146` — `f64_to_bytes()` uses `to_ne_bytes`; must respect declared `byte_order`
- [ ] `encode.rs:97-147` — `build_pipeline_config()` ignores `PayloadDescriptor.byte_order`; thread it through
- [ ] Add `byte_order` field to `PipelineConfig`
- [ ] Implement endian-aware conversion helpers: `bytes_to_f64_endian(data, byte_order)` etc.
- [ ] Reject or handle `simple_packing` with non-f64 dtypes explicitly (currently silently misinterprets)
- [ ] Add `dtype` to `PipelineConfig` so the pipeline knows the source element width
- [ ] Add cross-endian round-trip tests (encode big-endian, decode on native)

---

## PR 4: Replace lossy `as` casts with checked conversions

**Severity: 🟡 High — silent truncation on untrusted metadata**
**Effort: ~3h**

- [ ] `encode.rs:155` — `get_i64_param() as i32` → use `i32::try_from()` with error
- [ ] `encode.rs:156` — `get_i64_param() as i32` → use `i32::try_from()` with error
- [ ] `encode.rs:157` — `get_u64_param() as u32` → use `u32::try_from()` with error
- [ ] `encode.rs:130-132` — szip params `as u32` → use `u32::try_from()` with error
- [ ] `encode.rs:117` — shuffle element_size `as usize` → use `usize::try_from()` with error
- [ ] `encode.rs:166` — `i128 as f64` → validate range before cast
- [ ] `encode.rs:181` — `i128 as i64` → use `i64::try_from()` with error
- [ ] `encode.rs:196` — `i128 as u64` → use `u64::try_from()` with error (rejects negative CBOR ints)
- [ ] `output.rs:73` — `i128 as i64` → use `i64::try_from()` or preserve as string
- [ ] `framing.rs` — audit all `u64 as usize` casts (lines 26, 68, 77, 86-98, 145-149, 170)
- [ ] `decode.rs:66,156` — `shape.product::<u64>() as usize` → use `usize::try_from()` with error
- [ ] `decode.rs:129-130` — `offset as usize * element_size` → use `checked_mul()` to prevent overflow

---

## PR 5: Fix `decode_range` bug with shuffle filter

**Severity: 🟡 High — returns wrong data**
**Effort: ~30min**

- [ ] `decode.rs:96-109` — The guard only rejects shuffle when `encoding != "none" || compression != "none"`. For the case `encoding=none, compression=none, filter=shuffle`, it slices shuffled bytes directly and returns nonsense
- [ ] Add explicit rejection: `if payload_desc.filter != "none" { return Err(...) }`
- [ ] Add test: encode with shuffle + no encoding/compression, attempt `decode_range`, verify error

---

## PR 6: Improve `TensogramFile` I/O scalability

**Severity: 🟡 Medium — blocks adoption for large files**
**Effort: ~3h**

- [ ] `file.rs:54-56` — `ensure_scanned()` reads entire file into memory; switch to streaming scan with `BufReader`
- [ ] `file.rs:85-98` — `read_message()` reads entire file; switch to `seek()` + `read_exact()` for the target range
- [ ] `file.rs:101-109` — `messages()` clones all message bytes; provide iterator-based API or lazy access
- [ ] Cache `File` handle instead of re-opening on each read
- [ ] Remove or deprecate `messages() -> Vec<Vec<u8>>` (forces full file load + clone)

---

## PR 7: Fix CLI multi-object and `set` semantics

**Severity: 🟡 Medium — wrong behavior for multi-object messages**
**Effort: ~2h**

- [ ] `filter.rs:53-64` — `lookup_key()` returns first matching object only; document behavior or return all matches
- [ ] `commands/set.rs:80-117` — `apply_mutation()` writes to top-level metadata only, never per-object `extra`; support dot-notation for object-level keys
- [ ] `commands/set.rs:66-74` — Pass-through path double-opens the output file; open once
- [ ] `commands/set.rs:60-64` — Re-encoding drops hashes even when payload is untouched; preserve existing hash when only metadata changes
- [ ] `commands/copy.rs:64-72` — `expand_placeholders()` inherits first-object-wins ambiguity; document or fix

---

## PR 8: Code quality cleanup

**Severity: 🟢 Low — lint-level improvements**
**Effort: ~2h**

- [ ] Remove dead public type `DataObject` from `types.rs` and `lib.rs` re-export (unused anywhere)
- [ ] `output.rs:6,14` — Change `&[String]` params to `&[&str]` or `&[impl AsRef<str>]`
- [ ] `output.rs:24,33,45,57,70` and `filter.rs:95` — Replace `.clone()` with `.into()` or `Cow` where possible
- [ ] `wire.rs:59` — Replace `for i in 0..num_objects as usize` with iterator chain
- [ ] `decode.rs:25` — Replace `for i in 0..metadata.objects.len()` with `.enumerate()`
- [ ] `simple_packing.rs:32-36, 86-90` — NaN check loop could use `.enumerate().find()` or `.position()`
- [ ] Add doc comments to: `error.rs`, `commands/info.rs`, `commands/ls.rs`, `commands/dump.rs`, `commands/get.rs`, `commands/set.rs`
- [ ] `compression.rs` — Add `#[allow(dead_code)]` or remove unused fields on `SzipCompressor` (rsi, block_size, flags)
- [ ] `metadata.rs:57-58` — Canonicalize sort silently ignores `ciborium::into_writer` errors; at minimum add `debug_assert!`

---

## PR 9: Backfill adversarial tests

**Severity: 🟢 Low — safety net for the fixes above**
**Effort: ~3h**

- [ ] Test malformed binary header (decreasing object offsets, overlapping regions)
- [ ] Test negative CBOR integers in metadata params (should error, not wrap)
- [ ] Test non-f64 dtype with simple_packing (should error, not silently corrupt)
- [ ] Test oversized `total_length` / `num_objects` values (DoS via allocation)
- [ ] Test `shuffle` with `element_size = 0` (should error, not panic)
- [ ] Test `shuffle` with non-divisible data length (should error, not panic)
- [ ] Test `decode_range` with shuffle filter (should error, not return garbage)
- [ ] Test cross-endian round-trip (once PR 3 is done)
- [ ] Test payload size mismatch vs shape × dtype (once PR 2 is done)
- [ ] Test `TensogramFile` with file modified between operations (invalidation)
- [ ] Property-based test: arbitrary valid Metadata round-trips through CBOR

---

## Priority Order

| Order | PR | Risk | Effort |
|:---:|---|---|:---:|
| 1 | PR 1: Eliminate panics | 🔴 Crashes on untrusted input | ~2h |
| 2 | PR 2: Validate metadata | 🔴 Silent garbage in, garbage out | ~4h |
| 3 | PR 3: Fix byte order + dtype | 🔴 Silent cross-platform corruption | ~6h |
| 4 | PR 4: Checked casts | 🟡 Silent truncation | ~3h |
| 5 | PR 5: Fix decode_range | 🟡 Returns wrong data | ~30min |
| 6 | PR 7: Fix CLI semantics | 🟡 Wrong multi-object behavior | ~2h |
| 7 | PR 6: File I/O scalability | 🟡 Blocks large-file usage | ~3h |
| 8 | PR 9: Adversarial tests | 🟢 Safety net | ~3h |
| 9 | PR 8: Code cleanup | 🟢 Polish | ~2h |

**Total estimated effort: ~25h across 9 PRs.**
**Critical path (PRs 1-3): ~12h.**

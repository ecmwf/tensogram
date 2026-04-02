# Tensogram Improvements — Notepad

## Baseline (2026-04-02)
- Tests: 52 passing (3 CLI + 21 core-unit + 14 core-integration + 14 encodings)
- Clippy: 0 warnings
- Compiler: 0 errors

## Conventions
- Error types use `thiserror` derive — see `PackingError` in `simple_packing.rs` as pattern
- `PipelineError` → `TensogramError` conversion is manual `.to_string()` — DO NOT refactor to `#[from]`
- Test helpers pattern: see `make_float32_metadata`, `make_mars_metadata` in `tests/integration.rs`
- Commit format: `type(scope): description`

## Guardrails
- NO changes to `tensogram-python` or `tensogram-ffi` beyond keeping compilation
- NO new external dependencies
- NO sweeping rewrites of modules
- Framing.rs: only fix casts on UNTRUSTED DECODE paths, not all casts

## Key File Locations
- shuffle: `crates/tensogram-encodings/src/shuffle.rs`
- pipeline: `crates/tensogram-core/src/pipeline.rs`
- framing: `crates/tensogram-core/src/framing.rs`
- encode: `crates/tensogram-core/src/encode.rs`
- decode: `crates/tensogram-core/src/decode.rs`
- file: `crates/tensogram-core/src/file.rs`
- types: `crates/tensogram-core/src/types.rs` (ByteOrder enum already exists here)
- CLI set: `crates/tensogram-cli/src/commands/set.rs`
- CLI filter: `crates/tensogram-cli/src/filter.rs`
- integration tests: `crates/tensogram-core/tests/integration.rs`

## Wave Progress
- Wave 1: PENDING (Tasks 1, 2, 3)
- Wave 2: PENDING (Tasks 4, 5, 6)
- Wave 3: PENDING (Task 7)
- Wave 4: PENDING (Tasks 8, 9)

## PR 8 Cleanup Notes (2026-04-02)
- `DataObject` is unused outside `tensogram-core`; grep found only the definition and the re-export.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` is clean after the cleanup.
- `cargo test --workspace` still passes (52 tests).

## Wave 1 — PR 1 Completed (2026-04-02)

### ShuffleError pattern
- Added `ShuffleError` enum in `crates/tensogram-encodings/src/shuffle.rs`
- `shuffle()` and `unshuffle()` now return `Result<Vec<u8>, ShuffleError>`
- Clippy linted `data.len() % element_size != 0` → use `!data.len().is_multiple_of(element_size)`
- `PipelineError::Shuffle(String)` variant added; mapped via `.map_err(|e| PipelineError::Shuffle(e.to_string()))?`
- Never use `#[from]` for ShuffleError → TensogramError chain — always string-map

### Framing hardening
- Added monotonicity check in `decode_frame` BEFORE the OBJS/OBJE loop — catch non-monotonic offsets early
- Added bounds check in `extract_object_payload` to prevent `&buf[start..end]` panic when start > end
- Test crafts a "tricky" buffer where OBJS/OBJE markers are placed to fool the loop without the monotonicity guard

### file.rs
- `.unwrap()` on `message_offsets` → `.expect("message_offsets set by ensure_scanned")`
- Invariant is guaranteed by `ensure_scanned()` always running first

### TDD commit discipline
- "test: add failing tests" first (code compiles, tests fail at runtime)
- "fix(scope): description" second (tests go green)
- Wave 1 baseline after PR 1: 56 tests (was 53 before new tests added), 0 clippy warnings

## CLI metadata mutation notes (2026-04-02)
- `set` must copy existing `payload.hash` back after `encode(..., hash_algorithm: None)`, otherwise metadata-only edits silently drop integrity hashes.
- `lookup_key()` and `expand_placeholders()` both document first-match behavior now, which matters for multi-object messages with repeated namespaced keys.
- `rust-analyzer` is not installed in this workspace, so `lsp_diagnostics` on `.rs` files fails even when `cargo test`/`clippy` are green.

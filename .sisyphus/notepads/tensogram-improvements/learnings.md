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

---
description: Audit error paths across Rust / Python / C++ / FFI and tighten up
agent: build
---

# Improve Error Handling

Perform a comprehensive error handling audit across the entire codebase.

## Checks

### Rust
- Verify NO panics in library code — no `unwrap()`, `expect()`, `panic!()`, `todo!()` outside of tests
- All `Result` types propagated correctly with `?` operator
- Error types carry enough context for users to diagnose problems (file paths, indices, values)
- Error chains preserved — no `map_err(|_| ...)` that silently drops the cause
- FFI boundary: all Rust errors mapped to C error codes with `tgm_last_error()` message

### Python
- All Rust → Python errors surface as appropriate Python exceptions (ValueError, IOError, RuntimeError)
- Error messages match between languages for the same underlying error
- No bare `except:` or overly broad exception catching

### C / C++
- All FFI functions return error codes, never crash
- `tgm_last_error()` always set with a descriptive message on failure
- C++ wrappers throw typed exceptions with the original error message

### Documentation
- All error conditions documented in `docs/src/guide/error-handling.md`
- Each public API function's doc comments list possible errors
- Examples show proper error handling patterns

## Process

1. Scan all source files for panic sites, unwraps, and error-swallowing patterns
2. For each finding: classify severity (crash, silent failure, poor message, undocumented)
3. Fix each issue — propagate errors properly, improve messages, add docs
4. Run all tests to verify no regressions
5. Update error handling documentation in docs/
6. Summarize all changes made

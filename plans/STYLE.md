# Coding Style

Follow `DESIGN.md` principles. When in doubt, read the existing code.

## Principles

1. **Deep modules.** Hide complexity behind an API with a small surface.
2. **Design errors out of existence.** Idempotent ops, never panic, always return Result/error codes.
3. **Stateless.** No global state (thread-local error string in FFI is the sole exception).
4. **Self-contained data objects.** Every data object frame carries all information needed to decode it.

## Rust

- All public functions return `Result<T, TensogramError>` — no `unwrap()` in library code
- No panics in library code — `panic = "abort"` in both release and dev profiles (FFI safety)
- Feature-gate optional dependencies (`#[cfg(feature = "...")]`)
- Prefer `BTreeMap` over `HashMap` for deterministic serialization
- Derive `Debug` on public types

## C FFI

- All functions prefixed `tgm_`
- Opaque handle pattern — callers never see internal structs
- Error codes + `tgm_last_error()` for thread-local error messages
- Every allocation has a matching `_free()` function

## C++ Wrapper

- RAII with `std::unique_ptr` + custom deleters
- Move-only semantics (copy suppressed)
- `[[nodiscard]]` on all accessors
- Typed exception hierarchy mapping C error codes

## Python

- NumPy arrays as the primary data interface
- Context manager for file I/O
- ruff for linting/formatting (E/W/F/I/N/UP/B/SIM/PT/RUF rules)

## Functions

- Prefer short functions (10–50 lines)
- Decompose complex functions into focused helpers

## Comments

- Explain *why*, not *what*
- Document non-obvious design decisions inline
- Doxygen/rustdoc for public API

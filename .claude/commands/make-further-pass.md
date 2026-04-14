# Further Pass Review

Perform a quality review pass over the codebase or the specified area.

**This is pass number $ARGUMENTS (default: 2 if not specified).**

Track the pass number and increase strictness with each successive pass:

## Pass 1-2 (Foundation)
- Simplification opportunities — reduce complexity, remove redundancy
- Naming quality — variables, functions, types, modules
- Comments and doc quality — accurate, helpful, not redundant
- Running required formatter/lint/tests

## Pass 3-4 (Hardening)
Everything from Pass 1-2, PLUS:
- Scan for edge-cases and logical regressions
- No panics in Rust code (`unwrap()`, `expect()`, `panic!()`)
- All documentation up-to-date with changes
- Error messages are clear and actionable
- Public API surface is minimal and clean
- No dead code, no unused imports, no stale TODOs

## Pass 5+ (Polish)
Everything from Pass 3-4, PLUS with ZERO TOLERANCE:
- Every public function has doc comments
- Every error path is tested
- Every edge case has a test
- Code reads like well-written prose — a newcomer could understand it
- Performance: no unnecessary allocations, clones, or copies
- Consistency: similar patterns handled the same way everywhere
- Cross-language parity: Rust, Python, C FFI, C++ all expose equivalent functionality

## Process

1. State which pass number this is and what strictness level applies
2. Scan the codebase (or specified files/area)
3. List all findings grouped by category
4. Fix each finding, verifying the fix compiles and tests pass
5. Run all formatters and linters: `cargo fmt`, `cargo clippy`, `ruff check`, `ruff format`
6. Run all tests: `cargo test --workspace`, `python -m pytest tests/python/ -v`
7. Summarize what was changed and what the next pass should focus on

# Improve Code Coverage

Perform a comprehensive code coverage analysis and fill gaps to reach 95%+ coverage.

## Process

### 1. Measure Current Coverage

**Rust:**
```
cargo llvm-cov --workspace --all-features --summary-only
```

**Python:**
```
python -m pytest tests/python/ tensogram-xarray/tests/ --cov --cov-report=term-missing
python -m pytest tensogram-zarr/tests/ --cov --cov-report=term-missing
```

### 2. Identify Gaps

For each file below 95% coverage:
- Read the source file to understand what code paths are untested
- Categorize each gap as:
  - **(a) Needs new tests** — testable code paths with no coverage
  - **(b) Error handling** — defensive paths that are hard but possible to trigger
  - **(c) Dead code** — unreachable code that should be removed
  - **(d) Platform-specific** — code only reachable on specific platforms

### 3. Prioritize by Impact

Focus on the files with the largest absolute gaps first:
- FFI layer (C API functions)
- Core library (encode/decode/validate/remote/file)
- Python backends (xarray/zarr)

### 4. Write Tests

- Add tests to EXISTING test files, matching the existing patterns and style
- Each test should target a specific uncovered code path
- Test error paths, not just happy paths
- Use adversarial inputs for validation code

### 5. Remove Dead Code

If coverage analysis reveals unreachable code:
- Verify it's truly unreachable (not just untested)
- Remove it rather than adding `# pragma: no cover`

### 6. Verify

- Run full test suite to confirm all tests pass
- Re-measure coverage to confirm improvement
- Report before/after comparison by file

## Target

Aim for at least **95% line coverage** across the project. Acceptable exceptions:
- CLI `main.rs` (binary entrypoint)
- Platform-specific code paths
- Benchmark runner code

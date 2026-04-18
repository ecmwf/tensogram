# Prepare and Create Pull Request

Final preparation check and PR creation workflow.

## Pre-flight Checks

Run ALL of the following. If ANY step fails, STOP and report the failure.

### 1. Rust
```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test -p tensogram --features "remote,async"
```

### 2. Python
```bash
source .venv/bin/activate
ruff check --config python/bindings/pyproject.toml python/tests/
ruff format --check --config python/bindings/pyproject.toml python/tests/
python -m pytest python/tests/ -v
python -m pytest python/tensogram-xarray/tests/ -v
python -m pytest python/tensogram-zarr/tests/ -v
```

### 3. Examples
```bash
cargo build --workspace  # Rust examples built as part of workspace
```

### 4. Documentation
```bash
# Verify docs build if mdbook is available
which mdbook && mdbook build docs/ || echo "mdbook not installed, skip"
```

## PR Creation

If all checks pass:

1. Review what files to include — stage only source, test, doc, and config files
2. Do NOT stage: build artifacts, hidden directories (except `.claude/`, `.github/`), `*.so`, `*.dylib`, `target/`, `.venv/`, `Cargo.lock` inside subcrates
3. If not already on a feature branch, create one with a descriptive name
4. Commit with a clear, conventional-commit-style message
5. Push to origin
6. Create a pull request with:
   - Title: concise summary of the change
   - Body: structured summary with Added/Changed/Fixed sections
   - Link to any related issues
7. Report the PR URL

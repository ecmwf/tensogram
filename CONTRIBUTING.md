# Contributing to Tensogram

Thank you for your interest in contributing. This guide will get you from zero to a passing test suite in a few minutes.

## Prerequisites

- **Rust 1.75+** (`rustup install stable`)
- **C compiler** (for libaec, zfp, blosc2 FFI dependencies)
- Optional: Python 3.9+ and `uv` for Python bindings (`pip install uv` or https://docs.astral.sh/uv/getting-started/installation/)
- Optional: mdbook for documentation (`cargo install mdbook`)

## Quick Setup

```bash
git clone https://github.com/ecmwf/tensogram.git
cd tensogram

# Build everything
cargo build --workspace

# Run the test suite
cargo test --workspace

# Check formatting and lints
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

If all of that passes, you're ready to go.

## Project Structure

```
tensogram/
├── crates/
│   ├── tensogram-core/         # Core library: wire format, encode/decode, file API
│   │   ├── src/                # 11 modules (wire, framing, metadata, types, ...)
│   │   └── tests/              # Integration tests + golden binary files
│   ├── tensogram-encodings/    # Encoding pipeline: packing, shuffle, compression
│   ├── tensogram-cli/          # CLI tool (tensogram info/ls/dump/get/set/copy)
│   ├── tensogram-ffi/          # C FFI layer (generates tensogram.h via cbindgen)
│   └── tensogram-python/       # Python bindings (PyO3, excluded from default build)
├── examples/
│   ├── rust/                   # 10 runnable Rust examples
│   ├── cpp/                    # C++ examples using the FFI
│   └── python/                 # Python examples using the PyO3 bindings
├── docs/                       # mdbook documentation source
├── plans/                      # Design docs, implementation status, TODOs
├── ARCHITECTURE.md             # How the crates fit together
├── CHANGELOG.md                # Release history
└── VERSION                     # Current version
```

## Development Workflow

### 1. Make your changes

Work on a branch. The codebase follows these conventions:

- **No panics in library code.** All fallible operations return `Result<T, TensogramError>`.
- **Immutability by default.** Use `let` unless mutation is required.
- **Short functions.** 10-50 lines typical, extract helpers when a function grows past that.
- **CBOR determinism.** All metadata encoding goes through the canonicalization step.

See [plans/STYLE.md](plans/STYLE.md) for the full style guide.

### 2. Run the checks

Before submitting, run all four:

```bash
cargo build --workspace
cargo fmt
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

All four must pass with zero warnings. The CI pipeline runs the same commands.

### 3. Test optional features

If your change touches file I/O or adds new public API:

```bash
# Test with memory-mapped I/O support
cargo test -p tensogram-core --features mmap

# Test with async support
cargo test -p tensogram-core --features async

# Test with both
cargo test -p tensogram-core --features mmap,async
```

### 4. Update documentation

- **Code changes** should be reflected in `plans/DONE.md` (implementation status)
- **New public API** should have doc comments and appear in the mdbook guide under `docs/src/`
- **Edge cases** go in `docs/src/edge-cases.md`
- **Wire format changes** go in `docs/src/format/wire-format.md`

Build the docs to check:

```bash
cd docs && mdbook build
cargo doc --workspace --no-deps
```

### 5. Add examples

If you add a new API surface, add a runnable example in `examples/rust/`. The naming convention is `NN_description.rs` (e.g. `11_new_feature.rs`).

## Test Structure

| Location | Type | Count | Purpose |
|----------|------|-------|---------|
| `crates/tensogram-core/src/*.rs` | Unit | ~57 | Module-level tests alongside the code |
| `crates/tensogram-core/tests/integration.rs` | Integration | ~12 | Full encode/decode round-trips |
| `crates/tensogram-core/tests/adversarial.rs` | Adversarial | ~12 | Corrupted inputs, boundary conditions |
| `crates/tensogram-core/tests/golden_files.rs` | Golden | 6 | Deterministic binary file verification |
| `crates/tensogram-encodings/src/*.rs` | Unit | ~47 | Encoding pipeline tests |
| `crates/tensogram-cli/src/*.rs` | Unit | 5 | CLI argument parsing |

Golden binary files in `tests/golden/` are checked into the repo. If the wire format changes, regenerate them by running `cargo test --test golden_files`.

## Python Bindings

The Python crate is excluded from the default workspace build because it requires a Python interpreter and linker:

```bash
# First time: create a virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install maturin numpy pytest ruff

# Build and install the Rust extension into the active venv
cd crates/tensogram-python && maturin develop && cd ../..

# Run core Python tests
python -m pytest tests/python/ -v

# Optional: install and test xarray/zarr backends
uv pip install -e "tensogram-xarray/[dask]"
python -m pytest tensogram-xarray/tests/ -v
uv pip install -e tensogram-zarr/
python -m pytest tensogram-zarr/tests/ -v
```

> If `uv` is not available, substitute `python -m venv .venv` and `pip install` for the virtualenv and install steps above.

## C/C++ Bindings

The FFI crate generates `tensogram.h` via cbindgen:

```bash
cargo build -p tensogram-ffi
# Output:
#   crates/tensogram-ffi/tensogram.h
#   target/debug/libtensogram_ffi.a
#   target/debug/libtensogram_ffi.{so,dylib}
```

## Commit Messages

Use conventional commits:

```
feat: add new compression codec
fix: handle zero-length payloads in decode
refactor: decompose encode_message into helpers
docs: update wire format documentation
test: add golden binary test files
chore: update CI configuration
```

## Questions?

Open an issue on GitHub. For design questions, check [plans/DESIGN.md](plans/DESIGN.md) first.

# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

# Top-level Makefile dispatching builds and tests for all languages.
# Run `make help` to see available targets.

.PHONY: help all check test lint fmt docs clean \
        rust-check rust-test rust-lint rust-fmt rust-clippy \
        python-build python-dist python-dist-extras python-test python-lint python-fmt \
        cpp-build cpp-test \
        wasm-test docs-build \
        ts-install ts-build ts-test ts-typecheck

# ── Defaults ──────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

all: check test lint ## Run all checks, tests, and lints

# ── Rust ──────────────────────────────────────────────────────────────────

rust-check: ## Check Rust workspace compiles
	cargo check --workspace

rust-test: ## Run all Rust tests
	cargo test --workspace

rust-clippy: ## Run clippy on Rust workspace
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy -p tensogram --all-targets --features "remote,async" -- -D warnings

rust-fmt: ## Check Rust formatting
	cargo fmt --check

rust-lint: rust-clippy rust-fmt ## Run all Rust lints (clippy + fmt)

# ── Python ────────────────────────────────────────────────────────────────

PYTHON  ?= uv run python
MATURIN ?= uvx maturin
RUFF    ?= uv run --with ruff ruff
RUFF_CFG ?= python/bindings/pyproject.toml
# Space-separated --interpreter flags passed to maturin build; defaults to
# auto-discovery. Override in CI: MATURIN_INTERP_ARGS="--interpreter /path/to/py"
MATURIN_INTERP_ARGS ?= --find-interpreter

python-build: ## Build Python bindings via maturin (dev install into .venv)
	if [ ! -d .venv ] ; then uv venv ; fi
	cd python/bindings && $(MATURIN) develop --release
	VERSION=$$(cat VERSION) uv pip install ./python/tensogram-xarray
	VERSION=$$(cat VERSION) uv pip install ./python/tensogram-zarr
	VERSION=$$(cat VERSION) uv pip install ./python/tensogram-anemoi
	VERSION=$$(cat VERSION) uv pip install ./examples/jupyter

python-dist: ## Build tensogram distributable wheels for the current platform
	if [ ! -d .venv ] ; then uv venv ; fi
	cd python/bindings && $(MATURIN) build --release --out dist $(MATURIN_INTERP_ARGS)

python-dist-extras: ## Build distributable wheels for pure-Python extra packages
	uv build python/tensogram-xarray --out-dir dist/extras
	uv build python/tensogram-zarr --out-dir dist/extras
	uv build python/tensogram-anemoi --out-dir dist/extras

python-test: python-build ## Run all Python tests
	# TODO pip installs should come from workspace-level pyproject
	uv pip install pytest pytest-asyncio numpy
	$(PYTHON) -m pytest python/tests/ -v
	$(PYTHON) -m pytest python/tensogram-xarray/tests/ -v
	$(PYTHON) -m pytest python/tensogram-zarr/tests/ -v

python-lint: ## Run ruff check on Python code
	$(RUFF) check --config $(RUFF_CFG) python/tests/ python/tensogram-xarray/ python/tensogram-zarr/

python-fmt: ## Check Python formatting
	$(RUFF) format --check --config $(RUFF_CFG) python/tests/ python/tensogram-xarray/ python/tensogram-zarr/

# ── C++ ───────────────────────────────────────────────────────────────────

cpp-build: ## Build C++ tests via CMake
	cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Debug
	cmake --build build

cpp-test: cpp-build ## Run C++ tests
	cd build && ctest --output-on-failure

# ── WASM ──────────────────────────────────────────────────────────────────

wasm-test: ## Run WASM tests
	wasm-pack test --node rust/tensogram-wasm

# ── TypeScript ────────────────────────────────────────────────────────────

ts-install: ## Install TypeScript wrapper dependencies
	cd typescript && npm ci || npm install --no-audit --no-fund

ts-build: ts-install ## Build the TypeScript wrapper (wasm-pack + tsc)
	cd typescript && npm run build

ts-test: ts-build ## Run TypeScript wrapper tests (vitest)
	cd typescript && npm test

ts-typecheck: ts-build ## Strict typecheck source + tests
	cd typescript && npx tsc --noEmit -p tsconfig.test.json

# ── Docs ──────────────────────────────────────────────────────────────────

docs-build: ## Build mdbook documentation
	mdbook build docs/

# ── Aggregates ────────────────────────────────────────────────────────────

check: rust-check ## Check all builds
test: rust-test python-test ts-test ## Run all tests
lint: rust-lint python-lint python-fmt ts-typecheck ## Run all lints
fmt: rust-fmt python-fmt ## Check all formatting

# ── Cleanup ───────────────────────────────────────────────────────────────

clean: ## Remove build artifacts
	cargo clean
	rm -rf build/
	rm -rf .venv/
	rm -rf docs/book/
	rm -rf python/bindings/target/
	find python/bindings/python -name '*.so' -o -name '*.pyd' | xargs rm -f 2>/dev/null || true
	rm -rf typescript/dist/ typescript/wasm/ typescript/node_modules/
	rm -rf examples/typescript/node_modules/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

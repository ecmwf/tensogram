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
        python-build python-test python-lint python-fmt \
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

rust-fmt: ## Check Rust formatting
	cargo fmt --check

rust-lint: rust-clippy rust-fmt ## Run all Rust lints (clippy + fmt)

# ── Python ────────────────────────────────────────────────────────────────

PYTHON ?= python3
VENV ?= .venv
RUFF_CFG ?= python/bindings/pyproject.toml

python-build: ## Build Python bindings via maturin
	cd python/bindings && maturin develop --release

python-test: ## Run all Python tests
	$(PYTHON) -m pytest python/tests/ -v
	$(PYTHON) -m pytest python/tensogram-xarray/tests/ -v
	$(PYTHON) -m pytest python/tensogram-zarr/tests/ -v

python-lint: ## Run ruff check on Python code
	ruff check --config $(RUFF_CFG) python/tests/ python/tensogram-xarray/ python/tensogram-zarr/

python-fmt: ## Check Python formatting
	ruff format --check --config $(RUFF_CFG) python/tests/ python/tensogram-xarray/ python/tensogram-zarr/

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
	cd typescript && npm install

ts-build: ## Build the TypeScript wrapper (wasm-pack + tsc)
	cd typescript && npm run build

ts-test: ## Run TypeScript wrapper tests (vitest)
	cd typescript && npm test

ts-typecheck: ## Strict typecheck source + tests
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
	rm -rf typescript/dist/ typescript/wasm/ typescript/node_modules/
	rm -rf examples/typescript/node_modules/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

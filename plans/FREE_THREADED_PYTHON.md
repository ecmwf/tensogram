# Free-Threaded Python Support Plan

**Branch:** `feature/free-threaded-python` (from `feature/remote-polish`)
**Goal:** Enable tensogram's Python bindings to run without the GIL on Python 3.13t/3.14t, with benchmarks proving the performance gain.
**Reviewed by:** Oracle (deep review completed, corrections incorporated)

---

## 1. Background & Motivation

CPython 3.13 introduced an experimental "free-threaded" build (PEP 703) that removes the Global Interpreter Lock (GIL). Python 3.14 (PEP 779) makes this a supported configuration. For scientific computing libraries like tensogram — where the hot path is Rust-native encode/decode — this is a significant opportunity:

- **Current state:** File-based APIs (`TensogramFile` methods) already release the GIL via `py.allow_threads()` (15 call sites added in `feature/remote-polish`). However, **all module-level buffer APIs** (`py_encode`, `py_decode`, `py_decode_object`, `py_decode_range`, `py_scan`, `py_validate`, `PyBufferIter.__next__`) still hold the GIL for the entire Rust computation. The module is not declared as free-threading-safe.
- **After this work:** Module declared thread-safe via `#[pymodule(gil_used = false)]`, remaining buffer APIs release the GIL during Rust computation, and benchmarks prove multi-threaded scaling.

Tensogram is in an excellent position because:
1. The Rust core has **no shared mutable state** — encode/decode are pure functions on byte buffers
2. All `#[pyclass]` types own their data (no `Rc`, no `RefCell`, no raw pointers)
3. File-based APIs already release the GIL (15 `allow_threads` call sites)
4. PyO3 0.25 (currently used) already supports free-threaded builds since PyO3 0.23

---

## 2. Current State Analysis

### 2.1 PyO3 Version & Configuration
- **PyO3:** 0.25 with `extension-module` feature
- **numpy crate:** 0.25
- **maturin:** >=1.0,<2.0
- **Python:** >=3.9
- **Module declaration:** `#[pymodule] fn tensogram(...)` — **no** `gil_used` annotation (defaults to GIL-required on PyO3 <0.28)

### 2.2 GIL Release Status — What's Done vs What's Missing

**Already releasing GIL** (from `feature/remote-polish`):
| Call site | Function |
|-----------|----------|
| `TensogramFile::open` | `allow_threads(\|\| TensogramFile::open_source(...))` |
| `TensogramFile::open_remote` | `allow_threads(\|\| TensogramFile::open_remote(...))` |
| `TensogramFile::message_count` | `py.allow_threads(\|\| self.file.message_count())` |
| `TensogramFile::append` | `py.allow_threads(\|\| self.file.append(...))` |
| `TensogramFile::decode_message` | `allow_threads(\|\| self.file.decode_message(...))` |
| `TensogramFile::decode_metadata` | `allow_threads(\|\| self.file.decode_metadata(...))` |
| `TensogramFile::decode_descriptors` | `allow_threads(\|\| self.file.decode_descriptors(...))` |
| `TensogramFile::decode_object` | `allow_threads(\|\| self.file.decode_object(...))` |
| `TensogramFile::read_message` | `allow_threads(\|\| self.file.read_message(...))` |
| `TensogramFile::messages` | `allow_threads(\|\| self.file.messages())` |
| `TensogramFile::__iter__` | `py.allow_threads(\|\| self.file.message_count())` |
| `TensogramFile::__getitem__` | `allow_threads(\|\| self.file.message_count())` |
| `TensogramFile::__len__` | `py.allow_threads(\|\| self.file.message_count())` |
| `PyFileIter::__next__` | `allow_threads(\|\| self.file.decode_message(...))` |
| `TensogramFile::decode_range` | `allow_threads(\|\| self.file.decode_range(...))` |

**Still holding GIL** (the gap):
| Function | Hot path | Benefit of release |
|----------|----------|-------------------|
| `py_encode` | `encode()` | **High** — compression + hashing |
| `py_encode_pre_encoded` | `encode_pre_encoded()` | **Medium** — framing + hashing |
| `py_decode` | `decode()` | **High** — decompression + decoding |
| `py_decode_object` | `decode_object()` | **High** — single object decode |
| `py_decode_range` | `decode_range()` + `decode_descriptors()` | **Medium** — partial decode |
| `py_scan` | `scan()` | **Medium** — byte scanning |
| `py_validate` | `validate_message()` | **High** — full decode + check |
| `py_validate_file` | `core_validate_file()` | **High** — file I/O + decode |
| `PyBufferIter::__next__` | `decode()` | **High** — per-message decode |
| `py_decode_metadata` | `decode_metadata()` | Low — CBOR-only, fast |
| `py_decode_descriptors` | `decode_descriptors()` | Low — CBOR-only, fast |

**Decision:** Skip `py_decode_metadata` and `py_decode_descriptors` — the buffer copy cost would likely exceed the benefit for these lightweight CBOR-only operations.

### 2.3 `#[pyclass]` Types (6 total)

| Type | Fields | Send+Sync? | Notes |
|------|--------|------------|-------|
| `PyDataObjectDescriptor` | `DataObjectDescriptor` (Clone) | Yes | Immutable, all owned data |
| `PyMetadata` | `GlobalMetadata` (Clone) | Yes | Immutable, all owned data |
| `PyTensogramFile` | `TensogramFile` (PathBuf + cached offsets) | Yes* | All owned. `&mut self` methods = PyO3 runtime borrow check, NOT silent serialization. Concurrent calls on **same instance** will raise `RuntimeError` (borrow conflict), which is correct |
| `PyFileIter` | `TensogramFile + index + count` | Yes* | Same analysis — exclusive ownership |
| `PyBufferIter` | `Vec<u8> + offsets + index` | Yes | All owned, no shared state |
| `PyStreamingEncoder` | `Option<StreamingEncoder<Vec<u8>>>` | Needs verify | Verify with compile-time assertion on inner types |

**Important (Oracle correction):** `&mut self` does NOT make a pyclass "not Sync". It means concurrent access to the **same object** triggers PyO3's runtime borrow checker (raises `RuntimeError`). The types ARE `Sync` — concurrent use of **different instances** is perfectly safe. This is the correct behavior.

### 2.4 GIL-Sensitive Code
- **`GILOnceCell<PyObject>`** (line ~522) — caches the `tensogram.Message` namedtuple type
- **numpy API calls** (`PyArray::from_vec`, `frombuffer`, etc.) correctly require GIL — must happen after re-acquiring

### 2.5 Existing Benchmarks
- Rust criterion benchmarks in `benchmarks/` (codec-matrix, grib-comparison, encode_pre_encoded)
- **No Python-level benchmarks exist**
- **No threading/parallelism benchmarks exist**

---

## 3. Implementation Plan

### PR 1: Dependency Migration + Module Declaration

#### 1a. Upgrade PyO3 to 0.28
- `crates/tensogram-python/Cargo.toml`: `pyo3 = { version = "0.28", features = ["extension-module"] }`
- **Upgrade `numpy` crate** to version compatible with PyO3 0.28 (rust-numpy versions track PyO3; `numpy = "0.25"` almost certainly won't work with `pyo3 = "0.28"`)
- Pin `maturin >= 1.8` for free-threaded wheel building support
- Require Python-side `numpy >= 2.1` for free-threaded NumPy support
- Handle API migrations:
  - `Python::with_gil` → `Python::attach` (old name still works but deprecated)
  - `GILOnceCell` → `OnceLock` + `OnceLockExt` (or `PyOnceLock` if available)
  - `IntoPyObject` trait changes between 0.25 and 0.28
  - `allow_threads` → `detach` naming (both work)

#### 1b. Declare Module as Free-Threading Safe
```rust
#[pymodule(gil_used = false)]
fn tensogram(m: &Bound<'_, PyModule>) -> PyResult<()> { ... }
```
This sets `Py_MOD_GIL` to `Py_MOD_GIL_NOT_USED`. On GIL-enabled builds this is a no-op. On free-threaded builds it prevents the interpreter from re-enabling the GIL at runtime.

#### 1c. Send + Sync Compile-Time Assertions
Add assertions for the inner types that matter (not the pyclass wrappers, but the Rust types they contain):
```rust
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<tensogram_core::TensogramFile>();
        assert_send_sync::<tensogram_core::StreamingEncoder<Vec<u8>>>();
        assert_send_sync::<tensogram_core::DataObjectDescriptor>();
        assert_send_sync::<tensogram_core::GlobalMetadata>();
    }
};
```

#### 1d. Migrate GILOnceCell → OnceLock + OnceLockExt
```rust
use std::sync::OnceLock;
use pyo3::sync::OnceLockExt;

static MESSAGE_TYPE: OnceLock<PyObject> = OnceLock::new();

fn pack_message(py: Python<'_>, meta: PyMetadata, objects: PyObject) -> PyResult<PyObject> {
    let msg_type = MESSAGE_TYPE
        .get_or_try_init_py_attached(|| {
            // NOTE: Do NOT nest Python::with_gil/attach inside here
            let py = Python::with_gil(|py| {
                Ok::<_, PyErr>(py.import("tensogram")?.getattr("Message")?.unbind())
            })?;
            Ok(py)
        })?
        .bind(py);
    // ...
}
```
**Pitfall (Oracle warning):** Do not nest `Python::attach` inside the `OnceLockExt` closure — it can deadlock on free-threaded builds.

#### 1e. Update pyproject.toml
```toml
requires-python = ">=3.9"
dependencies = ["numpy>=2.1"]  # free-threaded support requires numpy 2.1+

[build-system]
requires = ["maturin>=1.8,<2.0"]
```

#### Verification:
- `cargo build` passes with new PyO3 version
- `maturin develop` succeeds
- All 200+ existing Python tests pass on Python 3.12/3.13
- Compile-time `Send + Sync` assertions pass
- No clippy warnings, `cargo fmt` clean

---

### PR 2: Thread-Safety Tests + CI

#### 2a. Thread-Safety Tests

Create `tests/python/test_free_threaded.py`:

```python
"""Thread-safety tests for free-threaded Python support."""
import threading
import numpy as np
import tensogram

def test_concurrent_encode_decode():
    """Multiple threads encoding and decoding simultaneously (separate data)."""
    errors = []
    def worker(thread_id):
        try:
            data = np.random.randn(1000).astype(np.float32)
            meta = {"version": 2, "base": [{}]}
            desc = {"type": "ntensor", "shape": [1000], "dtype": "float32",
                    "encoding": "simple_packing", "compression": "zstd"}
            for _ in range(100):
                encoded = tensogram.encode(meta, [(desc, data)])
                result = tensogram.decode(encoded)
                arr = result.objects[0][1]
                np.testing.assert_allclose(arr, data, rtol=1e-5)
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"Thread errors: {errors}"

def test_concurrent_file_decode():
    """Multiple threads decoding from SEPARATE TensogramFile handles."""
    # Each thread opens its own file handle — no shared mutable state

def test_concurrent_scan():
    """Multiple threads scanning the same immutable buffer."""

def test_concurrent_validate():
    """Multiple threads validating different messages simultaneously."""

def test_concurrent_streaming_encoder():
    """Each thread owns its own StreamingEncoder — no sharing."""

def test_concurrent_codec_backends():
    """Smoke test all codec combinations under concurrent load.
    
    Critical: native libraries (blosc2, zfp, sz3, libaec/szip) need
    concurrent coverage since they have their own internal state.
    """
    codecs = [
        ("none", "none"),
        ("simple_packing", "szip"),
        ("simple_packing", "zstd"),
        ("simple_packing", "lz4"),
        ("none", "zstd"),
    ]
    # Run each codec combination concurrently

def test_same_object_concurrent_access():
    """Verify that concurrent &mut self access on same TensogramFile
    raises RuntimeError (PyO3 borrow check), NOT corruption."""
```

#### 2b. CI Matrix Extension

Add to `.github/workflows/ci.yml`:
```yaml
python-free-threaded:
  name: Python 3.13t (free-threaded)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: deadsnakes/action@v3.2.0
      with:
        python-version: "3.13"
        nogil: true
    - uses: astral-sh/setup-uv@v5
    - name: Build and test
      env:
        PYTHON_GIL: "0"  # Explicitly disable GIL
      run: |
        uv venv .venv --python python3.13t
        source .venv/bin/activate
        # Verify GIL is actually disabled
        python -c "import sys; assert not sys._is_gil_enabled(), 'GIL should be disabled'"
        uv pip install maturin "numpy>=2.1" pytest
        cd crates/tensogram-python && maturin develop --release
        cd ../..
        python -m pytest tests/python/ -v
```

#### Verification:
- CI job for Python 3.13t is green
- All threading tests pass on both GIL-enabled and free-threaded builds
- `sys._is_gil_enabled()` returns `False` in free-threaded CI
- No deadlocks in repeated test runs (`pytest --count=50`)

---

### PR 3: Release GIL on Buffer APIs + Input Ownership

#### 3a. Pattern: Extract → Detach → Return

For each buffer API, the pattern is:
1. **Extract** Python objects into owned Rust types (GIL held)
2. **Copy** input buffers to owned `Vec<u8>` (GIL held)
3. **Detach** — run Rust computation via `py.allow_threads()` (GIL released)
4. **Return** — convert results to Python objects (GIL re-acquired)

#### 3b. `py_encode` — Data Already Owned
```rust
fn py_encode<'py>(py: Python<'py>, ...) -> PyResult<Bound<'py, PyBytes>> {
    let global_meta = dict_to_global_metadata(global_meta_dict)?;
    let pairs = extract_descriptor_data_pairs(py, descriptors_and_data)?;
    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();
    let options = make_encode_options(hash)?;
    
    // GIL released — pairs already own their data
    let msg = py.allow_threads(|| encode(&global_meta, &refs, &options).map_err(to_py_err))?;
    
    Ok(PyBytes::new(py, &msg))
}
```

#### 3c. `py_decode` — Buffer Copy Required
```rust
fn py_decode(py: Python<'_>, buf: &[u8], ...) -> PyResult<PyObject> {
    let owned_buf = buf.to_vec();  // copy once — buf reference needs GIL
    let options = DecodeOptions { verify_hash, native_byte_order, ..Default::default() };
    
    // GIL released — owned_buf is Send
    let (global_meta, data_objects) = py.allow_threads(|| {
        decode(&owned_buf, &options).map_err(to_py_err)
    })?;
    
    // GIL re-acquired — numpy conversion
    let result_list = data_objects_to_python(py, &data_objects)?;
    pack_message(py, PyMetadata { inner: global_meta }, result_list)
}
```

#### 3d. Full List of Functions to Modify

| Function | Copy needed? | Notes |
|----------|-------------|-------|
| `py_encode` | No | `extract_descriptor_data_pairs` returns owned data |
| `py_encode_pre_encoded` | No | `extract_pre_encoded_pairs` returns owned data |
| `py_decode` | Yes — `buf.to_vec()` | |
| `py_decode_object` | Yes — `buf.to_vec()` | |
| `py_decode_range` | Yes — `buf.to_vec()` | |
| `py_scan` | Yes — `buf.to_vec()` | |
| `py_validate` | Yes — `buf.to_vec()` | |
| `py_validate_file` | No | Path string, not buffer |
| `PyBufferIter::__next__` | No | Owns `self.buf: Vec<u8>` already |
| `StreamingEncoder::write_object` | No | Data extracted to owned Vec |

**Skipped** (low value, copy cost > benefit):
- `py_decode_metadata` — CBOR-only, microseconds
- `py_decode_descriptors` — CBOR-only, microseconds

#### 3e. Memory Consideration (Oracle Warning)

Buffer copy (`buf.to_vec()`) doubles RSS for the duration of the decode. For typical scientific messages (1-100 MB), this is acceptable. For very large messages (>500 MB) with many threads, this could become a problem. 

**Future optimisation:** Use `PyBuffer<u8>` for zero-copy GIL release (requires `unsafe` audit). Document this as a known trade-off.

#### Verification:
- All existing tests pass (no regressions)
- Threading tests from PR 2 now show actual parallelism
- Single-thread performance regression < 5% (from buffer copy overhead)

---

### PR 4: Benchmarks + Documentation

#### 4a. Benchmark Framework

Create `benchmarks/python/` with two key benchmarks measuring **two separate comparisons**:

**Comparison 1:** Regular CPython before vs after `allow_threads` (shows GIL-release benefit even on GIL-enabled Python, since other Python threads can run)

**Comparison 2:** Regular CPython (after detach) vs Free-Threaded 3.13t (after detach) (shows the additional benefit of removing the GIL entirely)

```
benchmarks/python/
    bench_threading.py     # Main multi-threaded scaling benchmark
    README.md              # How to run, interpret results
```

#### 4b. Benchmark Design

```python
"""Multi-threaded encode/decode scaling benchmark.

Measures two things:
1. GIL-release benefit: single-thread vs multi-thread throughput
2. Free-threaded benefit: GIL-enabled vs 3.13t scaling curves

Run on both regular CPython and free-threaded CPython to compare.
"""
import sys, os, time, threading, platform
import numpy as np
import tensogram

def detect_environment():
    """Report Python version, free-threading status, CPU cores."""
    is_free_threaded = hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled()
    return {
        "python": sys.version,
        "free_threaded": is_free_threaded,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "tensogram_version": tensogram.__version__,
    }

def prepare_workload(size, encoding="simple_packing", compression="zstd"):
    """Pre-generate all inputs (no allocation during measurement)."""
    data = np.random.randn(size).astype(np.float32)
    meta = {"version": 2, "base": [{}]}
    desc = {"type": "ntensor", "shape": [size], "dtype": "float32",
            "encoding": encoding, "compression": compression}
    msg = tensogram.encode(meta, [(desc, data)])
    return meta, desc, data, msg

def bench_scaling(operation, args, thread_counts, iterations):
    """Measure throughput scaling across thread counts.
    
    Uses barrier for synchronized start; reports wall time and per-thread time.
    Pin native-library thread counts to 1 to avoid confounding.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"
    
    results = {}
    for n_threads in thread_counts:
        barrier = threading.Barrier(n_threads + 1)
        thread_times = [0.0] * n_threads
        
        def worker(tid):
            barrier.wait()
            t0 = time.perf_counter_ns()
            for _ in range(iterations):
                operation(*args)
            thread_times[tid] = time.perf_counter_ns() - t0
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads: t.start()
        barrier.wait()
        wall_t0 = time.perf_counter_ns()
        for t in threads: t.join()
        wall_ns = time.perf_counter_ns() - wall_t0
        
        total_ops = n_threads * iterations
        results[n_threads] = {
            "wall_ms": wall_ns / 1e6,
            "throughput": total_ops / (wall_ns / 1e9),
            "per_thread_ms": [t / 1e6 / iterations for t in thread_times],
        }
    return results
```

#### 4c. Benchmark Matrix

| Workload | Size | Encoding | Compression | Purpose |
|----------|------|----------|-------------|---------|
| Small | 10K f32 | none | none | Overhead measurement (copy cost visible here) |
| Medium | 1M f32 | simple_packing | zstd | Typical scientific data |
| Large | 16M f64 | simple_packing | szip | Heavy compression |
| Encode | 1M f32 | simple_packing | zstd | Parallel encode scaling |
| Validate | 1M f32 | simple_packing | zstd | Validation scaling |

**Excluded from primary GIL benchmark:** File I/O (confounded by disk latency).
**Important:** Pin native library thread counts (`OMP_NUM_THREADS=1`, `BLOSC_NTHREADS=1`) to isolate GIL effect.

#### 4d. Expected Results (Corrected per Oracle)

**On regular CPython (GIL-enabled), after `allow_threads`:**
```
  1 thread:  450 ops/s (baseline)
  2 threads: ~800 ops/s (1.78x) ← GIL released during Rust, other threads can run
  4 threads: ~1500 ops/s (3.3x) ← limited by GIL re-acquisition overhead
  8 threads: ~2000 ops/s (4.4x) ← diminishing returns due to GIL contention on numpy
```

**On free-threaded CPython 3.13t, after `allow_threads`:**
```
  1 thread:  430 ops/s (slightly lower due to 3.13t overhead)
  2 threads: ~850 ops/s (1.98x) ← no GIL contention at all
  4 threads: ~1700 ops/s (3.95x) ← near-linear
  8 threads: ~3200 ops/s (7.4x) ← near-linear scaling
```

**Key insight (Oracle correction):** `allow_threads` ALREADY gives multi-thread benefit on regular CPython — threads alternate between "holding GIL for numpy" and "GIL-free during Rust". Free-threaded Python removes the remaining GIL contention on numpy array construction.

#### 4e. Output Format

```
============================================================
  Tensogram Python Threading Benchmark
  Python: 3.14.0 (free-threaded: yes)
  tensogram: 0.10.0 | NumPy: 2.2.0
  Platform: Linux x86_64 | CPU: 8 cores
  Date: 2025-XX-XX
============================================================

--- Single-Thread Baseline ---
  Encode 1M f32 (packing+zstd): 2.15 ms (min) / 2.22 ms (median)
  Decode 1M f32 (packing+zstd): 1.82 ms (min) / 1.89 ms (median)

--- Decode Scaling (1M f32, packing+zstd, 50 iters/thread) ---
  Threads | Wall Time  | Throughput  | Speedup | Per-Thread Avg
  --------|------------|-------------|---------|---------------
  1       | 91.2 ms    | 548 ops/s   | 1.00x   | 1.82 ms
  2       | 46.5 ms    | 1075 ops/s  | 1.96x   | 1.86 ms
  4       | 23.8 ms    | 2100 ops/s  | 3.83x   | 1.90 ms
  8       | 12.8 ms    | 3906 ops/s  | 7.12x   | 1.92 ms

--- Encode Scaling (1M f32, packing+zstd, 50 iters/thread) ---
  (same format)
```

#### 4f. Documentation

**`docs/src/guide/free-threaded-python.md`:**
- What free-threaded Python is and why it matters for tensogram
- How to install Python 3.13t/3.14t
- How to build tensogram for free-threaded Python
- Usage examples with `threading.Thread`
- Performance expectations (link to benchmark results)
- Known limitations (numpy dtype=object, shared mutable ndarray access)

**`docs/src/guide/python-benchmark-results.md`:**
- Machine specs, Python version, tensogram version
- Scaling charts for both GIL-enabled and free-threaded
- Interpretation guide

**Update existing docs:**
- `docs/src/guide/python-api.md` — threading safety notes
- `ARCHITECTURE.md` — mention free-threaded support
- `CHANGELOG.md` — release entry

#### Verification:
- Benchmarks run without errors on both GIL-enabled and free-threaded Python
- Results show expected scaling pattern
- Single-thread performance regression < 5%
- Documentation builds (`mdbook build`)

---

## 4. Dependency Changes Summary

| Dependency | Current | Target | Reason |
|------------|---------|--------|--------|
| `pyo3` | 0.25 | 0.28 | Free-threaded defaults, API improvements |
| `numpy` (Rust crate) | 0.25 | Latest compatible with PyO3 0.28 | rust-numpy tracks PyO3 versions |
| `numpy` (Python) | any | >=2.1 | Free-threaded NumPy support |
| `pyo3-build-config` | (none) | 0.28 | Only if `#[cfg(Py_GIL_DISABLED)]` branches are needed |
| `maturin` | >=1.0,<2.0 | >=1.8,<2.0 | Free-threaded wheel building |

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `numpy` crate incompatible with PyO3 0.28 | **Medium** | High | Check compatibility matrix first; pin known-good |
| PyO3 0.28 API migration breaks code | Low | Medium | Incremental migration, test suite green at each step |
| `StreamingEncoder<Vec<u8>>` not Send+Sync | Low | Medium | Compile-time assertion catches early |
| Buffer copy doubles RSS for large messages | Medium | Medium | Acceptable for <100MB; document; future zero-copy path |
| Native codec libraries (blosc2, zfp, sz3) not thread-safe | **Medium** | **High** | Concurrent codec smoke tests in PR 2 |
| Free-threaded NumPy limitations | Medium | Medium | Require numpy>=2.1; avoid shared mutable arrays; document |
| Same-instance concurrent access raises RuntimeError | Low | Low | This is correct behavior (PyO3 borrow check); document it |
| `OnceLockExt` closure deadlocks | Low | High | Don't nest `Python::attach` inside closure |
| Benchmark variance masks results | Medium | Low | Report min/median/p95; pin thread counts; warm up |

---

## 6. Out of Scope

- **xarray/zarr backend thread safety** — separate concern (dask, zarr locking)
- **Multi-threaded Rust encoding pipeline** — separate TODO (`multi-threaded-coding-pipeline`)
- **Python multiprocessing** — already works (separate GILs per process)
- **PyPy support** — free-threaded is CPython-only
- **Zero-copy buffer detach** — future optimisation for very large messages
- **`build.rs` + `#[cfg(Py_GIL_DISABLED)]`** — only add if actual code divergence is needed

---

## 7. Execution Order (Single PR)

All work ships in **one PR** on `feature/free-threaded-python`. The scope is well-contained:
dependency bump, ~9 function edits, one new test file, one benchmark script, CI job, docs.

**Execution order within the branch:**

```
Step 1: Dependency Migration
  ├── PyO3 0.25 → 0.28
  ├── numpy crate upgrade (track PyO3 0.28)
  ├── maturin >=1.8
  ├── pyproject.toml: numpy>=2.1
  └── cargo build + maturin develop — verify compilation

Step 2: Module Declaration + Safety
  ├── #[pymodule(gil_used = false)]
  ├── Send + Sync compile-time assertions
  ├── GILOnceCell → OnceLock + OnceLockExt
  └── Existing tests pass

Step 3: GIL Release on Buffer APIs
  ├── py.allow_threads() on 9 functions
  ├── buf.to_vec() for decode/scan/validate inputs
  └── PyBufferIter::__next__ detach

Step 4: Tests + CI
  ├── tests/python/test_free_threaded.py (8 scenarios)
  ├── Concurrent codec smoke tests
  ├── CI job for Python 3.13t (PYTHON_GIL=0)
  └── All tests green on GIL-enabled + free-threaded

Step 5: Benchmarks + Documentation
  ├── benchmarks/python/bench_threading.py
  ├── docs/src/guide/free-threaded-python.md
  ├── Updated ARCHITECTURE.md, python-api.md, CHANGELOG.md
  └── mdbook build passes
```

---

## 8. Definition of Done

- [ ] PyO3 0.28 + compatible numpy crate + maturin >= 1.8
- [ ] `#[pymodule(gil_used = false)]` declared
- [ ] All existing Python tests (200+) pass on Python 3.12 AND free-threaded 3.13t
- [ ] Thread-safety tests (8+ scenarios) pass with 8 concurrent threads, 100 iterations
- [ ] Concurrent codec smoke tests cover all compression backends
- [ ] CI job for Python 3.13t with `PYTHON_GIL=0` is green
- [ ] `py.allow_threads()` on 9 buffer API functions
- [ ] No single-thread performance regression (< 5% variance)
- [ ] Multi-thread decode speedup >= 3x with 4 threads on free-threaded Python
- [ ] Benchmark suite produces formatted report
- [ ] Documentation covers usage, performance, limitations
- [ ] No clippy warnings, cargo fmt clean, ruff clean

---

## 9. Oracle Review Summary

The following corrections were incorporated from Oracle deep review:

1. **File APIs already release GIL** — The `feature/remote-polish` branch added 15 `allow_threads` call sites on `TensogramFile` methods. The actual gap is only in module-level buffer APIs.

2. **`&mut self` ≠ non-Sync** — `&mut self` methods use PyO3's runtime borrow checker, not silent serialization. Concurrent calls on the same instance raise `RuntimeError`, which is correct behavior.

3. **Benchmark expectations corrected** — `allow_threads` already provides multi-thread benefit on regular CPython (threads alternate GIL). Free-threaded Python removes remaining contention. The story is NOT "flat until 3.13t" — it's "good scaling with GIL-release, better scaling without GIL".

4. **numpy crate version** — `numpy = "0.25"` almost certainly won't work with `pyo3 = "0.28"`. rust-numpy versions track PyO3. Python-side numpy >= 2.1 required for free-threaded support.

5. **Native codec thread safety** — blosc2, zfp, sz3, libaec/szip need concurrent smoke testing since they have their own internal state.

6. **PR reordering** — Tests + CI (PR 2) before hot-path changes (PR 3) so regressions are detectable.

7. **OnceLockExt pitfall** — Don't nest `Python::attach` inside the init closure to avoid deadlocks.

8. **Buffer copy trade-off** — `buf.to_vec()` is safe but doubles RSS. Document as known limitation; zero-copy path is future work.

# Plan: Python Async Bindings

**TODO.md item**: Python asyncio support
**Status**: Implemented (branch `feature/python-async-bindings`)
**Target**: v0.11.0

> **Note**: This plan was written for the initial TokioMutex-based architecture.
> The implementation evolved to use `Arc<TensogramFile>` with `&self` async methods
> (no mutex). Code snippets below may reference the old design. See the actual
> source code for the current implementation.

## 1. Goal

Expose the native async Rust path (`TensogramFile::*_async` methods) to Python as
first-class `asyncio` coroutines.  The async API enables non-blocking I/O in
asyncio-based applications and composable concurrency via `asyncio.gather`.

```python
import asyncio
import tensogram

async def main():
    f = await tensogram.AsyncTensogramFile.open("s3://bucket/data.tgm")

    # Multiple awaitable decode calls composed via gather.
    # Same-handle calls are safe but serialised by an internal mutex;
    # for true I/O overlap, use separate handles.
    results = await asyncio.gather(
        f.file_decode_object(0, 0),
        f.file_decode_object(0, 1),
        f.file_decode_object(0, 2),
    )
    for r in results:
        print(r["data"].shape)

asyncio.run(main())
```

### Concurrency semantics

`AsyncTensogramFile` wraps a single `TensogramFile` behind `Arc<tokio::sync::Mutex>`.
This means:

- **Same-handle calls are safe**: multiple coroutines may reference the same handle.
- **Same-handle calls are serialised**: the mutex is held for the duration of each I/O
  operation, so `asyncio.gather` on one handle provides composability (other asyncio tasks
  can run between operations) but not intra-handle I/O parallelism.
- **Multi-handle calls overlap**: opening the same file/URL from multiple
  `AsyncTensogramFile` instances provides true concurrent I/O.

True same-handle parallelism would require refactoring core `TensogramFile` mutability
(the async methods take `&mut self` due to lazy `OnceLock` initialisation).
That is out of scope for this PR; the Foundation delivers correct asyncio integration.

## 2. Architecture Decision

### Approach: `pyo3-async-runtimes` + `tokio` bridge

**Chosen**: `pyo3_async_runtimes::tokio::future_into_py` wrapping native Rust async.

**Rationale**:
- `pyo3-async-runtimes` 0.28.0 is released and compatible with pyo3 0.28 (tensogram's version).
- Properly bridges tokio futures to Python `asyncio` awaitables with correct waker propagation.
- Established pattern (e.g. nautilus_trader uses the same approach).
- No manual channel plumbing or `experimental-async` feature needed.

**Rejected alternatives**:
- pyo3 `experimental-async` + manual `futures::channel::oneshot` bridging:
  Correct but more boilerplate and requires manual waker management.
- Blocking async via `block_on` inside `py.detach()`:
  Already done by the sync API; does not enable `asyncio.gather` composition.

### Internal structure

```rust
#[cfg(feature = "async")]
#[pyclass(name = "AsyncTensogramFile")]
struct PyAsyncTensogramFile {
    file: Arc<tokio::sync::Mutex<TensogramFile>>,
    cached_source: String,     // captured at construction, immutable
    cached_is_remote: bool,    // captured at construction, immutable
}
```

- `Arc` for shared ownership across concurrent async calls.
- `tokio::sync::Mutex` for async-aware locking (`.lock().await` inside tokio context).
- `cached_source`/`cached_is_remote`: plain fields captured at construction, avoiding
  any mutex access for read-only queries.  These values never change after `open`.
- Separate class from `PyTensogramFile` to avoid perturbing the existing sync API.

### Return pattern

Each async method follows this flow:
1. Clone `Arc`, capture params (GIL held, cheap).
2. Validate arguments eagerly (errors raised at call time, not when awaited).
3. `future_into_py(py, async move { ... })` spawns on tokio.
4. Inside future: acquire mutex, call `TensogramFile::*_async`, **release mutex**.
5. `Python::attach(|py| ...)` to convert Rust data to Python objects (`Py<PyAny>`).
6. Return `Py<PyAny>` (Send + 'static, GIL-independent).

**Lock ordering**: the mutex is always released **before** `Python::attach()`.
This prevents GIL/mutex deadlocks (same discipline as the sync path where `py.detach()`
releases GIL before Rust work, here we release mutex before GIL re-acquisition).

**Argument validation**: `open_remote` validates `storage_options` dict conversion
**before** creating the coroutine.  Bad `storage_options` raises `ValueError` immediately
at call time, not when awaited.  Same for any other argument parsing.

### Tokio runtime

`pyo3-async-runtimes` manages its own tokio runtime
(`pyo3_async_runtimes::tokio::get_runtime()`).  The existing shared `OnceLock<Runtime>`
in `remote.rs` continues to serve `block_on_shared()` calls from the sync API.
The two runtimes are independent.

For **local backends**: `TensogramFile::*_async` internally uses
`tokio::task::spawn_blocking` for file I/O, which works on either runtime.

For **remote backends**: `RemoteBackend` async methods run natively on whatever
tokio runtime is active.  The shared runtime serves sync `block_on_shared()` calls.

**Runtime init failure**: if `pyo3-async-runtimes` fails to create its tokio runtime,
`future_into_py` raises `RuntimeError`.  This is analogous to the sync path where
`shared_runtime()` returns `Err(TensogramError::Remote(...))`.

## 3. Dependencies

Add to `crates/tensogram-python/Cargo.toml`:

```toml
[features]
default = ["async"]
async = ["tensogram-core/async", "dep:pyo3-async-runtimes", "dep:tokio"]

[dependencies]
pyo3-async-runtimes = { version = "0.28", features = ["tokio-runtime"], optional = true }
tokio = { version = "1", features = ["sync"], optional = true }
```

This adds:
- `pyo3-async-runtimes` 0.28 (bridges tokio futures to Python asyncio)
- `tokio` 1.x with `sync` feature (direct dependency for `tokio::sync::Mutex`)

Both gated behind the `async` feature.  All async Rust code behind `#[cfg(feature = "async")]`
including imports:

```rust
#[cfg(feature = "async")]
use std::sync::Arc;
#[cfg(feature = "async")]
use tokio::sync::Mutex as TokioMutex;
```

Feature gate: `async` is default-on.  The sync-only build path remains available via
`--no-default-features`.  Adding `default = ["async"]` changes the default build of
`tensogram-python`; this is intentional and must be verified through `maturin develop`
(the real build tool), not just `cargo build`.

Note: `tensogram-python` already hard-enables the `remote` feature on `tensogram-core`
(line 11 of `Cargo.toml`: `features = ["remote"]`).  The `async` feature adds
`tensogram-core/async` on top.  Therefore `open_source_async` (which requires
`remote + async`) is always available when `async` is enabled.
`--no-default-features` disables the Python async bindings but NOT remote support.

## 4. Implementation Steps

### Step 1: Cargo.toml & feature gate

**Files**: `crates/tensogram-python/Cargo.toml`

- Add `[features]` section with `default = ["async"]` and `async` feature
- Add `pyo3-async-runtimes` and `tokio` dependencies (both optional, gated on `async`)
- Verify `maturin develop` succeeds (default features, i.e. with async)
- Verify `maturin develop --no-default-features` succeeds (sync-only)
- Verify `cargo clippy -p tensogram-python` passes
- Verify `cargo clippy -p tensogram-python --no-default-features` passes (no unused imports)

### Step 2: Shared test infrastructure

**Files**: `tests/python/conftest.py` (new), `tests/python/test_remote.py` (update)

Extract **only pytest fixtures** to `tests/python/conftest.py` (pytest auto-discovers
fixtures from `conftest.py`, but **not** plain functions):

```python
# tests/python/conftest.py
import http.server
import threading
import numpy as np
import pytest
import tensogram


def _make_handler(file_data: bytes):
    """HTTP handler with Range support (moved from test_remote.py)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass
        def do_HEAD(self):
            ...  # same as current test_remote.py
        def do_GET(self):
            ...  # same as current test_remote.py
    return Handler


@pytest.fixture
def serve_tgm_bytes():
    """Start mock HTTP server for given bytes, isolated per call."""
    entries = []
    def _serve(data: bytes) -> str:
        handler = _make_handler(data)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        entries.append((server, thread))
        return f"http://127.0.0.1:{port}/test.tgm"
    yield _serve
    for server, thread in entries:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def tgm_path(tmp_path):
    """Create a local .tgm file with 3 messages for async testing.

    Message i contains a float32 [10] array filled with float(i).
    """
    path = str(tmp_path / "test.tgm")
    with tensogram.TensogramFile.create(path) as f:
        for i in range(3):
            meta = {"version": 2, "base": [{"index": i}]}
            desc = {
                "type": "ntensor", "shape": [10], "dtype": "float32",
                "byte_order": "little", "encoding": "none",
                "filter": "none", "compression": "none",
            }
            data = np.full(10, float(i), dtype=np.float32)
            f.append(meta, [(desc, data)])
    return path
```

Keep `make_global_meta`, `make_descriptor`, `encode_test_message` as module-local
helper functions in each test file that needs them.  They are not fixtures and should
not go in `conftest.py`.

Update `test_remote.py`: remove the `serve_tgm_bytes` fixture definition (now in
`conftest.py`).  Run `python -m pytest tests/python/test_remote.py -v` in isolation
to verify no regressions.

### Step 3: Minimal failing test (TDD anchor)

**Files**: `tests/python/test_async.py` (new)

Write a minimal async test that will fail until Step 5 is complete:

```python
import pytest
import tensogram

@pytest.mark.asyncio
async def test_async_open_exists(tgm_path):
    """TDD anchor: AsyncTensogramFile.open must exist and return a handle."""
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    assert f.source() == tgm_path

@pytest.mark.asyncio
async def test_async_decode_message_exists(tgm_path):
    """TDD anchor: decode_message must return Message namedtuple."""
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    meta, objects = await f.decode_message(0)
    assert meta.version == 2
```

### Step 4: `PyAsyncTensogramFile` class + static constructors

**Files**: `crates/tensogram-python/src/lib.rs`

Keep inline in `lib.rs`.  The file is ~1886 lines; async additions add ~200-250 lines.

New `#[pyclass]`:

```rust
#[cfg(feature = "async")]
#[pyclass(name = "AsyncTensogramFile")]
struct PyAsyncTensogramFile {
    file: Arc<TokioMutex<TensogramFile>>,
    cached_source: String,
    cached_is_remote: bool,
}
```

Two static constructors returning Python coroutines:

```python
f = await AsyncTensogramFile.open("file.tgm")                     # auto-detect
f = await AsyncTensogramFile.open_remote(url, {"key": "value"})   # explicit remote
```

Implementation pattern (`open`):
```rust
#[cfg(feature = "async")]
#[pymethods]
impl PyAsyncTensogramFile {
    #[staticmethod]
    fn open<'py>(py: Python<'py>, source: &str) -> PyResult<Bound<'py, PyAny>> {
        let source = source.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let file = TensogramFile::open_source_async(&source).await.map_err(to_py_err)?;
            let is_remote = file.is_remote();
            let source_str = file.source();
            let wrapped = PyAsyncTensogramFile {
                file: Arc::new(TokioMutex::new(file)),
                cached_source: source_str,
                cached_is_remote: is_remote,
            };
            Python::attach(|py| Ok(wrapped.into_pyobject(py)?.into_any().unbind()))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (source, storage_options=None))]
    fn open_remote<'py>(
        py: Python<'py>,
        source: &str,
        storage_options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Validate storage_options EAGERLY (error at call time, not when awaited).
        let opts = match storage_options {
            Some(dict) => {
                let mut map = std::collections::BTreeMap::new();
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    let val: String = v.extract::<String>().or_else(|_| {
                        v.str()
                            .map(|s| s.to_string())
                            .map_err(|_| PyValueError::new_err(format!(
                                "storage_options value for key '{key}' must be convertible to string"
                            )))
                    })?;
                    map.insert(key, val);
                }
                map
            }
            None => std::collections::BTreeMap::new(),
        };
        let source = source.to_string();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let file = TensogramFile::open_remote_async(&source, &opts).await.map_err(to_py_err)?;
            let is_remote = file.is_remote();
            let source_str = file.source();
            let wrapped = PyAsyncTensogramFile {
                file: Arc::new(TokioMutex::new(file)),
                cached_source: source_str,
                cached_is_remote: is_remote,
            };
            Python::attach(|py| Ok(wrapped.into_pyobject(py)?.into_any().unbind()))
        })
    }
}
```

### Step 5: Async decode methods

Method names mirror the existing sync `PyTensogramFile` API exactly:

| Async Python method | Sync Python equivalent | Rust async method | Returns |
|---|---|---|---|
| `decode_message(index, ...)` | `decode_message` | `decode_message_async` | `Message` namedtuple |
| `file_decode_metadata(msg_index)` | `file_decode_metadata` | `decode_metadata_async` | `Metadata` |
| `file_decode_descriptors(msg_index)` | `file_decode_descriptors` | `decode_descriptors_async` | `dict(metadata=..., descriptors=[...])` |
| `file_decode_object(msg_index, obj_index, ...)` | `file_decode_object` | `decode_object_async` | `dict(metadata=..., descriptor=..., data=ndarray)` |

All follow the same pattern (example: `file_decode_object`):

```rust
#[pyo3(signature = (msg_index, obj_index, verify_hash=false, native_byte_order=true))]
fn file_decode_object<'py>(
    &self,
    py: Python<'py>,
    msg_index: usize,
    obj_index: usize,
    verify_hash: bool,
    native_byte_order: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let file = Arc::clone(&self.file);
    let options = DecodeOptions { verify_hash, native_byte_order, ..Default::default() };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        // Async I/O — mutex held, no GIL.
        let (meta, desc, data) = {
            let mut f = file.lock().await;
            f.decode_object_async(msg_index, obj_index, &options).await.map_err(to_py_err)?
        };
        // Mutex released.  Acquire GIL, build Python objects.
        Python::attach(|py| {
            let arr = bytes_to_numpy(py, &desc, &data)?;
            let py_desc = PyDataObjectDescriptor { inner: desc }
                .into_pyobject(py)?.into_any().unbind();
            let result = PyDict::new(py);
            result.set_item("metadata", PyMetadata { inner: meta }.into_pyobject(py)?)?;
            result.set_item("descriptor", py_desc)?;
            result.set_item("data", arr)?;
            Ok(result.into_any().unbind())
        })
    })
}
```

`decode_message` additionally calls `pack_message()` to return the `Message` namedtuple,
matching the sync `decode_message` return type.

### Step 6: Async `read_message` + sync utility methods

```python
msg_bytes: bytes = await f.read_message(index)   # async, wraps read_message_async
is_remote: bool  = f.is_remote()                 # sync, reads cached field
source: str      = f.source()                    # sync, reads cached field
```

```rust
fn read_message<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
    let file = Arc::clone(&self.file);
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let bytes = {
            let mut f = file.lock().await;
            f.read_message_async(index).await.map_err(to_py_err)?
        };
        Python::attach(|py| Ok(PyBytes::new(py, &bytes).into_any().unbind()))
    })
}

fn is_remote(&self) -> bool {
    self.cached_is_remote
}

fn source(&self) -> &str {
    &self.cached_source
}
```

**`message_count()` is deferred** from Foundation.  Reason: the only implementation would
use `tokio::sync::Mutex::blocking_lock()`, which **panics** if called from within a tokio
runtime context.  This is a hard panic-safety violation.  `message_count` will be added
when core exposes `TensogramFile::message_count_async`, or users can open a sync
`TensogramFile` handle to query the count.

### Step 7: `__repr__` and module registration

```rust
fn __repr__(&self) -> String {
    format!("AsyncTensogramFile(source='{}')", self.cached_source)
}
```

Module registration (add to `#[pymodule]` in `lib.rs`):
```rust
#[cfg(feature = "async")]
m.add_class::<PyAsyncTensogramFile>()?;
```

No changes needed to `__init__.py`: `from .tensogram import *` already re-exports all
classes added via `m.add_class`.

### Step 8: Full test suite

**File**: `tests/python/test_async.py`

Add `pytest-asyncio` to test dependencies.  Use explicit `@pytest.mark.asyncio` markers.

```python
# tests/python/test_async.py
import asyncio

import numpy as np
import pytest

import tensogram

# ── Local helpers (not fixtures, not in conftest.py) ──

def _encode_test_message(shape, fill=42.0):
    meta = {"version": 2, "base": [{}]}
    desc = {
        "type": "ntensor", "shape": shape, "dtype": "float32",
        "byte_order": "little", "encoding": "none",
        "filter": "none", "compression": "none",
    }
    data = np.full(shape, fill, dtype=np.float32)
    return tensogram.encode(meta, [(desc, data)])

# ── Open tests ──

@pytest.mark.asyncio
async def test_open_local(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    assert not f.is_remote()
    assert f.source() == tgm_path

@pytest.mark.asyncio
async def test_open_remote(serve_tgm_bytes):
    msg = _encode_test_message([4])
    url = serve_tgm_bytes(msg)
    f = await tensogram.AsyncTensogramFile.open(url)
    assert f.is_remote()

# ── Decode tests ──

@pytest.mark.asyncio
async def test_file_decode_object(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    result = await f.file_decode_object(0, 0)
    assert "metadata" in result
    assert "descriptor" in result
    assert "data" in result
    np.testing.assert_allclose(result["data"], np.zeros(10, dtype=np.float32))

@pytest.mark.asyncio
async def test_decode_message(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    meta, objects = await f.decode_message(1)
    assert meta.version == 2
    np.testing.assert_allclose(objects[0][1], np.ones(10, dtype=np.float32))

@pytest.mark.asyncio
async def test_file_decode_metadata(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    meta = await f.file_decode_metadata(0)
    assert meta.version == 2

@pytest.mark.asyncio
async def test_file_decode_descriptors(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    result = await f.file_decode_descriptors(0)
    assert len(result["descriptors"]) == 1

# ── Gather tests (composability, not parallel I/O) ──

@pytest.mark.asyncio
async def test_gather_safe_on_same_handle(tgm_path):
    """asyncio.gather on one handle is safe (serialised internally)."""
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    results = await asyncio.gather(
        f.file_decode_object(0, 0),
        f.file_decode_object(1, 0),
        f.file_decode_object(2, 0),
    )
    for i, r in enumerate(results):
        np.testing.assert_allclose(r["data"], np.full(10, float(i), dtype=np.float32))

@pytest.mark.asyncio
async def test_gather_mixed_operations(tgm_path):
    """Concurrent metadata + object decodes on same handle."""
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    meta, obj = await asyncio.gather(
        f.file_decode_metadata(0),
        f.file_decode_object(1, 0),
    )
    assert meta.version == 2
    assert obj["data"].shape == (10,)

# ── Parity tests ──

@pytest.mark.asyncio
async def test_async_matches_sync(tgm_path):
    """Async decode produces identical results to sync decode."""
    sync_file = tensogram.TensogramFile.open(tgm_path)
    sync_meta, sync_objects = sync_file.decode_message(0)

    async_file = await tensogram.AsyncTensogramFile.open(tgm_path)
    async_meta, async_objects = await async_file.decode_message(0)

    assert sync_meta.version == async_meta.version
    np.testing.assert_array_equal(sync_objects[0][1], async_objects[0][1])

# ── Error tests ──

@pytest.mark.asyncio
async def test_open_nonexistent():
    with pytest.raises(OSError):
        await tensogram.AsyncTensogramFile.open("/nonexistent.tgm")

@pytest.mark.asyncio
async def test_decode_out_of_range(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    with pytest.raises((ValueError, RuntimeError)):
        await f.file_decode_object(999, 0)

@pytest.mark.asyncio
async def test_open_remote_bad_storage_options(serve_tgm_bytes):
    """Bad storage_options raises ValueError at call time, not when awaited."""
    msg = _encode_test_message([4])
    url = serve_tgm_bytes(msg)

    class Unconvertible:
        def __str__(self):
            raise RuntimeError("cannot convert")

    with pytest.raises(ValueError, match="convertible to string"):
        # Error must be raised HERE, not on await
        tensogram.AsyncTensogramFile.open_remote(url, {"key": Unconvertible()})

# ── Cancellation test ──

@pytest.mark.asyncio
async def test_cancel_then_reuse_handle(tgm_path):
    """Cancelling an in-flight decode does not wedge the mutex."""
    f = await tensogram.AsyncTensogramFile.open(tgm_path)

    task = asyncio.create_task(f.file_decode_object(0, 0))
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Handle must still be usable after cancellation.
    result = await f.file_decode_object(1, 0)
    np.testing.assert_allclose(result["data"], np.ones(10, dtype=np.float32))

# ── Remote async tests ──

@pytest.mark.asyncio
async def test_remote_async_decode(serve_tgm_bytes):
    msg = _encode_test_message([4], fill=42.0)
    url = serve_tgm_bytes(msg)
    f = await tensogram.AsyncTensogramFile.open(url)
    assert f.is_remote()
    result = await f.file_decode_object(0, 0)
    np.testing.assert_allclose(result["data"], np.full(4, 42.0, dtype=np.float32))

@pytest.mark.asyncio
async def test_remote_async_matches_sync(serve_tgm_bytes):
    """Remote async decode matches remote sync decode."""
    msg = _encode_test_message([10], fill=3.14)
    url = serve_tgm_bytes(msg)

    sync_file = tensogram.TensogramFile.open(url)
    sync_result = sync_file.file_decode_object(0, 0)

    async_file = await tensogram.AsyncTensogramFile.open(url)
    async_result = await async_file.file_decode_object(0, 0)

    np.testing.assert_array_equal(sync_result["data"], async_result["data"])

# ── Repr test ──

@pytest.mark.asyncio
async def test_repr(tgm_path):
    f = await tensogram.AsyncTensogramFile.open(tgm_path)
    r = repr(f)
    assert "AsyncTensogramFile" in r
    assert tgm_path in r
```

### Step 9: Example

**File**: `examples/python/15_async_operations.py`

Self-contained example showing:
1. Async open (local file, created inline)
2. Sequential async decode
3. Composing multiple decodes with `asyncio.gather` (with comment explaining
   same-handle serialisation vs event-loop composability)
4. Mixing async decode with other asyncio tasks (e.g. `asyncio.sleep`)

### Step 10: Documentation

Update the following files:

- **`docs/src/guide/python-api.md`**: Add "Async API" section covering
  `AsyncTensogramFile`, all methods, concurrency semantics, when to use async vs sync.
- **`docs/src/guide/remote-access.md`**: Add note about async remote access via
  `AsyncTensogramFile.open`.
- **Examples table in docs**: Add entry for `15_async_operations.py`.

### Step 11: CI

**File**: `.github/workflows/ci.yml`

No new test command needed; the existing `python -m pytest tests/python/ -v` already
runs all files including `test_async.py`.

Changes:
- Add `pytest-asyncio` to `uv pip install` in **every** CI job that runs `tests/python/`
  (standard Python job, free-threaded Python job if present).
- Verify `conftest.py` extraction does not break `test_remote.py` (run it in isolation).
- Add a verification step: `maturin develop --no-default-features` + basic sync test
  to confirm the feature-gate produces no dead code warnings.

## 5. API Surface Summary

### `AsyncTensogramFile` class

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `open` | `await .open(source)` | `AsyncTensogramFile` | Static, auto-detects local/remote |
| `open_remote` | `await .open_remote(source, storage_options=None)` | `AsyncTensogramFile` | Static, explicit remote |
| `decode_message` | `await .decode_message(index, verify_hash=False, native_byte_order=True)` | `Message` namedtuple | Full message decode |
| `file_decode_metadata` | `await .file_decode_metadata(msg_index)` | `Metadata` | Metadata only |
| `file_decode_descriptors` | `await .file_decode_descriptors(msg_index)` | `dict` | Metadata + descriptors |
| `file_decode_object` | `await .file_decode_object(msg_index, obj_index, verify_hash=False, native_byte_order=True)` | `dict` | Single object |
| `read_message` | `await .read_message(index)` | `bytes` | Raw wire bytes |
| `message_count` | `await .message_count()` | `int` | Async (locks mutex, calls sync count) |
| `is_remote` | `.is_remote()` | `bool` | Sync (cached field) |
| `source` | `.source()` | `str` | Sync (cached field) |
| `__repr__` | `repr(f)` | `str` | Sync |

### Deferred (not in Foundation)

| Feature | Reason | Escalation trigger |
|---|---|---|
| `__aenter__`/`__aexit__` | No close/shutdown semantics exist | Add when `close()` is introduced |
| `__len__` | Would need sync I/O on async class; use `await message_count()` instead | Revisit if needed |
| `__aiter__`/`__anext__` | Complex protocol; gather covers primary use case | Add if user demand |
| `file_decode_range` async | No `TensogramFile::decode_range_async` in core | Add when core exposes it |
| Async encode | CPU-bound; `py.detach()` sync already optimal | No escalation expected |

## 6. Risk Assessment

### Resolved

1. **pyo3-async-runtimes version**: 0.28.0 exists, compatible with pyo3 0.28.
2. **Tokio runtime**: independent runtimes, no conflict.
3. **GIL/mutex ordering**: mutex released before `Python::attach()` prevents deadlocks.
4. **`remote` feature availability**: hard-enabled, so `open_source_async` always available.
5. **`blocking_lock()` panic**: avoided by making `message_count()` fully async (locks mutex via `.lock().await`, not `blocking_lock`).
6. **Argument validation timing**: `open_remote` validates `storage_options` eagerly.
7. **Cancellation safety**: tokio mutex is cancel-safe; dropping the future while the
   mutex is locked does not leave it wedged (tokio guarantees this).

### Risks

1. **Breaking `test_remote.py`**: Extracting `serve_tgm_bytes` to `conftest.py` must be
   done carefully.  Run `test_remote.py` in isolation after extraction.
2. **Two tokio runtimes**: if core async methods internally call `block_on_shared()` while
   already running on the pyo3-async-runtimes tokio, the `block_on_shared` function handles
   this correctly (block_in_place for multi-thread, scoped thread for current-thread).
3. **`default = ["async"]` changes build surface**: Must verify through `maturin develop`,
   not just `cargo build`.  CI should test both default and `--no-default-features` paths.

### Open questions (to resolve during implementation)

1. **`pytest-asyncio` mode**: Use `mode = "auto"` or explicit `@pytest.mark.asyncio`?
   Recommendation: explicit markers for clarity.

## 7. Non-Goals (out of scope)

- **Async encode**: CPU-bound, `py.detach()` already optimal.
- **Async xarray/zarr backends**: Synchronous frameworks; async benefit from direct API.
- **Free-threaded Python**: Separate TODO item.  `AsyncTensogramFile` uses `Arc + Mutex`
  internally, but end-to-end free-threaded readiness is a separate, broader effort.
- **True same-handle parallel I/O**: Requires core `TensogramFile` mutability refactor.
  Foundation provides correct asyncio integration; parallelism can be added later.

# Plan: AsyncTensogramFile — Phase 2 (Previously Deferred Features)

**Status**: Implemented

> **Note**: This plan was written for the TokioMutex-based architecture.
> The implementation evolved to use `Arc<TensogramFile>` with `&self` async methods
> (no mutex), eliminating all the `blocking_lock` complexity discussed below.
> Also added: `decode_range_async`, `decode_range_batch_async` (batched HTTP),
> `prefetch_layouts_async`, and sync `decode_range_batch`.
**Branch**: `feature/python-async-bindings` (continuing)
**Prerequisite**: Phase 1 complete (34 async tests, 10 methods)

## 1. Goal

Complete the `AsyncTensogramFile` API so it has full parity with sync `TensogramFile`
for all read/decode operations.  After this phase, every read operation available on
the sync class has an async equivalent.

## 2. Features to implement

| Feature | Sync equivalent | Notes |
|---|---|---|
| `file_decode_range` | `PyTensogramFile.file_decode_range` | No core async version; use `spawn_blocking` + sync path |
| `__aenter__`/`__aexit__` | `__enter__`/`__exit__` | Async context manager protocol |
| `__len__` | `__len__` | Sync `len(f)`, calls `message_count` under `spawn_blocking` |
| `__aiter__`/`__anext__` | `__iter__`/`__next__` via `PyFileIter` | Async iteration yielding `Message` namedtuples |

## 3. Implementation details

### 3.1 `file_decode_range` (async)

The core `TensogramFile::decode_range` is sync-only (no `decode_range_async` exists).
The async binding wraps the sync call in `spawn_blocking`, following the same pattern
as `message_count`.

**Sync signature** (`lib.rs:509-535`):
```python
file_decode_range(msg_index, obj_index, ranges, join=False, verify_hash=False, native_byte_order=True)
```

**Async pattern**:
```rust
#[pyo3(signature = (msg_index, obj_index, ranges, join=false, verify_hash=false, native_byte_order=true))]
#[allow(clippy::too_many_arguments)]
fn file_decode_range<'py>(
    &self,
    py: Python<'py>,
    msg_index: usize,
    obj_index: usize,
    ranges: Vec<(u64, u64)>,
    join: bool,
    verify_hash: bool,
    native_byte_order: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let file = Arc::clone(&self.file);
    let options = DecodeOptions { verify_hash, native_byte_order, ..Default::default() };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (desc, parts) = tokio::task::spawn_blocking(move || {
            let f = file.blocking_lock();
            f.decode_range(msg_index, obj_index, &ranges, &options)
                .map_err(to_py_err)
                .map(|r| (r, ranges))
        })
        .await
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("task join: {e}")))?
        // Unpack: spawn_blocking returned (Result<(desc, parts)>, ranges)
        // Actually, need to restructure — see implementation step for clean version
        ?;
        // NOTE: Actual implementation will capture `ranges` in the closure
        // and pass both desc+parts+ranges to Python::attach for build_range_result.
        todo!("see implementation step")
    })
}
```

**Key complexity**: `build_range_result` needs `py: Python`, `desc.dtype`, `parts`, `ranges`, `join`.
All are `Send + 'static` except `py`.  We run `build_range_result` inside `Python::attach`
after the `spawn_blocking` completes.

**Clean implementation**:
```rust
fn file_decode_range<'py>(...) -> PyResult<Bound<'py, PyAny>> {
    let file = Arc::clone(&self.file);
    let options = DecodeOptions { ... };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (desc, parts) = tokio::task::spawn_blocking(move || {
            let f = file.blocking_lock();
            f.decode_range(msg_index, obj_index, &ranges, &options).map_err(to_py_err)
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))??;

        let dtype = desc.dtype;
        Python::attach(|py| build_range_result(py, dtype, parts, &ranges, join))
    })
}
```

**Problem**: `ranges` is moved into the `spawn_blocking` closure but also needed in
`Python::attach`.  Fix: clone `ranges` before the closure, or restructure to return it.

**Solution**: Clone ranges before spawn_blocking:
```rust
let ranges_for_result = ranges.clone();
// ... spawn_blocking uses `ranges` ...
// ... Python::attach uses `ranges_for_result` ...
```

### 3.2 `__aenter__` / `__aexit__` (async context manager)

The sync `__enter__` returns `self`.  The sync `__exit__` returns `false` (don't suppress).

For async:
- `__aenter__` must return an awaitable that resolves to `self`.
- `__aexit__` must return an awaitable that resolves to `false`.

pyo3 0.28 does not have native async dunder support for `__aenter__`/`__aexit__`.
We implement these as regular methods returning coroutines via `future_into_py`.

```rust
fn __aenter__<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let obj = slf.unbind();
    pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(obj) })
}

#[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
fn __aexit__<'py>(
    &self,
    py: Python<'py>,
    _exc_type: Option<&Bound<'_, pyo3::PyAny>>,
    _exc_val: Option<&Bound<'_, pyo3::PyAny>>,
    _exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(false) })
}
```

**Usage**:
```python
async with await tensogram.AsyncTensogramFile.open("data.tgm") as f:
    result = await f.file_decode_object(0, 0)
```

Note: `async with await open(...)` is the correct Python pattern when the constructor
is itself a coroutine.  The `await` resolves the open, then `async with` calls
`__aenter__`/`__aexit__`.

### 3.3 `__len__`

The sync `__len__` calls `self.file.message_count()` which does lazy I/O.
For the async class, we use `spawn_blocking` to avoid blocking the tokio worker,
matching the `message_count()` pattern.

```rust
fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
    let file = Arc::clone(&self.file);
    let count = std::thread::scope(|s| {
        s.spawn(move || {
            let f = file.blocking_lock();
            f.message_count().map_err(to_py_err)
        })
        .join()
        .map_err(|_| PyRuntimeError::new_err("thread panicked"))?
    })?;
    Ok(count)
}
```

**Why `std::thread::scope` instead of `spawn_blocking`**: `__len__` is a sync dunder —
it must return `usize` immediately, not a coroutine.  We can't use `await`.  We use
a scoped thread to avoid blocking the current thread (which might be a tokio worker)
while still avoiding `blocking_lock()` on the current thread.

**Alternative** (simpler, acceptable for `__len__`): Just use `blocking_lock` directly.
The rationale: `__len__` is called from sync Python code, never from inside a tokio
runtime context.  Python's `len()` is inherently sync.

```rust
fn __len__(&self) -> PyResult<usize> {
    self.file.blocking_lock().message_count().map_err(to_py_err)
}
```

**Decision**: Use the simple `blocking_lock` approach.  `len()` is called from sync
Python code.  If someone calls `len()` on an `AsyncTensogramFile` from within an
async context, they've already accepted sync semantics.  Document this.

### 3.4 `__aiter__` / `__anext__` (async iteration)

The sync `__iter__` creates a `PyFileIter` that opens a separate file handle and
iterates by index.  It does NOT work on remote files (raises RuntimeError).

For async iteration, we need:
1. An `AsyncTensogramFileIter` pyclass holding its own `Arc<TokioMutex<TensogramFile>>`
   plus index/count state.
2. `__aiter__` returns `self`.
3. `__anext__` returns a coroutine that resolves to the next message or raises
   `StopAsyncIteration`.

**Unlike sync iteration**, async iteration CAN work on remote files since it uses
the async decode path.

```rust
#[cfg(feature = "async")]
#[pyclass(name = "AsyncTensogramFileIter")]
struct PyAsyncTensogramFileIter {
    file: Arc<TokioMutex<TensogramFile>>,
    index: std::sync::atomic::AtomicUsize,
    count: usize,
}

#[cfg(feature = "async")]
#[pymethods]
impl PyAsyncTensogramFileIter {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let idx = self.index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if idx >= self.count {
            return Ok(None); // StopAsyncIteration
        }
        let file = Arc::clone(&self.file);
        let options = DecodeOptions::default();
        Ok(Some(pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move {
                let (meta, objects) = {
                    let mut f = file.lock().await;
                    f.decode_message_async(idx, &options).await.map_err(to_py_err)?
                };
                Python::attach(|py| {
                    let result_list = data_objects_to_python(py, &objects)?;
                    pack_message(py, PyMetadata { inner: meta }, result_list)
                })
            },
        )?))
    }

    fn __len__(&self) -> usize {
        let current = self.index.load(std::sync::atomic::Ordering::Relaxed);
        self.count.saturating_sub(current)
    }

    fn __repr__(&self) -> String {
        let current = self.index.load(std::sync::atomic::Ordering::Relaxed);
        let remaining = self.count.saturating_sub(current);
        format!("AsyncTensogramFileIter(position={current}, remaining={remaining})")
    }
}
```

**`__aiter__` on `PyAsyncTensogramFile`**:
```rust
fn __aiter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let file = Arc::clone(&self.file);
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let count = {
            let f = file.lock().await;
            f.message_count().map_err(to_py_err)?
        };
        let iter = PyAsyncTensogramFileIter {
            file,
            index: std::sync::atomic::AtomicUsize::new(0),
            count,
        };
        Python::attach(|py| Ok(iter.into_pyobject(py)?.into_any().unbind()))
    })
}
```

**Wait — `__aiter__` must return the iterator synchronously**, not as a coroutine.
The Python async iteration protocol requires `__aiter__` to return the iterator object
directly (not an awaitable).  Only `__anext__` returns awaitables.

**Corrected `__aiter__`**: Get message count synchronously (using `blocking_lock` or
cached count), construct the iterator, return it directly.

```rust
fn __aiter__(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<PyAsyncTensogramFileIter> {
    let this = slf.borrow();
    let file = Arc::clone(&this.file);
    let count = file.blocking_lock().message_count().map_err(to_py_err)?;
    Ok(PyAsyncTensogramFileIter {
        file,
        index: std::sync::atomic::AtomicUsize::new(0),
        count,
    })
}
```

**Problem**: `blocking_lock()` can panic inside a tokio context.  But `__aiter__` is
called by Python's `async for` machinery, which runs in the event loop thread (not a
tokio worker).  This is safe in practice.

**Better approach**: Cache message_count at first `__aiter__` call, or require the user
to call `await f.message_count()` first.

**Best approach**: Since we already have `message_count()` as async, we cache the result
on the struct after first call.  But that adds complexity.

**Simplest correct approach**: Use `blocking_lock()` in `__aiter__`.  Document that
`async for f` may briefly block on first use if the message count hasn't been cached.
This matches how the sync `__iter__` also does blocking I/O to scan offsets.

## 4. Implementation steps

### Step 1: `file_decode_range`
- Add method to `PyAsyncTensogramFile` impl block
- Test: basic range, join mode, out-of-range, parity with sync

### Step 2: `__aenter__` / `__aexit__`
- Add both methods to impl block
- Test: `async with await open(...)` pattern, exception propagation

### Step 3: `__len__`
- Add `__len__` using `blocking_lock`
- Test: `len(f)`, parity with sync len

### Step 4: `__aiter__` / `__anext__`
- Add `PyAsyncTensogramFileIter` pyclass
- Add `__aiter__` on `PyAsyncTensogramFile`
- Register new class in module
- Test: `async for msg in f`, count, early break, parity with sync iter, remote iteration

### Step 5: Documentation
- Update `docs/src/guide/python-api.md` Async API section with new methods
- Update API table in plan
- Add iteration and context manager examples

### Step 6: Tests
Full test list for all new features.

## 5. Test plan

```
# file_decode_range
test_file_decode_range_basic
test_file_decode_range_join
test_file_decode_range_out_of_range
test_file_decode_range_matches_sync

# context manager
test_async_context_manager
test_async_context_manager_exception_propagates

# __len__
test_len
test_len_matches_sync

# async iteration
test_aiter_all_messages
test_aiter_count
test_aiter_early_break
test_aiter_matches_sync_iter
test_aiter_repr
test_aiter_remote  (async for on remote file — sync __iter__ can't do this)
```

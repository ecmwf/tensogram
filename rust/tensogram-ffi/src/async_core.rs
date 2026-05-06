// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Async FFI core.
//!
//! Provides opaque task handles, cancellation tokens, and runtime
//! configuration for the C surface that exposes Tensogram's async
//! reads to C and C++ callers.  Spec: `plans/PLAN_CPP_ASYNC.md` §3.
//!
//! The runtime is fully contained: no tokio handle, executor, or any
//! tokio type ever appears in the public C API.  A process-global
//! tokio runtime is built lazily on first call and can be configured
//! once via [`tgm_runtime_configure`].

#![cfg(feature = "async")]

use std::ffi::{CStr, CString, c_void};
use std::os::raw::c_char;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use tensogram::{
    DataObjectDescriptor, DecodeOptions, GlobalMetadata, MessageLayout, TensogramError,
    TensogramFile,
};

use crate::{TgmBytes, TgmError, TgmMessage, TgmMetadata, set_last_error, to_error_code};

// ---------------------------------------------------------------------------
// Shared runtime
// ---------------------------------------------------------------------------

struct RuntimeConfig {
    workers: u32,
    /// Sized for callback dispatcher pool (PR 4 will consume).
    #[allow(dead_code)]
    dispatcher_workers: u32,
    /// Multipart upload part size for PR 3 streaming-write backends.
    #[allow(dead_code)]
    multipart_part_size_bytes: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            workers: std::cmp::min(num_cpus_or_default(), 8),
            dispatcher_workers: std::cmp::min(num_cpus_or_default(), 4),
            multipart_part_size_bytes: 8 * 1024 * 1024,
        }
    }
}

fn num_cpus_or_default() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();
static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();

fn runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get_or_init(RuntimeConfig::default)
}

fn runtime() -> Result<&'static tokio::runtime::Runtime, &'static str> {
    let cfg = runtime_config();
    match RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cfg.workers as usize)
            .enable_all()
            .thread_name("tensogram-async")
            .build()
            .map_err(|e| format!("failed to build tokio runtime: {e}"))
    }) {
        Ok(rt) => Ok(rt),
        Err(s) => Err(s.as_str()),
    }
}

// ---------------------------------------------------------------------------
// Task lifecycle
// ---------------------------------------------------------------------------

/// Result variants surfaced by an async task.  The variant must match
/// the join function the caller invokes; mismatches surface as
/// [`TgmError::InvalidArg`].
#[allow(dead_code)] // some variants reserved for PRs 4-5
pub enum TaskResult {
    File(Box<crate::TgmFile>),
    AsyncFile(Box<TgmAsyncFile>),
    AsyncStreamingEncoder(Box<crate::async_streaming::TgmAsyncStreamingEncoder>),
    Message(Box<TgmMessage>),
    Metadata(Box<TgmMetadata>),
    Bytes(Vec<u8>),
    /// Multiple decoded ranges; one `Vec<u8>` per requested range.
    MultiBytes(Vec<Vec<u8>>),
    Size(u64),
    Layouts(Vec<MessageLayout>),
    Void,
}

#[derive(PartialEq)]
enum TaskState {
    Pending,
    Ready,
    Consumed,
}

struct TaskInner {
    state: TaskState,
    result: Option<Result<TaskResult, TensogramError>>,
    completion_cb: Option<extern "C" fn(*mut c_void)>,
    completion_userdata: *mut c_void,
}

// SAFETY: completion_userdata is a caller-controlled pointer used only
// by the caller's callback.  Rust never dereferences it.
unsafe impl Send for TaskInner {}

/// Opaque task handle exposed to C as `tgm_async_task_t*`.
pub struct TgmAsyncTask {
    inner: Arc<TaskShared>,
}

struct TaskShared {
    state: Mutex<TaskInner>,
    ready: Condvar,
    cancel: tokio_util::sync::CancellationToken,
    /// External cancellation token (refcounted via Arc on the C side).
    /// Held to keep the token alive for the task's lifetime.
    _external_token: Option<Arc<TgmCancellationTokenInner>>,
}

impl TaskShared {
    fn new(external: Option<&TgmCancellationToken>) -> Arc<Self> {
        let cancel = tokio_util::sync::CancellationToken::new();
        // Link external token to internal: cancellation propagates.
        let external_clone = external.map(|t| t.inner.clone());
        if let Some(ext) = external {
            let internal = cancel.clone();
            let ext_token = ext.inner.token.clone();
            // Forward external cancellation to the task's internal token.
            // This task is detached on the runtime; if cancelled before
            // it ever polls, no harm done.
            if let Ok(rt) = runtime() {
                rt.spawn(async move {
                    ext_token.cancelled().await;
                    internal.cancel();
                });
            }
        }
        Arc::new(Self {
            state: Mutex::new(TaskInner {
                state: TaskState::Pending,
                result: None,
                completion_cb: None,
                completion_userdata: std::ptr::null_mut(),
            }),
            ready: Condvar::new(),
            cancel,
            _external_token: external_clone,
        })
    }

    /// Mark the task as Ready and dispatch the completion callback if
    /// one was registered.  Called once per task by the runtime.
    fn complete(&self, result: Result<TaskResult, TensogramError>) {
        let cb_to_fire = {
            let mut inner = self.state.lock().expect("task mutex poisoned");
            if inner.state != TaskState::Pending {
                // Already consumed / never armed.  Drop the result.
                return;
            }
            inner.result = Some(result);
            inner.state = TaskState::Ready;
            // Snapshot the callback so we can release the lock before
            // firing it.  The callback may call into FFI again and we
            // must not hold the task mutex across that boundary.
            inner.completion_cb.take().map(|cb| {
                (
                    cb,
                    std::mem::replace(&mut inner.completion_userdata, std::ptr::null_mut()),
                )
            })
        };
        self.ready.notify_all();
        if let Some((cb, userdata)) = cb_to_fire {
            cb(userdata);
        }
    }
}

/// Spawn a task on the shared runtime that resolves `fut` into a
/// [`TaskResult`].  Returns the task handle (heap-allocated for FFI).
///
/// Internally implements the timeout via `tokio::time::timeout` and
/// honours the cancellation token via `tokio::select!`.
pub(crate) fn spawn_task<F>(
    fut: F,
    cancel: Option<&TgmCancellationToken>,
    timeout_ms: u64,
) -> Result<*mut TgmAsyncTask, String>
where
    F: std::future::Future<Output = Result<TaskResult, TensogramError>> + Send + 'static,
{
    let rt = runtime()?;
    let shared = TaskShared::new(cancel);
    let shared_clone = shared.clone();
    let task_token = shared.cancel.clone();

    rt.spawn(async move {
        let outcome = if timeout_ms > 0 {
            let deadline = Duration::from_millis(timeout_ms);
            tokio::select! {
                _ = task_token.cancelled() => Err(TensogramError::Encoding(
                    "async task cancelled".to_string(),
                )),
                res = tokio::time::timeout(deadline, fut) => match res {
                    Ok(inner) => inner,
                    Err(_elapsed) => Err(TensogramError::Encoding(
                        "async task timed out".to_string(),
                    )),
                },
            }
        } else {
            tokio::select! {
                _ = task_token.cancelled() => Err(TensogramError::Encoding(
                    "async task cancelled".to_string(),
                )),
                res = fut => res,
            }
        };
        shared_clone.complete(outcome);
    });

    Ok(Box::into_raw(Box::new(TgmAsyncTask { inner: shared })))
}

/// Internal helper: classify an error into `Cancelled` / `Timeout` /
/// generic.  Looks at the message we set in [`spawn_task`] above.
fn classify_async_error(e: &TensogramError) -> TgmError {
    match e {
        TensogramError::Encoding(s) if s == "async task cancelled" => TgmError::Cancelled,
        TensogramError::Encoding(s) if s == "async task timed out" => TgmError::Timeout,
        other => to_error_code(other),
    }
}

// ---------------------------------------------------------------------------
// Cancellation tokens
// ---------------------------------------------------------------------------

pub(crate) struct TgmCancellationTokenInner {
    token: tokio_util::sync::CancellationToken,
}

/// Opaque cancellation-token handle exposed to C as
/// `tgm_cancellation_token_t*`.
pub struct TgmCancellationToken {
    inner: Arc<TgmCancellationTokenInner>,
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_create() -> *mut TgmCancellationToken {
    Box::into_raw(Box::new(TgmCancellationToken {
        inner: Arc::new(TgmCancellationTokenInner {
            token: tokio_util::sync::CancellationToken::new(),
        }),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_cancel(tok: *mut TgmCancellationToken) {
    if tok.is_null() {
        return;
    }
    let t = unsafe { &*tok };
    t.inner.token.cancel();
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_is_cancelled(tok: *const TgmCancellationToken) -> bool {
    if tok.is_null() {
        return false;
    }
    let t = unsafe { &*tok };
    t.inner.token.is_cancelled()
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_free(tok: *mut TgmCancellationToken) {
    if !tok.is_null() {
        unsafe { drop(Box::from_raw(tok)) };
    }
}

// ---------------------------------------------------------------------------
// Async task — public API
// ---------------------------------------------------------------------------

/// Register a completion callback on `task`.  Must be called exactly
/// once.  The callback fires on a dispatcher pool worker (NOT a tokio
/// worker) and must obey the §4.1 contract documented in the plan.
///
/// If the task has already resolved, the callback is invoked inline
/// on the calling thread before the function returns.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_set_completion(
    task: *mut TgmAsyncTask,
    cb: extern "C" fn(*mut c_void),
    userdata: *mut c_void,
) -> TgmError {
    if task.is_null() {
        set_last_error("null task");
        return TgmError::InvalidArg;
    }
    let t = unsafe { &*task };

    // Snapshot whether we should fire inline (state already Ready) or
    // store the callback for the resolver to fire later.  Lock is held
    // only across the inspect-and-store step.
    let fire_inline = {
        let mut inner = t.inner.state.lock().expect("task mutex poisoned");
        if inner.completion_cb.is_some() {
            set_last_error("completion callback already registered");
            return TgmError::InvalidArg;
        }
        match inner.state {
            TaskState::Pending => {
                inner.completion_cb = Some(cb);
                inner.completion_userdata = userdata;
                false
            }
            TaskState::Ready => true,
            TaskState::Consumed => {
                set_last_error("task result already consumed");
                return TgmError::InvalidArg;
            }
        }
    };

    if fire_inline {
        cb(userdata);
    }
    TgmError::Ok
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_is_ready(task: *const TgmAsyncTask) -> bool {
    if task.is_null() {
        return false;
    }
    let t = unsafe { &*task };
    let inner = t.inner.state.lock().expect("task mutex poisoned");
    inner.state != TaskState::Pending
}

/// Cancel an in-flight task by signalling its internal token.  The
/// task transitions to [`TgmError::Cancelled`] at the next yield point.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_cancel(task: *mut TgmAsyncTask) {
    if task.is_null() {
        return;
    }
    let t = unsafe { &*task };
    t.inner.cancel.cancel();
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_free(task: *mut TgmAsyncTask) {
    if !task.is_null() {
        unsafe { drop(Box::from_raw(task)) };
    }
}

// Block waiting for the task to be ready, then extract the result.
// Caller is responsible for type-matching the variant.  Sibling
// modules in this crate (e.g. `async_streaming`) implement their own
// typed join functions on top of this.
pub(crate) fn join_internal(task: *mut TgmAsyncTask) -> Result<TaskResult, TgmError> {
    if task.is_null() {
        set_last_error("null task");
        return Err(TgmError::InvalidArg);
    }
    let t = unsafe { &*task };
    let mut inner = t.inner.state.lock().expect("task mutex poisoned");
    while inner.state == TaskState::Pending {
        inner = t.inner.ready.wait(inner).expect("task condvar poisoned");
    }
    if inner.state == TaskState::Consumed {
        set_last_error("task result already consumed");
        return Err(TgmError::InvalidArg);
    }
    let res = inner.result.take().expect("ready task missing result");
    inner.state = TaskState::Consumed;
    drop(inner);
    res.map_err(|e| {
        let code = classify_async_error(&e);
        set_last_error(&e.to_string());
        code
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_void(task: *mut TgmAsyncTask) -> TgmError {
    match join_internal(task) {
        Ok(TaskResult::Void) => TgmError::Ok,
        Ok(_) => {
            set_last_error("task result type mismatch (expected void)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_size(task: *mut TgmAsyncTask, out: *mut u64) -> TgmError {
    if out.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match join_internal(task) {
        Ok(TaskResult::Size(n)) => {
            unsafe { *out = n };
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected size)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_bytes(
    task: *mut TgmAsyncTask,
    out: *mut TgmBytes,
) -> TgmError {
    if out.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match join_internal(task) {
        Ok(TaskResult::Bytes(mut v)) => {
            v.shrink_to_fit();
            let len = v.len();
            let ptr = v.as_mut_ptr();
            std::mem::forget(v);
            unsafe {
                (*out).data = ptr;
                (*out).len = len;
            }
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected bytes)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_message(
    task: *mut TgmAsyncTask,
    out: *mut *mut TgmMessage,
) -> TgmError {
    if out.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match join_internal(task) {
        Ok(TaskResult::Message(m)) => {
            unsafe { *out = Box::into_raw(m) };
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected message)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_metadata(
    task: *mut TgmAsyncTask,
    out: *mut *mut TgmMetadata,
) -> TgmError {
    if out.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match join_internal(task) {
        Ok(TaskResult::Metadata(m)) => {
            unsafe { *out = Box::into_raw(m) };
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected metadata)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_async_file(
    task: *mut TgmAsyncTask,
    out: *mut *mut TgmAsyncFile,
) -> TgmError {
    if out.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match join_internal(task) {
        Ok(TaskResult::AsyncFile(f)) => {
            unsafe { *out = Box::into_raw(f) };
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected async_file)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_join_multi_bytes(
    task: *mut TgmAsyncTask,
    out_array: *mut *mut TgmBytes,
    out_count: *mut usize,
) -> TgmError {
    if out_array.is_null() || out_count.is_null() {
        set_last_error("null out pointer");
        return TgmError::InvalidArg;
    }
    match join_internal(task) {
        Ok(TaskResult::MultiBytes(parts)) => {
            let count = parts.len();
            // Build a Vec<TgmBytes> on the heap; caller frees with
            // tgm_multi_bytes_free.
            let mut entries: Vec<TgmBytes> = parts
                .into_iter()
                .map(|mut v| {
                    v.shrink_to_fit();
                    let len = v.len();
                    let ptr = v.as_mut_ptr();
                    std::mem::forget(v);
                    TgmBytes { data: ptr, len }
                })
                .collect();
            entries.shrink_to_fit();
            let ptr = entries.as_mut_ptr();
            std::mem::forget(entries);
            unsafe {
                *out_array = ptr;
                *out_count = count;
            }
            TgmError::Ok
        }
        Ok(_) => {
            set_last_error("task result type mismatch (expected multi_bytes)");
            TgmError::InvalidArg
        }
        Err(code) => code,
    }
}

/// Free an array of `TgmBytes` returned by `tgm_async_task_join_multi_bytes`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_multi_bytes_free(array: *mut TgmBytes, count: usize) {
    if array.is_null() {
        return;
    }
    unsafe {
        let v = Vec::from_raw_parts(array, count, count);
        for entry in v {
            crate::tgm_bytes_free(entry);
        }
    }
}

// ---------------------------------------------------------------------------
// Async file handle
// ---------------------------------------------------------------------------

/// Async file handle backed by `Arc<TensogramFile>` so the same handle
/// is safely shareable across multiple in-flight tasks.
pub struct TgmAsyncFile {
    file: Arc<TensogramFile>,
    path_string: CString,
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_path(file: *const TgmAsyncFile) -> *const c_char {
    if file.is_null() {
        return std::ptr::null();
    }
    unsafe { (*file).path_string.as_ptr() }
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_close(file: *mut TgmAsyncFile) {
    if !file.is_null() {
        unsafe { drop(Box::from_raw(file)) };
    }
}

// ---------------------------------------------------------------------------
// Open / open_remote
// ---------------------------------------------------------------------------

/// Open a Tensogram file asynchronously.  Returns immediately with a
/// task handle; join via `tgm_async_task_join_async_file`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_open(
    path: *const c_char,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if path.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in path: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };

    let path_for_task = path_str.clone();
    let fut = async move {
        let f = TensogramFile::open(&path_for_task)?;
        let path_string = CString::new(path_for_task.as_str()).unwrap_or_default();
        Ok(TaskResult::AsyncFile(Box::new(TgmAsyncFile {
            file: Arc::new(f),
            path_string,
        })))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

#[cfg(feature = "async-remote")]
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_open_remote(
    url: *const c_char,
    storage_keys: *const *const c_char,
    storage_values: *const *const c_char,
    nopts: usize,
    bidirectional: bool,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if url.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let url_str = match unsafe { CStr::from_ptr(url) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in url: {e}"));
            return TgmError::InvalidArg;
        }
    };
    let mut storage = std::collections::BTreeMap::new();
    if nopts > 0 {
        if storage_keys.is_null() || storage_values.is_null() {
            set_last_error("null storage opts pointer");
            return TgmError::InvalidArg;
        }
        for i in 0..nopts {
            let kp = unsafe { *storage_keys.add(i) };
            let vp = unsafe { *storage_values.add(i) };
            if kp.is_null() || vp.is_null() {
                set_last_error("null storage opt entry");
                return TgmError::InvalidArg;
            }
            let k = match unsafe { CStr::from_ptr(kp) }.to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in storage key: {e}"));
                    return TgmError::InvalidArg;
                }
            };
            let v = match unsafe { CStr::from_ptr(vp) }.to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in storage value: {e}"));
                    return TgmError::InvalidArg;
                }
            };
            storage.insert(k, v);
        }
    }
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };

    let scan_opts = tensogram::RemoteScanOptions { bidirectional };
    let url_for_task = url_str.clone();
    let fut = async move {
        let f = TensogramFile::open_remote_async(&url_for_task, &storage, Some(scan_opts)).await?;
        let path_string = CString::new(url_for_task.as_str()).unwrap_or_default();
        Ok(TaskResult::AsyncFile(Box::new(TgmAsyncFile {
            file: Arc::new(f),
            path_string,
        })))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

// ---------------------------------------------------------------------------
// File-level async operations
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_message_count(
    file: *mut TgmAsyncFile,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if file.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &*file }.file.clone();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };
    let fut = async move {
        let n = f.message_count_async().await?;
        Ok(TaskResult::Size(n as u64))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_read_message(
    file: *mut TgmAsyncFile,
    index: usize,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if file.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &*file }.file.clone();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };
    let fut = async move {
        let bytes = f.read_message_async(index).await?;
        Ok(TaskResult::Bytes(bytes))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_decode_message(
    file: *mut TgmAsyncFile,
    index: usize,
    native_byte_order: bool,
    threads: u32,
    restore_non_finite: bool,
    verify_hash: bool,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if file.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &*file }.file.clone();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };
    let opts = DecodeOptions {
        native_byte_order,
        threads,
        restore_non_finite,
        verify_hash,
        ..Default::default()
    };
    let fut = async move {
        let (gm, objs) = f.decode_message_async(index, &opts).await?;
        Ok(TaskResult::Message(Box::new(build_tgm_message(gm, objs))))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_decode_metadata(
    file: *mut TgmAsyncFile,
    index: usize,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if file.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &*file }.file.clone();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };
    let fut = async move {
        let gm = f.decode_metadata_async(index).await?;
        Ok(TaskResult::Metadata(Box::new(TgmMetadata {
            global_metadata: gm,
            cache: std::cell::RefCell::new(std::collections::BTreeMap::new()),
        })))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_decode_object(
    file: *mut TgmAsyncFile,
    msg_index: usize,
    obj_index: usize,
    native_byte_order: bool,
    threads: u32,
    restore_non_finite: bool,
    verify_hash: bool,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if file.is_null() || out_task.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &*file }.file.clone();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };
    let opts = DecodeOptions {
        native_byte_order,
        threads,
        restore_non_finite,
        verify_hash,
        ..Default::default()
    };
    let fut = async move {
        let (gm, desc, payload) = f.decode_object_async(msg_index, obj_index, &opts).await?;
        // Wrap as a single-object TgmMessage so the existing accessors
        // (tgm_object_*, tgm_message_*) work transparently.
        Ok(TaskResult::Message(Box::new(build_tgm_message(
            gm,
            vec![(desc, payload)],
        ))))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_decode_range(
    file: *mut TgmAsyncFile,
    msg_index: usize,
    obj_index: usize,
    offsets: *const u64,
    counts: *const u64,
    n_ranges: usize,
    native_byte_order: bool,
    threads: u32,
    cancel: *mut TgmCancellationToken,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError {
    if file.is_null() || out_task.is_null() || offsets.is_null() || counts.is_null() {
        set_last_error("null argument");
        return TgmError::InvalidArg;
    }
    let f = unsafe { &*file }.file.clone();
    let cancel_ref = if cancel.is_null() {
        None
    } else {
        Some(unsafe { &*cancel })
    };
    let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(n_ranges);
    for i in 0..n_ranges {
        let o = unsafe { *offsets.add(i) };
        let c = unsafe { *counts.add(i) };
        ranges.push((o, c));
    }
    let opts = DecodeOptions {
        native_byte_order,
        threads,
        ..Default::default()
    };
    let fut = async move {
        let (_desc, parts) = f
            .decode_range_async(msg_index, obj_index, &ranges, &opts)
            .await?;
        Ok(TaskResult::MultiBytes(parts))
    };
    spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
}

// ---------------------------------------------------------------------------
// Runtime configuration
// ---------------------------------------------------------------------------

/// Configure the FFI tokio runtime.  Must be called before any other
/// `tgm_async_*` call; subsequent calls return [`TgmError::InvalidArg`].
///
/// Pass `0` to use the default for any field.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_runtime_configure(
    workers: u32,
    dispatcher_workers: u32,
    multipart_part_size_bytes: u64,
) -> TgmError {
    let cfg = RuntimeConfig {
        workers: if workers == 0 {
            std::cmp::min(num_cpus_or_default(), 8)
        } else {
            workers
        },
        dispatcher_workers: if dispatcher_workers == 0 {
            std::cmp::min(num_cpus_or_default(), 4)
        } else {
            dispatcher_workers
        },
        multipart_part_size_bytes: if multipart_part_size_bytes == 0 {
            8 * 1024 * 1024
        } else {
            multipart_part_size_bytes
        },
    };
    if RUNTIME_CONFIG.set(cfg).is_err() {
        set_last_error("runtime already configured or built");
        return TgmError::InvalidArg;
    }
    TgmError::Ok
}

/// Reserved ABI for graceful shutdown.  Currently a no-op that
/// always returns `0`.
///
/// **Status (v1):** the shared runtime lives behind a `OnceLock` and
/// cannot be torn down without leaking the slot.  Process exit is
/// abrupt by design (see `plans/PLAN_CPP_ASYNC.md` §6); in-flight
/// tasks are dropped by tokio at process teardown.
///
/// The signature is reserved here so a future implementation can
/// switch the singleton to an owning container, drain tasks
/// cooperatively, and return the count that did not finish within
/// `timeout_ms` — without breaking the C ABI.  The argument is
/// accepted but ignored today.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_runtime_shutdown_blocking(timeout_ms: u64) -> u64 {
    let _ = timeout_ms;
    0
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Spawn `fut` on the shared runtime and write the resulting task
/// handle into `out_task`.  Sibling modules in this crate (e.g.
/// `async_streaming`) call this to launch their own tasks without
/// re-implementing the runtime / cancellation / timeout plumbing.
pub(crate) fn spawn_or_set_error<F>(
    fut: F,
    cancel: Option<&TgmCancellationToken>,
    timeout_ms: u64,
    out_task: *mut *mut TgmAsyncTask,
) -> TgmError
where
    F: std::future::Future<Output = Result<TaskResult, TensogramError>> + Send + 'static,
{
    match spawn_task(fut, cancel, timeout_ms) {
        Ok(t) => {
            unsafe { *out_task = t };
            TgmError::Ok
        }
        Err(s) => {
            set_last_error(&s);
            TgmError::Io
        }
    }
}

fn build_tgm_message(
    gm: GlobalMetadata,
    objects: Vec<(DataObjectDescriptor, Vec<u8>)>,
) -> TgmMessage {
    let mut dtype_strings = Vec::with_capacity(objects.len());
    let mut type_strings = Vec::with_capacity(objects.len());
    let mut byte_order_strings = Vec::with_capacity(objects.len());
    let mut filter_strings = Vec::with_capacity(objects.len());
    let mut compression_strings = Vec::with_capacity(objects.len());
    let mut encoding_strings = Vec::with_capacity(objects.len());
    let hash_type_strings = vec![None; objects.len()];
    let hash_value_strings = vec![None; objects.len()];

    for (desc, _) in &objects {
        dtype_strings.push(CString::new(dtype_name(desc.dtype)).unwrap_or_default());
        type_strings.push(CString::new(desc.obj_type.as_str()).unwrap_or_default());
        byte_order_strings.push(
            CString::new(if desc.byte_order == tensogram::ByteOrder::Big {
                "big"
            } else {
                "little"
            })
            .unwrap_or_default(),
        );
        filter_strings.push(CString::new(desc.filter.as_str()).unwrap_or_default());
        compression_strings.push(CString::new(desc.compression.as_str()).unwrap_or_default());
        encoding_strings.push(CString::new(desc.encoding.as_str()).unwrap_or_default());
    }

    TgmMessage {
        global_metadata: gm,
        objects,
        dtype_strings,
        type_strings,
        byte_order_strings,
        filter_strings,
        compression_strings,
        encoding_strings,
        hash_type_strings,
        hash_value_strings,
    }
}

fn dtype_name(d: tensogram::dtype::Dtype) -> &'static str {
    use tensogram::dtype::Dtype;
    match d {
        Dtype::Float16 => "float16",
        Dtype::Bfloat16 => "bfloat16",
        Dtype::Float32 => "float32",
        Dtype::Float64 => "float64",
        Dtype::Complex64 => "complex64",
        Dtype::Complex128 => "complex128",
        Dtype::Int8 => "int8",
        Dtype::Int16 => "int16",
        Dtype::Int32 => "int32",
        Dtype::Int64 => "int64",
        Dtype::Uint8 => "uint8",
        Dtype::Uint16 => "uint16",
        Dtype::Uint32 => "uint32",
        Dtype::Uint64 => "uint64",
        Dtype::Bitmask => "bitmask",
    }
}

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
//! reads to C and C++ callers.
//!
//! The runtime is fully contained: no tokio handle, executor, or any
//! tokio type ever appears in the public C API.  A process-global
//! tokio runtime is built lazily on first call and can be configured
//! once via [`tgm_runtime_configure`].

#![cfg(feature = "async")]

use std::ffi::{CStr, CString, c_void};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use tensogram::{
    DataObjectDescriptor, DecodeOptions, GlobalMetadata, MessageLayout, TensogramError,
    TensogramFile,
};

/// Async-task outcome.  Distinguishes timeout / cancellation /
/// inner-error structurally so the FFI bridge can map them to the
/// matching [`TgmError`] codes without inspecting message strings
/// (the previous approach was brittle: any unrelated `Encoding` error
/// whose message happened to match would be misclassified).
pub(crate) enum AsyncOutcome {
    Ok(TaskResult),
    Timeout,
    Cancelled,
    Inner(TensogramError),
}

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
static DISPATCHER: OnceLock<DispatcherPool> = OnceLock::new();

/// Lifecycle state of the process-global tokio runtime.
///
/// The runtime is single-shot: built lazily on first use, and once
/// [`tgm_runtime_shutdown_blocking`] runs it transitions to
/// [`RuntimeState::ShutDown`] permanently — there is no rebuild.  A
/// build failure is cached so every subsequent call surfaces the same
/// diagnostic rather than retrying a doomed build.
///
/// `shutdown_timeout` consumes the owned [`tokio::runtime::Runtime`],
/// which is why this lives behind a `Mutex<…>` we can `take` from,
/// rather than the previous `OnceLock` (which can never hand `self`
/// back).
enum RuntimeState {
    /// No runtime built yet.
    Uninit,
    /// Runtime built and live.
    Built(tokio::runtime::Runtime),
    /// A previous build attempt failed; the error is cached and
    /// re-reported on every call.
    BuildFailed(String),
    /// The runtime has been shut down.  All async entry points fail
    /// fast from here on.
    ShutDown,
}

static RUNTIME: Mutex<RuntimeState> = Mutex::new(RuntimeState::Uninit);

/// Count of tasks spawned on the shared runtime that have not yet run
/// their completion path.  Incremented at the single spawn choke
/// point ([`spawn_task`]) and decremented by a drop-guard inside the
/// spawned future, so it covers normal completion, inner error,
/// cancellation, timeout, and a dropped/aborted task uniformly.
/// Read by [`tgm_runtime_shutdown_blocking`] to report how many tasks
/// did not drain within the timeout.
static LIVE_TASKS: AtomicUsize = AtomicUsize::new(0);

/// RAII guard pairing one [`LIVE_TASKS`] increment with exactly one
/// decrement.  [`new`](Self::new) increments on construction; `Drop`
/// decrements when the guard is dropped — whichever branch the owning
/// future exits through, **including** being dropped before its first
/// poll (e.g. when the runtime is torn down while the task is still
/// queued).  Encapsulating both halves in one type makes it impossible
/// to increment without arranging the matching decrement.
struct LiveTaskGuard;

impl LiveTaskGuard {
    fn new() -> Self {
        LIVE_TASKS.fetch_add(1, Ordering::SeqCst);
        LiveTaskGuard
    }
}

impl Drop for LiveTaskGuard {
    fn drop(&mut self) {
        LIVE_TASKS.fetch_sub(1, Ordering::SeqCst);
    }
}

fn runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get_or_init(RuntimeConfig::default)
}

// ---------------------------------------------------------------------------
// Dispatcher pool — runs user completion callbacks off the tokio runtime.
//
// Per the callback contract, completion
// callbacks must NOT execute on a tokio worker thread directly: a
// blocking or slow callback would stall the runtime.  This pool sits
// between `complete()` (the tokio resolver) and the user callback,
// running each callback on a dedicated dispatcher thread.
//
// Sized via [`RuntimeConfig::dispatcher_workers`].  A single
// `mpsc::SyncSender` feeds N parking workers; jobs are FIFO.  When
// the channel sender is dropped at process exit the workers exit
// cleanly.
// ---------------------------------------------------------------------------

struct DispatcherJob {
    cb: extern "C" fn(*mut c_void),
    /// Stored as `usize` because raw pointers aren't `Send`.  The
    /// integer is round-tripped to the original pointer by the worker.
    /// SAFETY: the user's callback contract requires `userdata` to be
    /// either NULL or to point to memory the user keeps alive until
    /// the callback fires.
    userdata: usize,
}

// SAFETY: see field comment above.
unsafe impl Send for DispatcherJob {}

struct DispatcherPool {
    sender: std::sync::mpsc::SyncSender<DispatcherJob>,
    _workers: Vec<std::thread::JoinHandle<()>>,
}

fn dispatcher_pool() -> &'static DispatcherPool {
    DISPATCHER.get_or_init(|| {
        let workers = runtime_config().dispatcher_workers.max(1) as usize;
        // Bounded channel so a runaway producer can't grow the queue
        // without bound; the bound is generous (4× workers) to avoid
        // backpressure stalls in the common case.
        let (tx, rx) = std::sync::mpsc::sync_channel::<DispatcherJob>(workers * 4);
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let rx = std::sync::Arc::clone(&rx);
            let h = std::thread::Builder::new()
                .name(format!("tensogram-dispatch-{i}"))
                .spawn(move || {
                    loop {
                        // Lock the receiver, recv, drop the lock before
                        // running the callback so other workers can
                        // pull jobs while this one is busy.
                        let job = {
                            let guard = rx.lock().expect("dispatcher rx poisoned");
                            match guard.recv() {
                                Ok(j) => j,
                                Err(_) => break, // sender dropped → shutdown
                            }
                        };
                        (job.cb)(job.userdata as *mut c_void);
                    }
                })
                .expect("dispatcher worker spawn");
            handles.push(h);
        }
        DispatcherPool {
            sender: tx,
            _workers: handles,
        }
    })
}

/// Enqueue a user callback for execution on the dispatcher pool.
///
/// Falls back to inline execution on the calling thread only if the
/// pool's bounded channel is full (extreme overload — the runtime is
/// producing completions faster than the workers can drain them).
/// In practice this should never fire under normal load.
fn dispatch_to_pool(cb: extern "C" fn(*mut c_void), userdata: *mut c_void) {
    let job = DispatcherJob {
        cb,
        userdata: userdata as usize,
    };
    if let Err(std::sync::mpsc::TrySendError::Full(job)) = dispatcher_pool().sender.try_send(job) {
        // Channel full: dispatch on the calling thread.  This is a
        // graceful-degradation path; users who hit it should bump
        // `dispatcher_workers` via `tgm_runtime_configure`.
        (job.cb)(job.userdata as *mut c_void);
    }
}

/// Resolve the shared runtime to a spawnable [`tokio::runtime::Handle`].
///
/// Builds the runtime lazily on first call.  The returned `Handle` is
/// a cheap clone of the runtime's handle, so callers spawn through it
/// **without** holding the runtime mutex across the spawn.  The mutex
/// is held only briefly to read/transition the [`RuntimeState`].
///
/// Errors:
///   * `Err("…")` with the cached build error if the runtime failed to
///     build (sticky — same message every call).
///   * `Err("runtime has been shut down")` once
///     [`tgm_runtime_shutdown_blocking`] has run — the runtime is
///     single-shot and never rebuilt.
fn runtime() -> Result<tokio::runtime::Handle, String> {
    let mut guard = RUNTIME.lock().expect("runtime mutex poisoned");
    match &*guard {
        RuntimeState::Built(rt) => Ok(rt.handle().clone()),
        RuntimeState::BuildFailed(e) => Err(e.clone()),
        RuntimeState::ShutDown => Err("runtime has been shut down".to_string()),
        RuntimeState::Uninit => {
            let cfg = runtime_config();
            match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(cfg.workers as usize)
                .enable_all()
                .thread_name("tensogram-async")
                .build()
            {
                Ok(rt) => {
                    let handle = rt.handle().clone();
                    *guard = RuntimeState::Built(rt);
                    Ok(handle)
                }
                Err(e) => {
                    let msg = format!("failed to build tokio runtime: {e}");
                    *guard = RuntimeState::BuildFailed(msg.clone());
                    Err(msg)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Task lifecycle
// ---------------------------------------------------------------------------

/// Result variants surfaced by an async task.  The variant must match
/// the join function the caller invokes; mismatches surface as
/// [`TgmError::InvalidArg`].
///
/// Internal to `tensogram-ffi`: the C ABI never touches this type
/// directly — every `tgm_async_task_join_*` function consumes one
/// specific variant and returns the typed payload.  Sibling modules
/// (`async_streaming`) construct variants directly.
#[allow(dead_code)] // some variants reserved for follow-up PRs
pub(crate) enum TaskResult {
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
    result: Option<AsyncOutcome>,
    completion_cb: Option<extern "C" fn(*mut c_void)>,
    /// Becomes `true` once `set_completion` has been called, even if
    /// the registered callback has already fired (and `completion_cb`
    /// is therefore `None`).  Enforces the documented exactly-once
    /// contract regardless of task state.
    completion_registered: bool,
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
    /// Spawn-loop-side clone of the external token, if any.  Held so
    /// the spawn loop can `select!` against it without a separate
    /// forwarder task (the previous forwarder leaked one tokio task
    /// per cancellable spawn whenever the external token was never
    /// cancelled, which is the common case).
    ///
    /// `tokio_util::sync::CancellationToken` is internally `Arc`-backed,
    /// so this clone keeps the cancellation machinery alive
    /// independently of the C-side `tgm_cancellation_token_t` handle's
    /// `Arc<TgmCancellationTokenInner>` — we don't need a separate
    /// `Arc` field for that.
    external: Option<tokio_util::sync::CancellationToken>,
}

impl TaskShared {
    fn new(external: Option<&TgmCancellationToken>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(TaskInner {
                state: TaskState::Pending,
                result: None,
                completion_cb: None,
                completion_registered: false,
                completion_userdata: std::ptr::null_mut(),
            }),
            ready: Condvar::new(),
            cancel: tokio_util::sync::CancellationToken::new(),
            external: external.map(|t| t.inner.token.clone()),
        })
    }

    /// Mark the task as Ready and dispatch the completion callback if
    /// one was registered.  Called once per task by the runtime.
    ///
    /// The callback is enqueued on the dispatcher pool so it runs on
    /// a non-tokio thread (per the §4.1 callback contract); a slow or
    /// blocking user callback therefore cannot stall the runtime.
    fn complete(&self, result: AsyncOutcome) {
        let cb_to_fire = {
            let mut inner = self.state.lock().expect("task mutex poisoned");
            if inner.state != TaskState::Pending {
                // Already consumed / never armed.  Drop the result.
                return;
            }
            inner.result = Some(result);
            inner.state = TaskState::Ready;
            // Snapshot the callback so we can release the lock before
            // dispatching.  The dispatch path itself does not call any
            // user code while the task lock is held.
            inner.completion_cb.take().map(|cb| {
                (
                    cb,
                    std::mem::replace(&mut inner.completion_userdata, std::ptr::null_mut()),
                )
            })
        };
        self.ready.notify_all();
        if let Some((cb, userdata)) = cb_to_fire {
            dispatch_to_pool(cb, userdata);
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
    let internal_token = shared.cancel.clone();
    // Take a direct clone of the external token (if any) into the
    // spawn loop.  Joining both tokens directly in this `select!`
    // eliminates the need for a separate forwarder task that would
    // otherwise leak one tokio task per spawn whenever the external
    // token was never cancelled (the common case).
    let external_token = shared.external.clone();

    // Count this task as live before it is spawned.  The guard is
    // constructed *here*, outside the async block, and moved into the
    // future so it is owned by the future itself — not merely created
    // on first poll.  This decrements `LIVE_TASKS` on every exit path
    // (completion, inner error, cancel, timeout) AND when the future
    // is dropped before it is ever polled (e.g. the runtime is torn
    // down with the task still queued).  Constructing it inside the
    // async block would skip the decrement in that drop-before-poll
    // case, permanently inflating the counter.  `LiveTaskGuard::new`
    // does the `fetch_add`; its `Drop` does the matching `fetch_sub`.
    let live_guard = LiveTaskGuard::new();

    rt.spawn(async move {
        // Bind the guard into the future so its lifetime is the
        // future's lifetime.  `let _live = live_guard;` rather than a
        // bare move expression so the binding is unambiguous.
        let _live = live_guard;
        let outcome: AsyncOutcome = if timeout_ms > 0 {
            let deadline = Duration::from_millis(timeout_ms);
            tokio::select! {
                _ = internal_token.cancelled() => AsyncOutcome::Cancelled,
                _ = async { match external_token { Some(t) => t.cancelled().await, None => std::future::pending().await } } => AsyncOutcome::Cancelled,
                res = tokio::time::timeout(deadline, fut) => match res {
                    Ok(Ok(v)) => AsyncOutcome::Ok(v),
                    Ok(Err(e)) => AsyncOutcome::Inner(e),
                    Err(_elapsed) => AsyncOutcome::Timeout,
                },
            }
        } else {
            tokio::select! {
                _ = internal_token.cancelled() => AsyncOutcome::Cancelled,
                _ = async { match external_token { Some(t) => t.cancelled().await, None => std::future::pending().await } } => AsyncOutcome::Cancelled,
                res = fut => match res {
                    Ok(v) => AsyncOutcome::Ok(v),
                    Err(e) => AsyncOutcome::Inner(e),
                },
            }
        };
        shared_clone.complete(outcome);
    });

    Ok(Box::into_raw(Box::new(TgmAsyncTask { inner: shared })))
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

/// Allocate a fresh cancellation token in the un-cancelled state.
///
/// The returned handle is caller-owned; free it with
/// `tgm_cancellation_token_free`.  A single token may be passed to
/// any number of `tgm_async_*` calls; cancelling it propagates to
/// all in-flight tasks holding a reference.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_create() -> *mut TgmCancellationToken {
    Box::into_raw(Box::new(TgmCancellationToken {
        inner: Arc::new(TgmCancellationTokenInner {
            token: tokio_util::sync::CancellationToken::new(),
        }),
    }))
}

/// Cancel the token.  Idempotent.  Cancelling a NULL handle is a
/// no-op.  Tasks attached to this token transition to
/// [`TgmError::Cancelled`] at their next yield point.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_cancel(tok: *mut TgmCancellationToken) {
    if tok.is_null() {
        return;
    }
    let t = unsafe { &*tok };
    t.inner.token.cancel();
}

/// Returns `true` if the token has been cancelled.  A NULL handle
/// returns `false`.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_cancellation_token_is_cancelled(tok: *const TgmCancellationToken) -> bool {
    if tok.is_null() {
        return false;
    }
    let t = unsafe { &*tok };
    t.inner.token.is_cancelled()
}

/// Free a cancellation token.  Safe to call before any in-flight
/// task completes — each task holds its own internal clone of the
/// underlying tokio token, so freeing the user-side handle does not
/// invalidate cancellation for already-spawned tasks.  Freeing a
/// NULL handle is a no-op.
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
/// once per task; subsequent calls return [`TgmError::InvalidArg`]
/// regardless of whether the task is still pending or already
/// resolved.
///
/// The callback fires on a **dispatcher pool worker** (a non-tokio
/// thread owned by `tensogram-ffi`), so a slow or blocking callback
/// does not stall the runtime.  The callback contract: callbacks
/// must complete quickly, must not
/// throw, and must not block on locks held by the caller of any
/// other tgm_* function on this thread.
///
/// If the task has already resolved when this function is called,
/// the callback is enqueued on the dispatcher pool immediately so
/// the worker runs the user code on a dispatcher thread.  Under
/// extreme overload (the dispatcher's bounded queue is full) the
/// callback may run inline on the caller's thread as a graceful-
/// degradation path; bump `dispatcher_workers` via
/// `tgm_runtime_configure` to avoid this.
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
    // only across the inspect-and-store step.  Exactly-once is enforced
    // by the persistent `completion_registered` flag — `completion_cb`
    // alone isn't enough because it gets `take()`n out by `complete()`
    // when the callback fires (so a Ready-state callsite would see
    // `None` and incorrectly accept a second registration).
    let fire_inline = {
        let mut inner = t.inner.state.lock().expect("task mutex poisoned");
        if inner.completion_registered {
            set_last_error("completion callback already registered");
            return TgmError::InvalidArg;
        }
        inner.completion_registered = true;
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
        dispatch_to_pool(cb, userdata);
    }
    TgmError::Ok
}

/// Returns `true` if the task has resolved (whether successfully,
/// by error, by timeout, or by cancellation).  A NULL handle returns
/// `false`.  Non-blocking; safe to call from any thread.
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
/// A NULL handle is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_task_cancel(task: *mut TgmAsyncTask) {
    if task.is_null() {
        return;
    }
    let t = unsafe { &*task };
    t.inner.cancel.cancel();
}

/// Free a task handle.  Must be called exactly once per task to
/// release the slot's heap allocation.  Safe to call after
/// [`tgm_async_task_join`]: the join consumes the result but does
/// not free the handle itself.  Freeing a NULL handle is a no-op.
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
    match res {
        AsyncOutcome::Ok(v) => Ok(v),
        AsyncOutcome::Timeout => {
            set_last_error("async task timed out");
            Err(TgmError::Timeout)
        }
        AsyncOutcome::Cancelled => {
            set_last_error("async task cancelled");
            Err(TgmError::Cancelled)
        }
        AsyncOutcome::Inner(e) => {
            set_last_error(&e.to_string());
            Err(to_error_code(&e))
        }
    }
}

/// Block the calling thread until the task is ready, then return
/// its outcome.
///
/// `_join_void` is for tasks whose success is ABI-empty (writes,
/// finishes, prefetches).  Returns [`TgmError::Ok`] on success.  On
/// failure, the matching error code is returned and
/// [`tgm_last_error`] is populated.
///
/// Call [`tgm_async_task_free`] to release the task slot after
/// joining.  Joining the same task twice returns
/// [`TgmError::InvalidArg`] (`task result already consumed`).
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

/// Block the calling thread until the task is ready, then write its
/// `usize` result to `*out` and return [`TgmError::Ok`].  See
/// [`tgm_async_task_join_void`] for the joint contract on errors,
/// type mismatches, and double-join.
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

/// Block until ready, then transfer ownership of the task's byte
/// buffer to the caller via `*out`.  The caller must release the
/// buffer with [`tgm_bytes_free`].  See [`tgm_async_task_join_void`]
/// for the joint contract.
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

/// Block until ready, then transfer ownership of the decoded message
/// to the caller via `*out`.  The caller must release the message
/// with [`tgm_message_free`].  See [`tgm_async_task_join_void`] for
/// the joint contract.
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

/// Block until ready, then transfer ownership of the decoded
/// metadata to the caller via `*out`.  The caller must release it
/// with [`tgm_metadata_free`].  See [`tgm_async_task_join_void`]
/// for the joint contract.
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

/// Block until ready, then transfer ownership of the opened async
/// file handle to the caller via `*out`.  The caller must release it
/// with [`tgm_async_file_close`].  See [`tgm_async_task_join_void`]
/// for the joint contract.
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

/// Block until ready, then transfer ownership of an array of byte
/// buffers (one per requested range) to the caller.  Writes the
/// array pointer to `*out_array` and the entry count to
/// `*out_count`.  The caller must release the array via
/// [`tgm_multi_bytes_free`].  See [`tgm_async_task_join_void`] for
/// the joint contract.
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

/// Borrowed pointer to the file's path string.  Valid until the
/// handle is closed.  Returns NULL on a NULL handle.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_async_file_path(file: *const TgmAsyncFile) -> *const c_char {
    if file.is_null() {
        return std::ptr::null();
    }
    unsafe { (*file).path_string.as_ptr() }
}

/// Close an async file handle.  Internally backed by
/// `Arc<TensogramFile>`, so any task currently using the handle
/// keeps the underlying file alive until the task completes.
/// Closing a NULL handle is a no-op.
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

/// Open a remote `.tgm` (S3 / GCS / Azure / HTTP).  Always exported
/// at the C ABI level so consumers linking the cdylib never see an
/// undefined symbol; when the `async-remote` Cargo feature is off
/// the function returns [`TgmError::Remote`] with a clear
/// `tgm_last_error()` message instead of dispatching.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
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
    #[cfg(not(feature = "async-remote"))]
    {
        let _ = (
            url,
            storage_keys,
            storage_values,
            nopts,
            bidirectional,
            cancel,
            timeout_ms,
            out_task,
        );
        set_last_error(
            "tgm_async_file_open_remote: this build of tensogram-ffi was compiled \
             without the `async-remote` Cargo feature; rebuild with --features=async-remote \
             to enable S3/GCS/Azure/HTTP support",
        );
        TgmError::Remote
    }

    #[cfg(feature = "async-remote")]
    {
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
            let f =
                TensogramFile::open_remote_async(&url_for_task, &storage, Some(scan_opts)).await?;
            let path_string = CString::new(url_for_task.as_str()).unwrap_or_default();
            Ok(TaskResult::AsyncFile(Box::new(TgmAsyncFile {
                file: Arc::new(f),
                path_string,
            })))
        };
        spawn_or_set_error(fut, cancel_ref, timeout_ms, out_task)
    }
}

// ---------------------------------------------------------------------------
// File-level async operations
// ---------------------------------------------------------------------------

/// Async equivalent of [`tgm_file_message_count`].  Returns
/// immediately with a task handle that resolves to the message count
/// (joinable via [`tgm_async_task_join_size`]).  See the type
/// docstrings for the cancellation/timeout/null-handle contract
/// shared by every `tgm_async_file_*` entry point.
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

/// Async equivalent of [`tgm_file_read_message`].  Resolves to the
/// raw message bytes (joinable via [`tgm_async_task_join_bytes`]).
/// See the type docstrings for the shared cancellation/timeout
/// contract.
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

/// Async equivalent of [`tgm_file_decode_message`].  Resolves to a
/// fully decoded [`TgmMessage`] (joinable via
/// [`tgm_async_task_join_message`]).  `threads = 0` selects the
/// `tensogram::DecodeOptions` default.  See the type docstrings for
/// the shared cancellation/timeout contract.
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

/// Async equivalent of [`tgm_file_decode_metadata`].  Resolves to a
/// [`TgmMetadata`] handle (joinable via
/// [`tgm_async_task_join_metadata`]) without materialising tensor
/// payloads.  See the type docstrings for the shared
/// cancellation/timeout contract.
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

/// Async equivalent of [`tgm_file_decode_object`].  Resolves to a
/// single-object [`TgmMessage`] (joinable via
/// [`tgm_async_task_join_message`]) so the existing `tgm_object_*`
/// and `tgm_message_*` accessors work uniformly.  See the type
/// docstrings for the shared cancellation/timeout contract.
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

/// Async equivalent of [`tgm_file_decode_range`].  Each `(offset,
/// count)` pair in `offsets[]` / `counts[]` (length `n_ranges`)
/// describes a sub-range of the object's logical element stream.
/// Resolves to a vector of byte buffers, one per range, joinable via
/// [`tgm_async_task_join_multi_bytes`].  See the type docstrings for
/// the shared cancellation/timeout contract.
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

/// Shut the shared async runtime down, blocking for up to
/// `timeout_ms` while in-flight tasks drain.
///
/// Returns the number of tasks that had **not** finished when the
/// timeout elapsed (`0` on a clean drain).  After this call the
/// runtime is permanently shut down: every subsequent `tgm_async_*`
/// entry point fails fast with `TGM_ERROR_IO` and a descriptive
/// `tgm_last_error()` (`"runtime has been shut down"`).  The runtime
/// is single-shot — there is no rebuild.
///
/// Behaviour by prior state:
///   * **never built** (no async op ran) → transitions straight to
///     `ShutDown` and returns `0`; nothing was ever spawned.
///   * **build previously failed** → transitions to `ShutDown` and
///     returns `0`; there was no runtime to drain.
///   * **already shut down** → idempotent no-op returning `0`.
///   * **live** → takes ownership of the runtime, drains in-flight
///     tasks up to `timeout_ms`, tears the runtime down, and reports
///     the count of tasks that had not finished by the deadline.
///
/// Must not be called from inside an async callback running on the
/// runtime's own worker threads (tokio forbids dropping a runtime
/// from within itself).  The intended caller is the application's
/// main/teardown thread.
#[unsafe(no_mangle)]
pub extern "C" fn tgm_runtime_shutdown_blocking(timeout_ms: u64) -> u64 {
    // Take the runtime out under the lock and mark the singleton
    // shut down, then release the lock *before* the (potentially
    // long) blocking drain so concurrent `tgm_async_*` callers see
    // `ShutDown` immediately and fail fast rather than blocking on
    // the mutex.
    let taken = {
        let mut guard = RUNTIME.lock().expect("runtime mutex poisoned");
        match std::mem::replace(&mut *guard, RuntimeState::ShutDown) {
            RuntimeState::Built(rt) => Some(rt),
            // Uninit / BuildFailed / ShutDown: nothing to drain.
            _ => None,
        }
    };

    let Some(rt) = taken else {
        return 0;
    };

    // Cooperative drain.  Poll `LIVE_TASKS` until it reaches zero or
    // the deadline elapses, then capture the residual *before* the
    // forced teardown below.  Reading the count here — rather than
    // after `shutdown_timeout` — is what makes the result accurate:
    // dropping the runtime drops every still-pending future, which
    // runs each `LiveTaskGuard` and would otherwise drive the count
    // to zero regardless of whether those tasks actually finished.
    // Cooperative drain via elapsed-time comparison rather than an
    // absolute `Instant` deadline.  `Instant + Duration` panics on
    // overflow, and this is a non-panicking FFI entry point, so we
    // never add a caller-controlled `Duration` to an `Instant`.
    // Instead we measure `start.elapsed()` (always well-defined) and
    // compare against the requested timeout — no arithmetic that can
    // overflow on a hostile `timeout_ms`.
    let start = std::time::Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let poll_interval = Duration::from_millis(5);
    loop {
        if LIVE_TASKS.load(Ordering::SeqCst) == 0 {
            break;
        }
        let elapsed = start.elapsed();
        if elapsed >= timeout {
            break;
        }
        // Sleep the shorter of the poll interval and the remaining
        // budget so we do not overshoot the deadline.
        std::thread::sleep(poll_interval.min(timeout - elapsed));
    }
    let unfinished = LIVE_TASKS.load(Ordering::SeqCst) as u64;

    // Force the runtime down now.  Any tasks still in flight are
    // abandoned; we already captured their count above.  `ZERO`
    // because the cooperative wait already happened — this call is
    // just the teardown.
    rt.shutdown_timeout(Duration::ZERO);

    unfinished
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Full single-shot shutdown lifecycle, in one test because the
    /// runtime is process-global and shutdown is irreversible — a
    /// second test calling shutdown would observe an already-torn-down
    /// runtime.  Determinism comes from spawning a task that sleeps far
    /// longer than the shutdown timeout, so it is guaranteed to still
    /// be in flight when we drain.
    ///
    /// Asserts:
    ///   1. a task in flight past the drain deadline is reported as
    ///      unfinished (count ≥ 1);
    ///   2. after shutdown the runtime is `ShutDown` and `spawn_task`
    ///      fails with the documented diagnostic;
    ///   3. a second shutdown is an idempotent no-op returning 0.
    #[test]
    fn shutdown_blocking_drains_and_reports_unfinished() {
        // Spawn a task that outlives any reasonable drain window.
        let task = spawn_task(
            async {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(TaskResult::Void)
            },
            None,
            0,
        )
        .expect("spawn before shutdown should succeed");

        // Drain for far less than the task's sleep, so it is still
        // in flight at the deadline and must be counted.
        let unfinished = tgm_runtime_shutdown_blocking(50);
        assert!(
            unfinished >= 1,
            "a task sleeping for an hour must be reported unfinished after a 50 ms drain, got {unfinished}"
        );

        // The runtime is now single-shot: further spawns fail with the
        // documented diagnostic (which the FFI maps to TGM_ERROR_IO).
        let err = spawn_task(async { Ok(TaskResult::Void) }, None, 0)
            .expect_err("spawn after shutdown must fail");
        assert!(
            err.contains("shut down"),
            "post-shutdown spawn error should mention shutdown, got: {err}"
        );

        // Second shutdown is an idempotent no-op.
        assert_eq!(
            tgm_runtime_shutdown_blocking(10),
            0,
            "second shutdown must be a no-op returning 0"
        );

        // The leaked task handle is intentionally not freed/joined: the
        // runtime that owned its future is gone, so joining would block
        // forever.  Leaking one Arc in a unit test is acceptable.
        let _ = task;
    }
}

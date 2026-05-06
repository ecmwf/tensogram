# Plan: Asynchronous C++ API for Tensogram

> **Status.** Accepted. All design questions resolved; ready for implementation.
>
> **Scope.** Add a true asynchronous read/write surface to the C++
> wrapper for Tensogram, sharing Tensogram's internal tokio runtime
> with no externally-visible Rust dependency.  Driven by an HPC
> streaming producer/consumer scenario where two independent jobs on
> the same cluster pipe data through a `.tgm` artefact.
>
> **Non-goals.**  External tokio interop (users supplying their own
> runtime), GPU-direct paths, Windows MSVC, observability/tracing
> hooks (deferred), batched-sync intermediate APIs (we go straight to
> true async).

---

## 1. Driver scenario

Two HPC jobs on the same cluster:

- **Producer**: a simulation/inference job emits forecast steps as
  they are produced.  Each step is a Tensogram message.  The producer
  must not stall on the consumer.
- **Consumer**: a downstream post-processing or visualisation job
  reads each message as soon as it lands.  Must not stall on the
  producer either — slow steps shouldn't propagate.

Both run as standard C++ binaries.  Both need the async surface
because:

- Producer wants `streaming_encoder` writes to a shared filesystem or
  S3-like store to overlap with the next step's compute.
- Consumer wants per-message decode to overlap with the next message's
  fetch.

The shared-filesystem path has byte-range support (Lustre, GPFS,
WekaFS, BeeGFS — all handle range reads natively); the object-store
path is `s3://`, `gs://`, `az://` via the existing `object_store`
backend.

---

## 2. High-level architecture

Three-layer design:

```
┌────────────────────────────────────────────────────────────────────┐
│ User code (C++17 or C++20)                                          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ #include <tensogram/async/callback.hpp>
                              │   or coro.hpp / std_future.hpp
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ C++ frontend headers (header-only, opt-in)                          │
│  • async/callback.hpp    callback-based, C++17, always available    │
│  • async/coro.hpp        C++20 task<T> coroutines (opt-in)          │
│  • async/std_future.hpp  std::future<T> wrappers (opt-in)           │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ all frontends call the same FFI
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ tensogram-ffi async core (new)                                      │
│  Opaque tgm_async_task_t handles + completion-callback FFI          │
│  Cancellation tokens, deadlines, ABI-stable task lifecycle          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ Rust async fn on TensogramFile / RemoteBackend
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ tensogram (Rust core)                                               │
│  Existing async fns: open_async / decode_*_async / read_message_… │
│  SHARED_RUNTIME (tokio multi-thread, contained — no leak to caller) │
└────────────────────────────────────────────────────────────────────┘
```

**Key invariant.** Every C++ frontend boils down to the same
callback-based FFI core.  We never duplicate the "async work
machinery" — it lives once at the FFI boundary, and every frontend is
a thin syntactic adapter.  Three frontends ship in v1; further
frontends (e.g. external-runtime adapters) can be added later as
header-only files without re-architecture.

---

## 3. FFI core API (new)

The FFI layer grows a new task abstraction.  Tasks are opaque,
heap-allocated handles with three states (`Pending`, `Ready`,
`Cancelled`) and a single completion callback.

### 3.1 Task handle and lifecycle

```c
// tensogram.h additions

typedef struct tgm_async_task tgm_async_task_t;
typedef struct tgm_cancellation_token tgm_cancellation_token_t;

/* Completion callback. Invoked exactly once per task, on the tokio
 * runtime thread that resolves the underlying future. */
typedef void (*tgm_completion_cb)(void *userdata);

/* Register the completion callback. Must be called exactly once,
 * before any tgm_async_task_join_*. The callback fires either:
 *   - immediately and inline, if the task is already Ready/Cancelled
 *   - asynchronously, when the task transitions to Ready/Cancelled
 *
 * The callback contract is: do as little as possible; signal a
 * condition variable / promise / coroutine handle and return. */
void tgm_async_task_set_completion(tgm_async_task_t *task,
                                   tgm_completion_cb cb,
                                   void *userdata);

/* Polling without blocking. */
bool tgm_async_task_is_ready(const tgm_async_task_t *task);

/* Result extraction.  Each result type has a typed join function;
 * the FFI knows the static type of every task at creation time, so
 * a wrong call returns TGM_ERROR_INVALID_ARG without UB.  Each join
 * blocks the calling thread until completion (use
 * tgm_async_task_set_completion + a condvar / promise / coroutine
 * for non-blocking). */
tgm_error tgm_async_task_join_message(tgm_async_task_t *task,
                                      tgm_message_t **out);
tgm_error tgm_async_task_join_metadata(tgm_async_task_t *task,
                                       tgm_metadata_t **out);
tgm_error tgm_async_task_join_bytes(tgm_async_task_t *task,
                                    tgm_bytes_t *out);
tgm_error tgm_async_task_join_size(tgm_async_task_t *task,
                                   size_t *out);
tgm_error tgm_async_task_join_void(tgm_async_task_t *task);
tgm_error tgm_async_task_join_message_batch(tgm_async_task_t *task,
                                            tgm_message_t ***out_messages,
                                            size_t *out_count);
tgm_error tgm_async_task_join_bytes_batch(tgm_async_task_t *task,
                                          tgm_bytes_t **out_bytes,
                                          size_t *out_count);

/* Releases task storage. Must always be called, even after a
 * successful join (joins free the result, not the task itself).
 * Idempotent: free of a NULL pointer is a no-op. */
void tgm_async_task_free(tgm_async_task_t *task);
```

### 3.2 Cancellation tokens

```c
tgm_cancellation_token_t *tgm_cancellation_token_create(void);
void tgm_cancellation_token_cancel(tgm_cancellation_token_t *tok);
bool tgm_cancellation_token_is_cancelled(const tgm_cancellation_token_t *tok);
void tgm_cancellation_token_free(tgm_cancellation_token_t *tok);
```

A token may be passed to any async-launching FFI call.  The token is
**not consumed** — it can be shared across many tasks (a typical
"cancel all my pending work" pattern).  Tokens are ref-counted
internally so freeing the token while tasks still reference it is
safe.

### 3.3 Async file open / read / decode

```c
/* Async file opens. Path may be a local path, s3://, gs://, az://,
 * https://. The resulting tgm_async_file_t* is internally backed by
 * Arc<TensogramFile> so it is freely shareable across threads. */
tgm_async_task_t *tgm_async_file_open(
    const char *path,
    tgm_cancellation_token_t *token,  /* nullable */
    uint64_t timeout_ms);              /* 0 = no timeout */

tgm_async_task_t *tgm_async_file_open_remote(
    const char *url,
    const char **storage_keys, const char **storage_values, size_t nopts,
    bool bidirectional,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

/* tgm_async_file_t* result type joined via a new typed join. */
tgm_error tgm_async_task_join_file(tgm_async_task_t *task,
                                   tgm_async_file_t **out);

/* Per-method async entry points. Same semantics as their sync
 * counterparts but return immediately with a task handle. */
tgm_async_task_t *tgm_async_file_message_count(
    tgm_async_file_t *file,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_read_message(
    tgm_async_file_t *file, size_t index,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_decode_message(
    tgm_async_file_t *file, size_t index,
    bool native_byte_order, uint32_t threads,
    bool restore_non_finite, bool verify_hash,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_decode_metadata(
    tgm_async_file_t *file, size_t index,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_decode_descriptors(
    tgm_async_file_t *file, size_t index,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_decode_object(
    tgm_async_file_t *file, size_t msg_idx, size_t obj_idx,
    bool native_byte_order, uint32_t threads,
    bool restore_non_finite, bool verify_hash,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_decode_range(
    tgm_async_file_t *file, size_t msg_idx, size_t obj_idx,
    const uint64_t *offsets, const uint64_t *counts, size_t nranges,
    bool joined,
    bool native_byte_order, uint32_t threads,
    bool restore_non_finite, bool verify_hash,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

/* Batch operations: fan out via tokio::join! / get_ranges
 * internally. The caller's task waits for ALL inner futures. */
tgm_async_task_t *tgm_async_file_decode_object_batch(
    tgm_async_file_t *file,
    const size_t *msg_indices, const size_t *obj_indices, size_t count,
    /* … decode opts … */,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_decode_range_batch(
    tgm_async_file_t *file,
    const tgm_range_request *requests, size_t count,
    /* … decode opts … */,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_file_prefetch_layouts(
    tgm_async_file_t *file,
    const size_t *indices, size_t count,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

void tgm_async_file_close(tgm_async_file_t *file);
```

### 3.4 Async streaming encoder

The producer scenario.  Streaming writes that overlap with the next
step's compute.

```c
typedef struct tgm_async_streaming_encoder
                tgm_async_streaming_encoder_t;

/* Local file or remote object-store URL.  Both supported via the
 * tokio AsyncWrite + object_store async write paths. */
tgm_async_task_t *tgm_async_streaming_encoder_create(
    const char *path_or_url,
    const char *metadata_json,
    /* … encode opts including hashing, threads, allow_nan, … … */,
    const char **storage_keys, const char **storage_values, size_t nopts,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_error tgm_async_task_join_streaming_encoder(
    tgm_async_task_t *task,
    tgm_async_streaming_encoder_t **out);

tgm_async_task_t *tgm_async_streaming_encoder_write(
    tgm_async_streaming_encoder_t *enc,
    const char *descriptor_json,
    const uint8_t *data, size_t len,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_streaming_encoder_write_preceder(
    tgm_async_streaming_encoder_t *enc,
    const char *metadata_json,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_streaming_encoder_write_pre_encoded(
    tgm_async_streaming_encoder_t *enc,
    const char *descriptor_json,
    const uint8_t *data, size_t len,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

tgm_async_task_t *tgm_async_streaming_encoder_finish(
    tgm_async_streaming_encoder_t *enc,
    bool backfill_total_length,
    tgm_cancellation_token_t *token,
    uint64_t timeout_ms);

void tgm_async_streaming_encoder_free(tgm_async_streaming_encoder_t *enc);
```

The Rust side needs new wrappers.  The existing
`StreamingEncoder<W: Write>` is sync only; we add
`AsyncStreamingEncoder<W: AsyncWrite + Unpin>` that drives the same
frame layout but writes via tokio's `AsyncWriteExt` trait.  For
`object_store` URLs the writer is `object_store`'s
`MultipartUpload`/`Writer` trait.  Implementation detail captured in §9.

### 3.5 Task error semantics

Every `_join_*` returns the `tgm_error` of the underlying operation.
Two new error codes:

```c
typedef enum tgm_error {
    /* … existing codes … */
    TGM_ERROR_TIMEOUT     = 12,  /* deadline elapsed before completion */
    TGM_ERROR_CANCELLED   = 13,  /* token was cancelled */
} tgm_error;
```

A task hits `TIMEOUT` exactly when the wall-clock since launch exceeds
`timeout_ms`.  Internally implemented as
`tokio::time::timeout(deadline, future)`.  A task hits `CANCELLED` when
its `tgm_cancellation_token_t` is signalled — the underlying tokio
future is dropped at the next yield point.

Both states are terminal; the task transitions to the C++-visible
`Ready` state with the appropriate error code on join.

### 3.6 Task lifecycle and the `set_completion` race

`tgm_async_task_t` holds a small atomic state machine:

```text
Pending  ─── future resolves ───────►  ResolvedNoCallback
   │                                          │
   │ set_completion(cb)                       │ set_completion(cb)
   ▼                                          ▼
Pending+Cb  ─── future resolves ───►  ResolvedCallbackFired
```

- `set_completion(cb)` uses a single CAS on the state word to install
  the callback.  The CAS arms one of two outcomes:
  - State was `Pending` → store `(Pending+Cb, cb_ptr)`.  When the
    future resolves, the resolver atomically transitions to
    `ResolvedCallbackFired` and dispatches `cb` on the dispatcher
    pool.
  - State was `ResolvedNoCallback` → atomically transition to
    `ResolvedCallbackFired` and dispatch `cb` inline (still on the
    caller's thread).
- A second `set_completion` call on the same task observes the
  callback already installed and returns `TGM_ERROR_INVALID_ARG`.
- A `join_*` after `set_completion` is legal: the join blocks until
  the state is `ResolvedCallbackFired`, then consumes the result.
  The result is owned by exactly one of {callback, joiner}; the
  encoder dictates which by configuration (default: result is
  delivered to the callback if registered, else available via join).
- A `join_*` without `set_completion` blocks the calling thread on
  a condition variable signalled by the runtime.

Each task holds a strong refcount on the cancellation token and a
strong refcount on the parent file/encoder so the underlying handles
cannot be freed while a task is in flight.  Pinned in
`test_async_lifecycle.cpp` and `test_async_double_completion.cpp`.

#### Result ownership across callback/join boundary

If a callback is registered, the result is delivered to the
callback's `result<T>` argument and **consumed** there.  A
subsequent `tgm_async_task_join_*` returns `TGM_ERROR_INVALID_ARG`
("result already consumed by callback").  Callers who want both
must drain the result from the callback into a `std::promise<T>`
and call `.get()` on the future — that's what the `std_future`
frontend does internally.

---

## 4. C++ Frontends

### 4.1 `async/callback.hpp` — callback frontend (C++17, always available)

The minimum-viable C++ surface.  All other frontends are layered on
top of this one.

```cpp
namespace tensogram::async_callback {

// Opaque RAII handles
class async_file;
class async_streaming_encoder;
class cancellation_token;

// tl::expected-style; never throws (so it composes with -fno-exceptions builds)
template <typename T>
class result {
public:
    bool ok() const noexcept;
    T& value();              // throws on !ok if exceptions enabled
    const error_info& error() const noexcept;
};

struct error_info {
    tgm_error code;
    std::string message;
};

// Completion handler signature
template <typename T>
using completion_handler = std::function<void(result<T>)>;

class async_file {
public:
    // Static factory functions are themselves async — they return
    // immediately and invoke cb when the file is opened.
    static void open(const std::string& path,
                     completion_handler<async_file> cb,
                     cancellation_token* token = nullptr,
                     std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());

    static void open_remote(const std::string& url,
                            const storage_options& opts,
                            completion_handler<async_file> cb,
                            cancellation_token* token = nullptr,
                            std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());

    void message_count(completion_handler<std::size_t> cb,
                       cancellation_token* token = nullptr,
                       std::chrono::milliseconds timeout = {});

    void decode_message(std::size_t index,
                        decode_options opts,
                        completion_handler<message> cb,
                        cancellation_token* token = nullptr,
                        std::chrono::milliseconds timeout = {});
    // … one per FFI entry point …
};

class cancellation_token {
public:
    cancellation_token();
    void cancel() noexcept;
    bool cancelled() const noexcept;
    // Non-copyable, move-only
};

}  // namespace tensogram::async_callback
```

This frontend is **always available** even if no other frontend is
included.

#### Callback contract (concrete bounds)

The callback executes on the FFI's **dispatcher pool**, NOT on a
tokio worker thread directly.  This is a deliberate insulation
layer: even if user callbacks block, the tokio runtime is
unaffected.  The dispatcher pool is sized at
`min(num_cpus, 4)` worker threads by default and is configurable
alongside the runtime via `tgm_runtime_configure`.

The contract is:

- **Must complete in < 100 µs of CPU time.**  The dispatcher pool is
  small; long callbacks queue up and starve other completions.
- **Must not allocate** in the hot path — prefer `noexcept`-decorated
  operations that signal a condition variable, fulfil a
  `std::promise`, or resume a `std::coroutine_handle<>`.  Lifting
  heavy work to a downstream thread is the user's responsibility.
- **Must not lock** any mutex that may be held outside the
  dispatcher pool.  Two-step pattern recommended: callback signals
  a `std::atomic<bool>` and notifies a condvar; consumer thread
  drains.
- **Must not throw.**  If a user callback throws, `panic = "abort"`
  on the FFI side terminates the process.  The plan deliberately
  does not silently swallow exceptions because doing so would mask
  user bugs.
- `result<T>::ok()` is the success/failure discriminator; check it
  before reading `value()`.

Stress test pinned in §11.5: a synthetic workload that registers
1000 callbacks, each running for `>= 1 ms`, must surface as
backpressure (queue saturation) rather than tokio runtime starvation
or deadlock.

### 4.2 `async/coro.hpp` — C++20 coroutines (opt-in)

The most ergonomic frontend.  Header-only, requires `-std=c++20`.

```cpp
namespace tensogram::coro {

template <typename T>
class task;  // lazy, single-await, awaitable<T>

class async_file {
public:
    [[nodiscard]] static task<async_file> open(
        const std::string& path,
        cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = {});

    [[nodiscard]] static task<async_file> open_remote(
        const std::string& url,
        const storage_options& opts = {},
        cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = {});

    [[nodiscard]] task<std::size_t>     message_count();
    [[nodiscard]] task<message>         decode_message(std::size_t i, decode_options = {});
    [[nodiscard]] task<metadata>        decode_metadata(std::size_t i);
    [[nodiscard]] task<decoded_object>  decode_object(std::size_t msg, std::size_t obj, decode_options = {});
    [[nodiscard]] task<std::vector<message>> decode_message_batch(
        std::span<const std::size_t> indices, decode_options = {});
    // …
};

class async_streaming_encoder {
public:
    [[nodiscard]] static task<async_streaming_encoder> create(
        const std::string& path_or_url,
        std::string_view metadata_json,
        encode_options opts = {});

    [[nodiscard]] task<void> write_object(std::string_view descriptor_json,
                                          std::span<const uint8_t> data);
    [[nodiscard]] task<void> write_preceder(std::string_view metadata_json);
    [[nodiscard]] task<void> finish(bool backfill = true);
};

// Block-on helper for top-level entry (e.g. main())
template <typename T>
T block_on(task<T> t);

// when_all helper
template <typename... Ts>
[[nodiscard]] task<std::tuple<Ts...>> when_all(task<Ts>... tasks);

template <typename T>
[[nodiscard]] task<std::vector<T>> when_all(std::vector<task<T>> tasks);

}  // namespace tensogram::coro
```

Producer use case:

```cpp
tensogram::coro::task<void> emit_steps(forecast_engine& fc) {
    auto enc = co_await tensogram::coro::async_streaming_encoder::create(
        "s3://my-bucket/run-2026-04-30/forecast.tgm",
        global_metadata_json,
        opts);

    while (auto step = co_await fc.next_step()) {
        // The next-step compute and the previous step's network write
        // overlap on a single thread because of co_await.
        co_await enc.write_object(step.descriptor_json, step.bytes);
    }

    co_await enc.finish(/*backfill=*/true);
}
```

Consumer use case:

```cpp
tensogram::coro::task<void> consume(const std::string& url) {
    auto f = co_await tensogram::coro::async_file::open_remote(url);
    auto count = co_await f.message_count();

    // Pipeline of two: while message i decodes, message i+1 prefetches.
    auto next = f.decode_message(0);
    for (std::size_t i = 0; i + 1 < count; ++i) {
        auto cur = std::move(next);
        next = f.decode_message(i + 1);
        auto msg = co_await std::move(cur);
        process(msg);
    }
    auto last = co_await std::move(next);
    process(last);
}
```

`task<T>` implementation:
- Lazy: nothing happens until `co_await`.
- Single-await: `task<T>` cannot be awaited twice (compile-time error
  via `&&`-qualified `await_resume`).
- Move-only.
- Holds `tgm_async_task_t*` + a `std::coroutine_handle<>` for the
  awaiting coroutine; `await_suspend` registers a completion callback
  that calls `coroutine_handle::resume()`.
- Continuation runs on the tokio worker thread by default; users can
  hop back to a specific executor with `co_await on(executor, …)`
  (deferred to a follow-up).

### 4.3 `async/std_future.hpp` — `std::future<T>` (C++17, **shipped in v1**)

For users who want the standard library type without coroutines.

```cpp
namespace tensogram::stdfuture {

class async_file {
public:
    [[nodiscard]] static std::future<async_file>
        open(const std::string& path,
             cancellation_token* token = nullptr,
             std::chrono::milliseconds timeout = {});

    [[nodiscard]] std::future<std::size_t>     message_count();
    [[nodiscard]] std::future<message>         decode_message(std::size_t i, decode_options = {});
    [[nodiscard]] std::future<metadata>        decode_metadata(std::size_t i);
    [[nodiscard]] std::future<decoded_object>  decode_object(std::size_t msg, std::size_t obj, decode_options = {});
    [[nodiscard]] std::future<std::vector<message>> decode_message_batch(
        std::span<const std::size_t> indices, decode_options = {});
    // …
};

class async_streaming_encoder {
public:
    [[nodiscard]] static std::future<async_streaming_encoder> create(
        const std::string& path_or_url,
        std::string_view metadata_json,
        encode_options opts = {});

    [[nodiscard]] std::future<void> write_object(std::string_view descriptor_json,
                                                 std::span<const uint8_t> data);
    [[nodiscard]] std::future<void> write_preceder(std::string_view metadata_json);
    [[nodiscard]] std::future<void> finish(bool backfill = true);
};

}  // namespace tensogram::stdfuture
```

Internally implemented via `std::promise<T>` set inside a
`completion_handler` callback.  Each future `.get()` blocks the
calling thread; failure throws the same typed `tensogram::error`
hierarchy used by the sync API.

Composition story is intentionally weak — `std::future<T>` has no
`.then`, no `when_all`.  Users wanting composition should use the
`coro` frontend.  Users on `-fno-exceptions` builds should use the
`core` callback frontend instead (the future surface needs
exceptions to propagate failures through `.get()`).

---

## 5. Streaming async writes (the producer path)

The HPC producer scenario is what makes streaming async writes
first-class instead of a follow-up.

### 5.1 What's new on the Rust side

`StreamingEncoder<W: Write>` (sync only) gains a sibling
`AsyncStreamingEncoder<W: AsyncWrite + Unpin>` in
`rust/tensogram/src/streaming_async.rs`.  The two share frame layout
code; only the I/O calls differ:

```rust
// Existing:
pub struct StreamingEncoder<W: Write> { /* … */ }
impl<W: Write> StreamingEncoder<W> {
    pub fn write_object(&mut self, desc: &DataObjectDescriptor, data: &[u8])
        -> Result<()>;
    pub fn finish(self) -> Result<()>;
}

// New:
pub struct AsyncStreamingEncoder<W: AsyncWrite + Unpin> { /* … */ }
impl<W: AsyncWrite + Unpin> AsyncStreamingEncoder<W> {
    pub async fn write_object(&mut self, desc: &DataObjectDescriptor, data: &[u8])
        -> Result<()>;
    pub async fn finish(self) -> Result<()>;
}
```

The frame-encoding logic (CBOR descriptor, hash slot computation,
length math) is shared via a private `FrameBuilder` helper.  Only the
final byte-emission step differs.

### 5.2 Sink types

For the FFI entry point, the sink is selected from the URL/path:

| Path | Sink |
|---|---|
| `/local/path.tgm` | `tokio::fs::File` |
| `s3://…`, `gs://…`, `az://…` | `object_store::MultipartUpload`-backed adapter |
| `https://…` (PUT-capable) | dedicated reqwest streaming sink |

#### Object-store buffering policy

`object_store::MultipartUpload` accepts whole parts, not byte-streams,
and AWS S3 enforces a **5 MiB minimum part size** (the last part is
exempt).  GCS and Azure have similar but not identical thresholds.
Tensogram frames are usually larger but can be smaller (header
metadata frames, small index/hash frames, small first messages).

The async encoder maintains an internal **part-buffer** sized
`MULTIPART_PART_SIZE_BYTES` (default 8 MiB, overridable via
`tgm_runtime_configure`).  Frames are appended to the buffer; when
the buffer reaches the threshold, the buffered prefix is uploaded as
one multipart part.  `finish()` uploads the buffer's residual
contents as the final part (which is exempt from the minimum-size
rule) and calls `CompleteMultipartUpload`.

#### Retry / part-idempotency

If `object_store` retries a part on transient network failure, the
**same byte range must be uploaded byte-identically**.  The encoder
guarantees this by:

1. Holding each part-buffer until `MultipartUpload::put_part` returns
   `Ok` — the part is not freed until the underlying store
   acknowledges it.
2. Hashing happens **once per part**, before upload.  The xxh3-64
   inline-slot digest covers the whole frame body and is computed
   in the calling thread before any object_store call.  Retry of a
   part does **not** trigger a re-hash; the digest in the frame
   footer is the value computed on first upload attempt.
3. The encoder rejects any `put_part` failure that is not classified
   as transient (e.g. 4xx errors) and surfaces it as
   `TGM_ERROR_REMOTE`.  The `.tgm` file is not finalised; cancellation
   semantics from §5.4 apply.

#### Backend-specific notes

| Backend | Quirks |
|---|---|
| AWS S3 | 5 MiB minimum part size; up to 10 000 parts per upload; `MultipartUpload` fully supported |
| GCS | Resumable upload semantics; `object_store` translates `MultipartUpload` to resumable; no minimum part size |
| Azure Blob | Block blob with 4 MiB block size default; `object_store` translates accordingly |

Per-backend tests in PR 3 cover the happy path and one transient
retry scenario.

### 5.3 Backfill behaviour

`finish_with_backfill` requires `Seek + Write` on the sync side.  On
the async side this becomes `AsyncSeek + AsyncWrite`:

- Local files: `tokio::fs::File` implements both.
- Object stores: **cannot seek** — multipart uploads append-only.  In
  this case, `backfill = true` is silently ignored and the message
  ships with `total_length = 0` (forward-scan-only).  We log a
  one-shot `tracing::warn!` and document this in the C++ header.

### 5.4 Cancellation mid-stream

If a producer's encoder task is cancelled mid-stream (timeout, user
cancel, job kill, node failure):

- The on-disk `.tgm` is **left as-is** — no truncation, no delete.
- The trailing partial frame is structurally invalid and a
  validating reader will reject the message.
- `finish` is never called; `total_length` in the preamble stays
  `0`, and there is no postamble — readers will see the invalid
  trailer and stop.

This matches the operational reality the user flagged:
truncated/incomplete files are not trusted by downstream operational
systems anyway.  The producer's caller is expected to detect failure
(via the cancellation/timeout error) and route the partial file to
quarantine or simply move on.

### 5.5 Producer-side hash-while-writing

The sync streaming encoder already does hash-while-encoding (per
`pipeline::copy_and_hash`).  The async encoder reuses the same code
path — hashing remains in the calling thread, never crosses thread
boundaries — so the determinism contract is identical:

- Transparent codecs: byte-identical output across thread counts.
- Opaque codecs (blosc2, zstd with workers): may differ but always
  round-trip losslessly.

#### Interaction with multipart retry

The hash is computed **once per frame**, in the calling thread,
before any object_store call.  If `object_store::MultipartUpload`
retries a part — even one that contains the frame whose hash was
just installed — the hash is **not** recomputed.  This guarantees:

- Bit-exact correspondence between the frame's installed hash slot
  and the bytes finally landed in the object store.
- No double-hashing CPU overhead under retry.
- Frame contents cannot drift between attempts because the encoder
  never mutates a buffer after hashing it.

---

## 6. Runtime model

**Fully contained, no leak to the caller.**

- `tensogram-ffi` owns the tokio runtime via the existing
  `SHARED_RUNTIME: OnceLock<Result<Runtime, String>>` in `remote.rs`.
- The runtime is multi-thread.  A new FFI entry
  `tgm_runtime_configure(workers: u32, dispatcher_workers: u32, multipart_part_size_bytes: u64)`
  may be called **once, before any other tgm_async_*** call;
  subsequent calls return `TGM_ERROR_INVALID_ARG`.  Defaults:
  - `workers = min(num_cpus, 8)` — chosen to give HPC nodes some
    headroom over the original 2-worker remote backend default,
    without over-provisioning since Tensogram's async work is mostly
    network-bound.  PR 6 includes a benchmark that measures
    saturation on a representative HPC node and may revise this
    default before merge.
  - `dispatcher_workers = min(num_cpus, 4)` — sized to handle
    callback dispatch without blocking tokio workers (see §4.1
    callback contract).
  - `multipart_part_size_bytes = 8 * 1024 * 1024` (8 MiB) — see §5.2.
- Direct exposure of the runtime handle, executor, or any tokio type
  is **explicitly forbidden** in the public C/C++ API.  No header
  ever includes a tokio type.
- The C++ wrapper compiles without any tokio headers, build flags,
  or transitive dependencies.
- Per-file isolation (separate runtime per `async_file`) is **not**
  in v1.  Q8 resolution: defer to v2 if operational evidence
  surfaces a need.
- **Runtime build failure** at startup (e.g. on systems where tokio
  cannot initialise due to ulimit or sandbox restrictions) is
  cached in the `OnceLock` and surfaces to every subsequent
  `tgm_async_*` call as `TGM_ERROR_IO` with a descriptive
  `tgm_last_error()` string.  No retry; users must restart the
  process after fixing the environment.

### Lifecycle

- Runtime is built lazily on first async FFI call.
- Runtime is **not torn down on `dlclose`** (would risk hanging on
  in-flight tasks); it lives until process exit.
- **Process exit is abrupt** — no `atexit` handler.  Tasks
  in-flight when the program exits are dropped without their
  completion callbacks firing.  Users who care about orderly
  shutdown call `tgm_runtime_shutdown_blocking(timeout_ms)`
  explicitly: this fires `runtime.shutdown_timeout(...)`, cancels
  all pending tasks, frees all `tgm_async_file_t` /
  `tgm_async_streaming_encoder_t` handles, and **returns the count
  of tasks that did not finish within the timeout** so the caller
  can log/abort accordingly.  Returning the count (rather than
  silently swallowing the timeout) is the "return to user if
  possible" surface the user requested.

---

## 7. Timeout and cancellation semantics

### 7.1 Timeouts

Every async-launching FFI entry takes a `uint64_t timeout_ms`.

- `0` = no timeout.
- `> 0` = task transitions to `TGM_ERROR_TIMEOUT` after the deadline.
- The deadline is wall-clock from the moment the FFI returns the task
  handle, not from when the underlying tokio future first polls.
- A timeout cancels the underlying future; partial work (e.g. half a
  decode) is dropped at the next yield point.  No partial result is
  returned.
- Implemented as `tokio::time::timeout(Duration::from_millis(ms), fut)`.

### 7.2 Cancellation

`tgm_cancellation_token_t` is an opaque, ref-counted token.

- A token may be passed to many tasks.
- Calling `tgm_cancellation_token_cancel` sets a flag; all referenced
  tasks transition to `TGM_ERROR_CANCELLED` at their next yield.
- Cancellation is **cooperative**: a task that has not yet hit a
  yield point (e.g. mid-CBOR-decode) does not stop until it does.
- The token can be safely freed before tasks complete; tasks hold
  their own strong reference.
- Implemented via a `tokio_util::sync::CancellationToken` cloned into
  every spawned task.

### 7.3 Interaction

A task that is both timed out and cancelled reports whichever fired
first.  Internal implementation uses `tokio::select!` over
`time::sleep(deadline)`, `cancellation_token.cancelled()`, and the
underlying future.

---

## 8. Error model

### 8.1 In the FFI core

Errors are integer codes (existing `tgm_error` enum, plus new
`TGM_ERROR_TIMEOUT` and `TGM_ERROR_CANCELLED`).  No exceptions cross
the FFI boundary — `panic = "abort"` is set, so any Rust panic
terminates the process rather than unwinding through C.

### 8.2 In `async/callback.hpp`

`result<T>` is a `tl::expected`-style discriminated union.  No
exceptions are thrown by callbacks; they always receive a
`result<T>` and inspect `.ok()`.

### 8.3 In `async/coro.hpp`

`co_await` on a failed task throws a typed C++ exception (the same
`tensogram::error` hierarchy used by the sync API).  This is
consistent with existing C++ idiom and with the sync API.

### 8.4 In `async/std_future.hpp`

`.get()` on a failed future throws.  Same exception types as the
coroutine frontend.

### 8.5 `-fno-exceptions` build mode

`async/callback.hpp` is the only frontend usable under
`-fno-exceptions`.  This is documented; the coroutine and
`std::future` frontends **require exceptions** because their type
contracts demand it.

### 8.6 Rust panic propagation

Tokio's worker traps panics in spawned tasks as task aborts; with
`panic = "abort"` set globally on the workspace, **any panic in
Rust async code terminates the process immediately** rather than
unwinding through the C ABI.

This is the same contract as the existing sync surface and is
relied upon by every binding.  The plan does not add any
panic-recovery layer:

- A panic in encode/decode is a library bug; users see a clean
  process abort with the panic message on stderr.
- A user callback that throws (in `async/callback.hpp`) is treated
  the same — `panic = "abort"` covers the C++→Rust→`panic_unwind`
  path.
- No `TGM_ERROR_PANIC` code is introduced; panics are not
  recoverable.

Rationale: introducing panic recovery would create silent failure
modes that mask library bugs.  The strict-input philosophy of the
codebase (§ existing AGENTS.md) is preserved.

---

## 9. Thread-safety and handle sharing

### 9.1 `tgm_async_file_t`

Internally backed by `Arc<TensogramFile>` (Python's
`AsyncTensogramFile` already does this).  Multiple threads may hold
clones of the handle simultaneously; ref count drops to zero when all
handles are closed.

This is a **change in semantics** from the sync `tgm_file_t`, which
is single-thread-owned.  Documented in the header.

### 9.2 `tgm_async_streaming_encoder_t`

Single-thread-owned.  Streaming writes are inherently sequential
(frame i must complete before frame i+1 writes), so allowing
concurrent access is meaningless.  Internally protected by a
`tokio::sync::Mutex` for safety only; misuse is detected and
reported as an error.

### 9.3 Tasks

Tasks themselves are safe to poll/join from any thread, but the
result must be consumed exactly once.  After a successful
`tgm_async_task_join_*` the result is owned by the caller; the task
handle is invalid for further joins (returns
`TGM_ERROR_INVALID_ARG`).

---

## 10. What's NOT in scope for v1

- **External tokio interop.**  Users supplying their own tokio
  runtime to host Tensogram's async work.
- **Boost.Asio / Folly frontends.**  Removed from the plan; can be
  added later as additive header-only files if a concrete user
  surfaces.
- **Observability hooks.**  `tracing` events are not exposed to C++.
  Deferred until a clear demand surfaces (likely
  `tgm_runtime_set_log_callback` later).
- **GPU-direct paths.**  Async FFI consumes/produces CPU buffers
  only; GPU offload is a separate axis.
- **MSVC / Windows.**  Linux + macOS only in v1.  Windows is a
  follow-up (mostly a CMake config issue, not a code-level redesign,
  but adds CI cost).
- **Streaming async reads as a separate API.**  Already covered by
  per-message async `decode_message` on the file type; an explicit
  "stream open from byte zero, get one message at a time" API can be
  added later if needed (it's syntactic sugar on top).
- **Python parity for streaming-async-write.**  The existing
  `AsyncTensogramFile` (Python) does not gain streaming-async-write
  in this scope.  The existing per-message async decode methods are
  unchanged.  Tracked as a v2 item.
- **Per-file runtime isolation.**  Single global runtime config in
  v1 (Q8); per-file isolation deferred.
- **Credential rotation for long-running producers.**  STS tokens
  expiring mid-stream are out of scope (Q12).

---

## 11. Test matrix

### 11.1 New Rust tests (under `rust/tensogram/tests/`)

- `streaming_async_local.rs` — `AsyncStreamingEncoder` to local file,
  parity check against sync `StreamingEncoder` (same bytes).
- `streaming_async_object_store.rs` — to S3-mock + GCS-mock via
  `object_store` test fixtures.
- `streaming_async_backfill.rs` — `finish` with `backfill=true` on
  seekable async sink; assertion that `total_length` is patched in
  both preamble and postamble.
- `cancellation.rs` — every async fn cancellable mid-flight; token
  fires; partial state cleaned up; subsequent calls work.
- `timeout.rs` — every async fn respects timeout; `TGM_ERROR_TIMEOUT`
  surfaces; partial state cleaned up.

### 11.2 New C++ tests (under `cpp/tests/`)

Mirror the Python `test_async.py` taxonomy:

- `test_async_callback.cpp` — `async/callback.hpp` happy path for
  every entry point.
- `test_async_coro.cpp` — `async/coro.hpp` happy path; requires C++20.
- `test_async_stdfuture.cpp` — `async/std_future.hpp` happy path.
- `test_async_streaming.cpp` — async streaming encoder, local file.
- `test_async_streaming_remote.cpp` — async streaming encoder
  against in-process HTTP fixture (no external network).
- `test_async_cancellation.cpp` — token-driven cancellation.
- `test_async_timeout.cpp` — deadline-driven cancellation.
- `test_async_threadsafety.cpp` — `tgm_async_file_t` sharable across
  threads; concurrent decode_message calls succeed.
- `test_async_lifecycle.cpp` — handle/task ownership invariants
  (double-free, use-after-free guards).
- `test_async_cross_frontend.cpp` — same workload via `callback`,
  `coro`, and `std_future`; all produce identical decoded bytes.
- `test_async_callback_contract.cpp` — pins the §4.1 contract
  bounds: 1000 callbacks each running ≥ 1 ms must surface as
  dispatcher backpressure rather than tokio runtime starvation;
  callback throwing aborts cleanly; callback allocating in hot
  path is detected by ASAN/leak-counter.

### 11.3 New cross-language parity tests

- `tests/cpp_async_parity/` — C++ async producer + Python async
  consumer round-trip, both reading from the same in-process HTTP
  fixture.
- `tests/streaming_producer_consumer/` — Two-process integration:
  C++ producer streams to a local FIFO / tmpfs path; C++ consumer
  reads concurrently.  Mocks the HPC scenario at OS level only —
  HPC-filesystem-specific testing (Lustre, GPFS, WekaFS, BeeGFS) is
  out of scope for CI and **handled separately** by the operational
  team on real cluster hardware.

### 11.4 Examples

- `examples/cpp/19_async_decode_remote.cpp` — open S3, decode 100
  messages with `coro::when_all`.
- `examples/cpp/20_async_producer.cpp` — streaming encoder from a
  fake forecast loop.
- `examples/cpp/21_async_consumer.cpp` — pipelined decode of a remote
  file.
- `examples/cpp/22_async_callback.cpp` — same as 19 but using the
  `callback` frontend.
- `examples/cpp/23_async_stdfuture.cpp` — `std::future` flavour.
- `examples/cpp/24_async_cancellation.cpp` — show timeout + cancel.

### 11.5 Concurrent failure-mode tests (new)

Production-bug-shaped scenarios that the happy-path matrix won't
catch:

- `test_async_concurrent_cancel.cpp` — 4 threads issuing
  `decode_message` while a 5th thread fires `cancel()` on a shared
  token; all 4 transition to `TGM_ERROR_CANCELLED` cleanly; no
  task leaks.
- `test_async_shutdown_during_flight.cpp` — `tgm_runtime_shutdown_blocking`
  called while N tasks are mid-flight; returns the count of
  unfinished tasks; subsequent `tgm_async_*` calls on already-built
  handles return `TGM_ERROR_IO`.
- `test_async_double_join.cpp` — `tgm_async_task_join_*` called
  twice on the same task returns `TGM_ERROR_INVALID_ARG` on the
  second call; result is consumed exactly once.
- `test_async_double_completion.cpp` — `tgm_async_task_set_completion`
  called twice on the same task returns `TGM_ERROR_INVALID_ARG`.
- `test_async_token_after_task.cpp` — token freed before tasks
  complete; tasks still resolve correctly (refcount semantics).
- `test_async_token_outlives_tasks.cpp` — tasks freed before token;
  cancel on the surviving token is a no-op.

### 11.6 Memory / sanitiser tests

- ASAN + LSAN run as part of `cpp-async-callback` and
  `cpp-async-coro` CI lanes (not just the existing `cpp` lane).
- Specific allocation pin: under ASAN, the callback frontend's hot
  path (`decode_message` + completion) must not allocate beyond the
  result buffer.  Implemented as a thread-local allocator counter
  that the test reads before/after the callback fires.
- TSAN run on the threadsafety + concurrent tests at least once per
  CI release cycle (not on every PR — cost-prohibitive).

---

## 12. Build matrix

### 12.1 Rust side — explicit feature inventory

Three crates change.  Here is the complete feature matrix:

| Crate | Existing features | New features | Pulls in (new) |
|---|---|---|---|
| `tensogram` | `async`, `remote`, `mmap`, codec gates | `async-write` (gates `streaming_async.rs`) | `tokio/io-util`, `tokio/fs` (already implied by `async`) |
| `tensogram-ffi` | (none — single crate, default-features) | `async` (gates the new FFI surface) | `tensogram/async`, `tensogram/async-write`, `tokio-util` for `CancellationToken`, `tokio/sync`, `tokio/time` |
| `tensogram-cli` | `grib`, `netcdf` | (no new features needed — CLI is sync) | (none) |

`tensogram-ffi`'s new `async` feature is **off by default** in cdylib
release builds.  It is enabled **on by default** when the C++ wrapper
is built with `TENSOGRAM_ASYNC=ON` (the cargo invocation in
`cpp/CMakeLists.txt` adds `--features async` to the cargo build line).

Cargo features are additive; turning on `tensogram-ffi/async` also
enables `tensogram/async` and `tensogram/async-write` transitively.

For Rust users who want to depend on `tensogram-ffi` directly (e.g.
embedding the C ABI into another Rust binary), the `async` feature
gates all `tgm_async_*` exports cleanly; the symbols are not present
in the cdylib when the feature is off.

### 12.2 C++ side

CMake additions:

```cmake
option(TENSOGRAM_ASYNC "Build async FFI surface" ON)

if(TENSOGRAM_ASYNC)
    target_compile_definitions(tensogram_ffi PRIVATE TENSOGRAM_ASYNC=1)
    # Pass --features async to cargo invocation
endif()

# Header detection and flag for coroutine frontend
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
    #if !defined(__cpp_impl_coroutine) || __cpp_impl_coroutine < 201902L
    #error
    #endif
    int main(){}
" TENSOGRAM_HAS_COROUTINES)

target_compile_definitions(tensogram INTERFACE
    TENSOGRAM_ASYNC_AVAILABLE=$<IF:$<BOOL:${TENSOGRAM_ASYNC}>,1,0>
    TENSOGRAM_HAS_COROUTINES=$<IF:$<BOOL:${TENSOGRAM_HAS_COROUTINES}>,1,0>
)
```

The `coro.hpp` header self-disables with `#if !TENSOGRAM_HAS_COROUTINES`
emitting a `#error` instructing the user to switch to `callback.hpp`
or `std_future.hpp`.  Similarly for users who include `async/*.hpp`
when `TENSOGRAM_ASYNC` is off.

### 12.3 Compiler matrix

| Compiler | C++17 sync | C++17 `callback` / `std_future` | C++20 `coro` |
|---|---|---|---|
| GCC ≥ 9 | ✅ | ✅ | ❌ (coroutine support partial) |
| GCC ≥ 11 | ✅ | ✅ | ✅ |
| Clang ≥ 12 | ✅ | ✅ | ✅ |
| Apple Clang (Xcode 13+) | ✅ | ✅ | ✅ |
| Apple Clang (Xcode 14+) | ✅ | ✅ | ✅ |
| MSVC | (out of scope) | (out of scope) | (out of scope) |

### 12.4 CI lanes

- `cpp-async-callback` — Linux + macOS, builds & tests
  `callback.hpp` / `std_future.hpp` on C++17.
- `cpp-async-coro` — Linux GCC 11 + Clang 14 + macOS Xcode 14, builds
  & tests `coro.hpp` on C++20.
- Existing `cpp` lane stays untouched (sync-only).

#### CI cost estimate

The two new lanes plus the integration test in PR 6 roughly **double
the C++ CI wall-clock time**.  PR 6 should include a brief
investigation of whether the `cpp-async-coro` lane needs to run on
every PR or whether it can be gated to the release branch only.
Recommendation: every PR for v1, narrow to release-branch only if
the cost becomes painful.

---

## 13. PR / sequencing breakdown

Six PRs, sized for review and bisect-friendliness:

### PR 1 — Rust `AsyncStreamingEncoder`

- New module `rust/tensogram/src/streaming_async.rs`.
- Refactor `StreamingEncoder` to factor frame-emission code into a
  `FrameBuilder` helper shared with the async sibling.
- Tests under `rust/tensogram/tests/streaming_async_*.rs`.
- No FFI / C++ changes.

### PR 2 — FFI async core (read path)

- Add `tgm_async_task_t`, `tgm_cancellation_token_t`, completion
  callbacks, typed joins for the existing read-side async fns.
- Wire to `TensogramFile` async methods.
- New error codes `TGM_ERROR_TIMEOUT`, `TGM_ERROR_CANCELLED`.
- `tgm_runtime_configure`, `tgm_runtime_shutdown_blocking`.
- Tests under `rust/tensogram-ffi/tests/`.

### PR 3 — FFI async core (write path)

- `tgm_async_streaming_encoder_*` family.
- Object-store and local file backends.
- Tests including in-process HTTP fixture for object-store path.

### PR 4 — C++ `async/callback.hpp` (callback frontend)

- Header-only; all definitions `inline` (Q10 resolution).
- Wraps every FFI entry point.
- `result<T>`, `cancellation_token`, completion-handler shape.
- Implements the §4.1 callback contract via the FFI dispatcher pool.
- Includes `test_async_callback_contract.cpp` from §11.2.
- 7+ test files, 3+ examples.

### PR 5 — C++ `async/coro.hpp` and `async/std_future.hpp`

- Both frontends in one PR (they share the `callback` lifetime
  management, just present different surfaces).
- C++20 `task<T>` implementation including `when_all`,
  `block_on`, and `async_for_each` (Q9 resolution).
- C++17 `std::future` adapter.
- All headers `inline` (Q10 resolution).
- 4+ test files, 3+ examples.
- New CMake CI lanes.

### PR 6 — Producer/consumer integration test, docs, polish

- Two-process HPC scenario test.
- `docs/src/guide/cpp-async.md` user guide.
- `docs/src/guide/cpp-streaming-async.md` producer recipe.
- Update `plans/DONE.md`, README, ARCHITECTURE.md.
- Migrate `plans/PLAN_CPP_ASYNC.md` → `plans/DONE.md` summary.

---

## 14. Risk register

| Risk | Mitigation |
|---|---|
| `object_store::MultipartUpload` semantics differ subtly across S3/GCS/Azure | Adopt the trait-level abstraction; per-backend tests in PR 3 |
| Coroutine lifetime bugs in `task<T>` (use-after-free of the awaiting handle) | Lean on existing reference impl patterns (cppcoro, libunifex); fuzz with sanitizers in CI |
| Mixed sync + async use of the same file handle leads to deadlock | Document that `tgm_file_t` and `tgm_async_file_t` are distinct; static assertion in C++ to prevent accidental aliasing |
| Cancellation races: token cancelled while task is mid-join | Tokens hold strong refs; tasks hold strong refs; both can outlive each other safely |
| Tokio runtime not torn down at process exit on Linux causes finalizer warnings | Already true for sync `remote` feature; explicit `tgm_runtime_shutdown_blocking` for users who care |
| Producer-consumer integration test flakiness on CI shared filesystems | Use in-process pipes / `mkfifo` / tmpfs only; HPC-filesystem testing is out-of-scope for CI (handled separately by ops) |
| C++20 coroutine ABI differences across libstdc++ / libc++ versions | CI matrix covers both; `coro.hpp` is purely header-only so each consumer compiles its own copy |
| Streaming encoder hash-while-writing with multipart upload races | Hashing is in calling thread, AFTER bytes are computed but BEFORE they're handed to the AsyncWrite — same model as sync; pinned via existing test fixtures |
| Backfill on object stores silently ignored confuses users | Documented in header + docs + one-shot `tracing::warn!`; `finish()` returns the actual `total_length` written so callers can detect |
| Producer killed mid-stream leaves an invalid `.tgm` on disk | Acceptable per design (§5.4): operational systems don't trust truncated files anyway; cancellation/timeout error surfaces to caller for quarantine routing |
| User callback blocks/allocates and starves dispatcher pool | §4.1 callback contract; dispatcher pool insulates tokio runtime; stress test in §11.2 (`test_async_callback_contract.cpp`) pins the bound |
| Header-only frontends inflate compile time (one copy per TU) | Documented as an explicit trade-off (Q10); ABI safety is the priority over compile-time cost; benchmark in PR 6 may revise if the overhead is large |
| `tgm_async_task_set_completion` race with task already-Ready | Atomic `state` enum with CAS dispatch; callback fires inline if state was Ready at registration time, otherwise queued for dispatcher pool; pinned in `test_async_double_completion.cpp` |
| Tokio runtime fails to build at startup (low ulimits, sandbox) | `OnceLock` caches the build error; every `tgm_async_*` returns `TGM_ERROR_IO` with descriptive `tgm_last_error()`; documented in §6 |
| Default 8 worker threads under-saturate large HPC nodes | PR 6 includes a benchmark on a representative HPC node measuring saturation; default may be revised before merge |

---

## 15. Resolved and open questions

### 15.1 Resolved

| # | Question | Decision |
|---|---|---|
| Q1 | Default worker count | `min(num_cpus, 8)`, override via `tgm_runtime_configure` |
| Q3 | `async/std_future.hpp` priority | **Ship in v1** alongside `core` and `coro` |
| Q5 | Tokio runtime shutdown semantics | Abrupt termination on process exit; explicit `tgm_runtime_shutdown_blocking(timeout_ms)` returns the count of tasks that did not finish so callers can react |
| Q6 | HPC filesystem testing | Out of scope for CI; tested separately on real cluster hardware by the ops team |
| Q7 | Streaming encoder finish on cancellation | Leave file as-is; truncated `.tgm` is not trusted by operational systems anyway, and the cancellation/timeout error surfaces to the caller |

### 15.2 Resolved (all recommendations accepted)

| # | Question | Decision |
|---|---|---|
| Q8 | `tgm_runtime_configure` shape | **Single global call** for v1. Per-file isolation deferred to v2 if ops demand surfaces. |
| Q9 | Async iterator helpers | **Ship `coro::async_for_each(file, fn)`** in `coro.hpp` — ~30 lines, matches Python pattern, near-zero cost. |
| Q10 | C++ symbol export / ABI | **Entire `async/*.hpp` family is header-only** — all out-of-line definitions marked `inline`. No new shared-library ABI surface beyond the C FFI. |
| Q11 | Naming `core.hpp` vs `callback.hpp` | **`async/callback.hpp`** — describes what users see; "core" reserved for internal Rust nomenclature. |
| Q12 | Credential rotation for long-running producers | **Out of scope for v1.** Track separately once an operational job hits the multi-hour STS limit. |

All design questions are now resolved. The plan is ready for implementation.

---

## 16. Acceptance checklist (post-implementation)

When all PRs land, the plan is done iff:

- [ ] `cargo test --workspace --all-features` passes including new async streaming tests.
- [ ] All new C++ test binaries compile and run on Linux + macOS, GCC + Clang, C++17 + C++20.
- [ ] The producer/consumer integration test reliably round-trips through both local file and object-store backends.
- [ ] Cancellation tokens fire and clean up resources within 100ms of being signalled.
- [ ] Timeouts surface as `TGM_ERROR_TIMEOUT` deterministically; partial state is cleaned up.
- [ ] The C++ wrapper builds with `-fno-exceptions` if only `core.hpp` is included.
- [ ] All async examples (`19_*` through `24_*`) build and run end-to-end.
- [ ] `tgm_runtime_shutdown_blocking` drains in-flight tasks within the supplied timeout and reports cleanly on overflow.
- [ ] Documentation under `docs/src/guide/cpp-async.md` and `cpp-streaming-async.md` is published.
- [ ] `plans/DONE.md` summary entry written; `plans/PLAN_CPP_ASYNC.md` retired.

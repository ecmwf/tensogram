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
                              │ #include <tensogram/async/coro.hpp>
                              │   or std_future.hpp / asio.hpp / core.hpp
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ C++ frontend headers (header-only, opt-in)                          │
│  • async/core.hpp        callback-based, C++17, always available    │
│  • async/coro.hpp        C++20 task<T> coroutines (opt-in)          │
│  • async/std_future.hpp  std::future<T> wrappers (opt-in)           │
│  • async/asio.hpp        Boost.Asio awaitables (slot reserved)      │
│  • async/folly.hpp       Folly SemiFuture wrappers (slot reserved)  │
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
a thin syntactic adapter.  This is what makes adding Asio/Folly
frontends later a header-only edit, not a re-architecture.

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

---

## 4. C++ Frontends

### 4.1 `async/core.hpp` — callback core (C++17, always available)

The minimum-viable C++ surface.  All other frontends are layered on
top of this one.

```cpp
namespace tensogram::async_core {

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

}  // namespace tensogram::async_core
```

This frontend is **always available** even if no other frontend is
included.  Its callback contract:
- Callbacks fire on the tokio worker thread that resolved the future.
- Callbacks must be cheap — they should signal a condition variable,
  promise, coroutine handle, or executor task and return.  Doing
  significant work on the tokio thread will starve the runtime.
- `result<T>::ok()` is the success/failure discriminator; check it
  before reading `value()`.

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

### 4.4 `async/asio.hpp` — Boost.Asio awaitables (slot reserved)

Header-only but **not shipped in v1**.  We design the FFI core to make
this addable later as a single new header.  Sketch:

```cpp
namespace tensogram::asio {
template <typename CompletionToken>
auto async_open(const std::string& path, CompletionToken&& tok);
// uses boost::asio::async_initiate to bridge into Asio's executor
}
```

### 4.5 `async/folly.hpp` — Folly SemiFuture (slot reserved)

Same: header-only, not shipped in v1, designed for additivity.

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

The remote object_store path is the trickiest: it needs to buffer
into multipart parts (typically 5 MiB minimum on AWS S3) and flush on
`finish`.  We adopt `object_store::MultipartUpload` directly — it
handles the part sequencing, ETag bookkeeping, and final CompleteMultipartUpload
call.

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

### 5.4 Producer-side hash-while-writing

The sync streaming encoder already does hash-while-encoding (per
`pipeline::copy_and_hash`).  The async encoder reuses the same code
path — hashing remains in the calling thread, never crosses thread
boundaries — so the determinism contract is identical:

- Transparent codecs: byte-identical output across thread counts.
- Opaque codecs (blosc2, zstd with workers): may differ but always
  round-trip losslessly.

---

## 6. Runtime model

**Fully contained, no leak to the caller.**

- `tensogram-ffi` owns the tokio runtime via the existing
  `SHARED_RUNTIME: OnceLock<Result<Runtime, String>>` in `remote.rs`.
- The runtime is multi-thread, currently 2 worker threads (matching
  the existing default).  A new FFI entry `tgm_runtime_configure(workers: u32)`
  may be called **once, before any other tgm_async_*** call;
  subsequent calls are no-ops.  Default workers = `min(num_cpus, 4)`
  — chosen for HPC nodes which typically have many cores but where
  Tensogram's network-bound async work doesn't benefit beyond ~4
  workers.
- Direct exposure of the runtime handle, executor, or any tokio type
  is **explicitly forbidden** in the public C/C++ API.  No header
  ever includes a tokio type.
- The C++ wrapper compiles without any tokio headers, build flags,
  or transitive dependencies.
- Default workers = `min(num_cpus, 8)` — chosen to give HPC nodes
  some headroom over the original 2-worker remote backend default,
  without over-provisioning since Tensogram's async work is mostly
  network-bound.  Override via `tgm_runtime_configure(workers)`.

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

### 8.2 In `async/core.hpp`

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

`async/core.hpp` is the only frontend usable under `-fno-exceptions`.
This is documented; the coroutine and `std::future` frontends
**require exceptions** because their type contracts demand it.

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
- **Asio frontend** (`async/asio.hpp`).  Designed for, not shipped.
- **Folly frontend** (`async/folly.hpp`).  Designed for, not shipped.
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
- **Async iteration helpers** (`async_for_each` etc.).  Easy to
  build on the coro frontend; ship the primitive first.

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

- `test_async_core_callback.cpp` — `async/core.hpp` happy path for
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
- `test_async_cross_frontend.cpp` — same workload via `core`,
  `coro`, and `std_future`; all produce identical decoded bytes.

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
  `core` callback frontend.
- `examples/cpp/23_async_stdfuture.cpp` — `std::future` flavour.
- `examples/cpp/24_async_cancellation.cpp` — show timeout + cancel.

---

## 12. Build matrix

### 12.1 Rust side

`tensogram-ffi` gains a `feature = "async"` flag that pulls in:
- `tokio` (already a transitive dep via `tensogram --features async,remote`).
- `tokio-util` for `CancellationToken`.
- `bytes` for AsyncWrite shim glue.

Default off in cdylib release builds.  Enabled by default when the
`async` Cargo feature on the `tensogram` crate is on (the C++ side
just enables the upstream feature).

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
emitting a `#error` instructing the user to switch to `core.hpp` or
`std_future.hpp`.  Similarly for users who include `async/*.hpp`
when `TENSOGRAM_ASYNC` is off.

### 12.3 Compiler matrix

| Compiler | C++17 sync | C++17 `core` / `std_future` | C++20 `coro` |
|---|---|---|---|
| GCC ≥ 9 | ✅ | ✅ | ❌ (coroutine support partial) |
| GCC ≥ 11 | ✅ | ✅ | ✅ |
| Clang ≥ 12 | ✅ | ✅ | ✅ |
| Apple Clang (Xcode 13+) | ✅ | ✅ | ✅ |
| Apple Clang (Xcode 14+) | ✅ | ✅ | ✅ |
| MSVC | (out of scope) | (out of scope) | (out of scope) |

### 12.4 CI lanes

- `cpp-async-callback` — Linux + macOS, builds & tests `core.hpp` /
  `std_future.hpp` on C++17.
- `cpp-async-coro` — Linux GCC 11 + Clang 14 + macOS Xcode 14, builds
  & tests `coro.hpp` on C++20.
- Existing `cpp` lane stays untouched (sync-only).

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

### PR 4 — C++ `async/core.hpp` (callback frontend)

- Header-only.
- Wraps every FFI entry point.
- `result<T>`, `cancellation_token`, completion-handler shape.
- 6+ test files, 3+ examples.

### PR 5 — C++ `async/coro.hpp` and `async/std_future.hpp`

- Both frontends in one PR (they share the `core` lifetime
  management, just present different surfaces).
- C++20 `task<T>` implementation including `when_all`.
- C++17 `std::future` adapter.
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

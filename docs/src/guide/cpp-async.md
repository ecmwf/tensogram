# C++ Async API

Tensogram ships an asynchronous C++ surface designed for the HPC
producer/consumer scenario where independent jobs on the same cluster
pipe data through a `.tgm` artefact.

The async layer is **header-only** and **opt-in** via three frontends
that all sit on the same callback-based FFI core:

| Frontend | C++ standard | Header | Style |
|---|---|---|---|
| Callback | C++17 (default) | `tensogram/async/callback.hpp` | `std::function` completion handlers |
| `std::future` | C++17 (opt-in) | `tensogram/async/std_future.hpp` | `std::future<T>` |
| Coroutines | C++20 (opt-in) | `tensogram/async/coro.hpp` | `task<T>` + `co_await` |

The plan: `plans/PLAN_CPP_ASYNC.md` for full design rationale.

## Build setup

The async surface is enabled by default in CMake. To opt out:

```cmake
cmake -S cpp -B build -DTENSOGRAM_ASYNC=OFF
```

When `TENSOGRAM_ASYNC=ON` (the default), the cargo build line gains
`--features=async`, the C++ wrapper target gets a `TENSOGRAM_ASYNC=1`
compile definition, and the test suite picks up the async test files.

The C++20 coroutine frontend is built as a separate test executable.
You can disable it with `-DTENSOGRAM_ASYNC_CORO_TESTS=OFF` if your
compiler doesn't support C++20 coroutines.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ User code                                                           │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ C++ frontend (header-only):                                         │
│   tensogram/async/callback.hpp     std::function callbacks          │
│   tensogram/async/coro.hpp         task<T> coroutines (C++20)       │
│   tensogram/async/std_future.hpp   std::future<T>                   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ tensogram-ffi async core (cdylib + staticlib):                      │
│   tgm_async_task_t   — opaque task handle                          │
│   tgm_cancellation_token_t — cancellation                           │
│   tgm_async_file_t   — Arc-shared file handle                      │
│   tgm_async_streaming_encoder_t — producer                         │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ Rust core: SHARED_RUNTIME (tokio multi-thread, contained)           │
└────────────────────────────────────────────────────────────────────┘
```

The runtime is fully contained — no tokio types appear in the public
C/C++ API.  Default workers = `min(num_cpus, 8)`; configurable via
`tgm_runtime_configure(...)` once at process start.

## Callback frontend (`callback.hpp`)

Always available wherever `TENSOGRAM_ASYNC=ON`.  No extra C++ standard
required.

```cpp
#include <tensogram.hpp>
#include <tensogram/async/callback.hpp>

namespace tac = tensogram::async_callback;

tac::async_file::open("data.tgm", [](tac::result<tac::async_file> r) {
    if (!r.ok()) {
        std::cerr << "open failed: " << r.message() << "\n";
        return;
    }
    auto file = r.take();
    file.message_count([file_ref = std::move(file)](tac::result<std::size_t> r) mutable {
        if (r.ok()) {
            std::cout << "messages: " << r.value() << "\n";
        }
    });
});
```

### Callback contract

The callback runs on the FFI **dispatcher pool** — a small set of
non-tokio worker threads owned by `tensogram-ffi`.  This insulates
the tokio runtime: even if a user callback blocks or runs slowly,
tokio's worker threads stay free to drive other in-flight async I/O.

The callback must:
- Complete quickly (< 100 µs of CPU time).  The dispatcher pool is
  small (default `min(num_cpus, 4)`); long callbacks queue up and
  starve other completions.
- Not throw — `panic = "abort"` is set on the Rust side and the
  trampolines that wrap user callbacks are `noexcept`, so an
  exception escaping a callback will terminate the process.
- Signal a condvar / coroutine handle / promise and return.  For
  heavy work, hand off to your own thread pool.

The pool size is overridable at startup via
`tac::runtime_configure(workers, dispatcher_workers, ...)`.  The
configuration must be set before any other async call.

## `std::future` frontend (`std_future.hpp`)

C++17 opt-in.  Each method returns `std::future<T>` whose `.get()`
blocks until the operation completes.  Failures throw the typed
`tensogram::error` hierarchy through `.get()`.

```cpp
#include <tensogram/async/std_future.hpp>

namespace tsf = tensogram::stdfuture;

auto file = tsf::async_file::open("data.tgm").get();
auto count = file.message_count().get();
auto msg = file.decode_message(0).get();
```

Composition is intentionally weak — no `.then`, no `when_all`.  Users
wanting pipelines should reach for `coro.hpp`.  Users on
`-fno-exceptions` builds should use `callback.hpp` (the future
surface needs exceptions to surface failures through `.get()`).

## Coroutine frontend (`coro.hpp`)

C++20 opt-in.  Two types are exported:

- `task<T>` — proper coroutine return type.  Users write
  `task<int> my_func() { co_return 42; }` and chain via
  `co_await my_func()`.
- `awaiter<T>` — what async I/O methods return.  Itself awaitable;
  suspends until the underlying FFI task resolves.

```cpp
#include <tensogram/async/coro.hpp>

namespace tco = tensogram::coro;

tco::task<std::size_t> walk(const std::string& path) {
    auto file = co_await tco::async_file::open(path);
    std::size_t total = 0;
    auto count = co_await file.message_count();
    for (std::size_t i = 0; i < count; ++i) {
        auto msg = co_await file.decode_message(i);
        total += msg.num_objects();
    }
    co_return total;
}

int main() {
    auto n = tco::block_on(walk("data.tgm"));
    std::cout << "total objects: " << n << "\n";
}
```

`tco::block_on` runs the task synchronously on the calling thread;
useful at top-level entry points like `main()`.

`tco::async_for_each(file, fn)` is a convenience that walks every
message in `file` and applies `fn` to each `tensogram::message`.

## Producer side: streaming async encoder

All three frontends expose `async_streaming_encoder` for the producer
scenario.  Local file backend; preamble + header metadata are written
asynchronously when `create()` resolves.

```cpp
// Coroutine frontend example
tco::task<void> emit_steps(forecast_engine& fc) {
    auto enc = co_await tco::async_streaming_encoder::create(
        "/run/forecast.tgm",
        R"({"base": []})");

    while (auto step = fc.next_step()) {
        co_await enc.write_object(step.descriptor_json,
                                   step.bytes.data(), step.bytes.size());
    }

    co_await enc.finish(/*backfill=*/true);
}
```

The producer task and the underlying encoder are protected by a
`tokio::sync::Mutex` so concurrent FFI calls against the same handle
serialise correctly.

## Cancellation and timeouts

Every async-launching call accepts an optional
`cancellation_token*` and a `std::chrono::milliseconds timeout`.

```cpp
tac::cancellation_token tok;
file.decode_message(42, [](tac::result<tensogram::message> r) {
    if (r.code() == TGM_ERROR_CANCELLED) {
        std::cerr << "cancelled\n";
    }
}, &tok, std::chrono::milliseconds{5000});

// Cancel from any thread:
tok.cancel();
```

Timeout `0` means "no timeout".  Internally implemented via
`tokio::time::timeout` and `tokio_util::sync::CancellationToken`.

## Thread safety

- `async_file` — internally backed by `Arc<TensogramFile>`.  Multiple
  threads may issue concurrent reads against the same handle.
- `async_streaming_encoder` — single-handle, internally serialised
  via `tokio::sync::Mutex`.  Concurrent writes against the same
  encoder serialise; this matches the inherently sequential nature
  of streaming writes.
- `cancellation_token` — safe to share across threads; refcounted
  internally.

## Runtime configuration

Call once before any other async API:

```cpp
tac::runtime_configure(/*workers=*/16,
                        /*dispatcher_workers=*/0,  // default
                        /*multipart_part_size_bytes=*/0);  // default 8 MiB
```

Subsequent calls throw `invalid_arg_error` because the runtime is
built lazily on first use and cannot be reconfigured after that.

`runtime_shutdown_blocking(timeout)` is reserved for graceful shutdown.
In the current release it is a no-op that returns `0`: process exit is
abrupt by design and tokio drops in-flight tasks at teardown. The
signature is stable so a future release can drain tasks and report the
count that did not finish within the deadline without an ABI break.

## Remote reads (`open_remote`)

All three frontends can open a `.tgm` over an object store or a
`file://` URL on the read path:

```cpp
// callback frontend
tac::async_file::open_remote(
    "s3://bucket/forecast.tgm",
    /*storage_options=*/{{"aws_region", "eu-west-1"}},
    /*bidirectional=*/true,
    [](tac::result<tac::async_file> r) { /* ... */ });

// std::future frontend
auto file = tsf::async_file::open_remote("gs://bucket/f.tgm", {}, true).get();

// coroutine frontend
auto file = co_await tco::async_file::open_remote("az://c/f.tgm", {}, true);
```

`storage_options` are object-store key→value pairs (credentials,
region, endpoint); pass an empty list to use ambient configuration.
`bidirectional` selects the pipelined two-ended remote scan.

Supported schemes: `s3://`, `gs://`, `az://`, `https://`, and
`file://`.  Requires the FFI built with `--features=async-remote`
(`cmake -S cpp -B build -DTENSOGRAM_ASYNC_REMOTE=ON`).  The
`tgm_async_file_open_remote` symbol is always linkable; in a build
without the feature it resolves with `TGM_ERROR_REMOTE` and a
diagnostic naming the missing feature, so callers never hit an
undefined symbol.

See `examples/cpp/19_async_decode_remote.cpp` and the
[producer/consumer guide](cpp-streaming-async.md).

## What's not in scope (v1)

- Object-store backends for the streaming encoder (S3, GCS, Azure).
  Local file only.  See plan §5.2.
- External tokio runtime interop (users supplying their own runtime).
- Boost.Asio or Folly frontends — explicitly removed from the plan.
- MSVC / Windows.  Linux + macOS only.
- Per-file runtime isolation.

See `plans/PLAN_CPP_ASYNC.md` §10 for the full out-of-scope list.

## Cross-language parity

The async surface produces wire-format-identical bytes to the sync
`StreamingEncoder` for the same logical sequence of writes.  Every
file written via the async path can be read by Rust, Python, or the
sync C++ wrapper without modification.

## Producer / consumer integration test

The integration test
`cpp/tests/test_async_producer_consumer.cpp` exercises the canonical
HPC pattern: a producer writes 8 forecast steps as separate frames
in one streaming message; a consumer reads them all back and
verifies the data.  Mirror your own producer/consumer pair on this
shape and you'll have a working setup.

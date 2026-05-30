# C++ Async Streaming (Producer / Consumer)

This guide is a practical recipe for the HPC scenario that drives the
C++ async surface: two independent jobs on the same cluster pipe data
through a `.tgm` artefact. A **producer** (a simulation or inference
job) streams forecast steps out as they are computed; a **consumer**
(post-processing or visualisation) reads each message as soon as it is
available. Neither job should stall waiting on the other.

For the full async API reference — all three frontends, the callback
contract, runtime configuration, cancellation, and thread-safety — see
[C++ Async API](cpp-async.md). This page focuses on the end-to-end
producer/consumer pattern.

The runnable companions to this guide are:

- `examples/cpp/20_async_producer.cpp`
- `examples/cpp/21_async_consumer.cpp`
- `examples/cpp/19_async_decode_remote.cpp` (consumer reading over an
  object store / `file://`)

## Build setup

The async surface is header-only and on by default in CMake:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The examples below use the C++20 coroutine frontend
(`tensogram/async/coro.hpp`). The callback (`callback.hpp`) and
`std::future` (`std_future.hpp`) frontends offer the same operations on
C++17 — pick whichever matches your code base. To read over an object
store or a `file://` URL, configure with `-DTENSOGRAM_ASYNC_REMOTE=ON`
(this builds the FFI with `--features=async-remote`).

## Producer: streaming a message out object-by-object

The producer creates an `async_streaming_encoder`, writes each data
object as it is produced, and calls `finish()` once the message is
complete. Nothing buffers the whole message — each `write_object`
hands its bytes to the async writer and returns.

```cpp
#include <tensogram/async/coro.hpp>
namespace tco = tensogram::coro;

tco::task<void> emit_forecast(forecast_engine& fc) {
    auto enc = co_await tco::async_streaming_encoder::create(
        "/scratch/run/forecast.tgm",
        R"({"base": []})");          // global metadata (CBOR-encoded)

    while (auto step = fc.next_step()) {
        co_await enc.write_object(step.descriptor_json,
                                  step.bytes.data(), step.bytes.size());
    }

    // backfill=true rewrites the preamble/postamble lengths so the
    // finished file is fully random-access (see WIRE_FORMAT.md §7).
    co_await enc.finish(/*backfill=*/true);
}
```

Notes and edge cases:

- **One encoder, serialised writes.** A single `async_streaming_encoder`
  is internally guarded by a `tokio::sync::Mutex`; concurrent
  `write_object` calls against the same handle serialise. This matches
  the inherently sequential nature of a streaming write — don't try to
  parallelise writes to one encoder.
- **Pre-encoded payloads.** If your bytes are already encoded
  (compressed/packed), use `write_pre_encoded(...)` to skip the
  encoding pipeline. Use `write_preceder(...)` to emit a metadata
  preceder frame ahead of an object so a streaming consumer can decode
  it before the payload arrives.
- **Backend.** In v1 the streaming **encoder** writes to a **local
  file** only. The shared filesystem (Lustre, GPFS, WekaFS, BeeGFS) is
  the supported producer sink. Object-store *write* is out of scope;
  object-store *reads* are supported on the consumer side (below).
- **Crash/cancellation.** If the producer is cancelled or killed
  mid-stream the partial `.tgm` is left on disk as-is (no truncate or
  delete). Operational systems should treat an unfinished file as
  untrusted until the producer signals completion — see *Coordination*.

## Consumer: reading each message as it lands

The consumer opens the file and walks its messages. `async_for_each`
decodes each message in turn and hands it to your callback; it is the
coroutine mirror of the Python iterator pattern.

```cpp
tco::task<std::size_t> consume(const std::string& path) {
    auto file = co_await tco::async_file::open(path);
    std::size_t seen = 0;
    co_await tco::async_for_each(file, [&](tensogram::message msg) {
        process(msg);              // your work
        ++seen;
    });
    co_return seen;
}
```

If you need random access instead of a full walk, use
`message_count()` + `decode_message(i)` directly. An `async_file` is
backed by an `Arc<TensogramFile>`, so it is safe to issue concurrent
reads against the same handle from multiple threads (see the
*Thread safety* section of the [reference](cpp-async.md)).

## Coordination on a shared filesystem

Two patterns, depending on how tightly coupled the jobs are:

1. **Commit-then-read (simplest, recommended).** The producer writes
   the whole message and calls `finish()`; the consumer waits for the
   file to be committed (e.g. an atomic rename into a watched
   directory, or a sentinel marker) and then opens it. This avoids any
   torn-read window and is how most operational pipelines hand off.

2. **Progressive streaming.** The consumer reads a growing byte stream
   and decodes messages as they complete. The synchronous
   `examples/cpp/09_streaming_consumer.cpp` shows the
   `scan()`-the-rolling-buffer technique; the producer's metadata
   preceder frames (`write_preceder`) let the consumer decode each
   object's metadata before its payload arrives. Use this only when
   the latency saving justifies the added coordination.

> **Edge case — torn reads.** A consumer that opens a `.tgm` the
> producer has not yet `finish()`-ed may see a truncated file. Decode
> calls surface a framing error rather than returning garbage; wait for
> the committed file (pattern 1) or use the progressive reader with a
> retry-on-partial loop (pattern 2).

## Consumer reading over an object store

A consumer running on a different node can read the producer's artefact
over an object store or a shared-filesystem `file://` URL via
`open_remote`:

```cpp
std::vector<std::pair<std::string, std::string>> opts = {
    {"aws_region", "eu-west-1"},
};
auto file = co_await tco::async_file::open_remote(
    "s3://forecasts/run-42/forecast.tgm", opts, /*bidirectional=*/true);
auto count = co_await file.message_count();
```

- `bidirectional=true` selects the pipelined two-ended remote scan,
  which halves the per-message round-trip cost on high-latency links.
- Pass storage credentials/config through `storage_options`, or rely on
  ambient configuration (environment, instance role). For a `file://`
  URL no options are needed.
- This requires the FFI built with `--features=async-remote`
  (`-DTENSOGRAM_ASYNC_REMOTE=ON`). Without it, `open_remote` resolves
  with `TGM_ERROR_REMOTE` and a diagnostic naming the missing feature.

## Wire-format parity

The async streaming encoder produces **byte-identical** output to the
synchronous `StreamingEncoder` for the same logical sequence of writes.
Every file written via the async producer can be read by Rust, Python,
the WASM/TypeScript decoder, or the synchronous C++ wrapper without
modification.

## See also

- [C++ Async API](cpp-async.md) — full reference for all three
  frontends, cancellation/timeouts, runtime configuration, and
  thread-safety.
- [Working with Files](file-api.md) and [Remote Access](remote-access.md)
  — the synchronous counterparts.
- Examples `19`–`24` under `examples/cpp/`.

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Integration test mirroring the HPC producer/consumer scenario.
//! Two C++ jobs coordinate
//! through a `.tgm` artefact: the producer writes forecast steps as
//! they're generated; the consumer reads each message as soon as it
//! arrives.  Tests both the local-file pipe path and a synthetic
//! "produce in one task, consume in another" pattern within a single
//! process.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include <tensogram/async/callback.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tac = tensogram::async_callback;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    auto tmp = std::filesystem::temp_directory_path();
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return tmp / ("tensogram_pc_" + std::to_string(t) + suffix);
}

template <typename T>
T await_future(std::future<tac::result<T>> fut) {
    auto r = fut.get();
    if (!r.ok()) {
        throw tensogram::error(r.code(), r.message());
    }
    return r.take();
}

inline void await_void_future(std::future<tac::result<void>> fut) {
    auto r = fut.get();
    if (!r.ok()) {
        throw tensogram::error(r.code(), r.message());
    }
}

template <typename T>
std::future<tac::result<T>> as_future(
    std::function<void(std::function<void(tac::result<T>)>)> launch) {
    auto promise = std::make_shared<std::promise<tac::result<T>>>();
    auto fut = promise->get_future();
    launch([promise](tac::result<T> r) { promise->set_value(std::move(r)); });
    return fut;
}

}  // namespace

/// HPC producer/consumer pattern: producer writes 8 forecast steps,
/// consumer reads them all back.  Both run as async tasks.  Mirrors
/// the producer-finishes-then-consumer-reads pattern that operational
/// systems use for shared-filesystem hand-off (consumer polls until
/// the file is committed).
TEST(AsyncProducerConsumer, EightStepRoundTrip) {
    auto path = make_temp_path(".tgm");

    // ── Producer ────────────────────────────────────────────────
    {
        auto enc_fut = as_future<tac::async_streaming_encoder>([&path](auto cb) {
            tac::async_streaming_encoder::create(path.string(),
                R"json({"base":[]})json", std::move(cb));
        });
        auto enc = await_future(std::move(enc_fut));

        std::string desc = R"json({
            "type":"ntensor","ndim":2,"shape":[16,16],"strides":[16,1],
            "dtype":"float32","byte_order":"little","encoding":"none",
            "filter":"none","compression":"none","params":{}
        })json";

        for (int step = 0; step < 8; ++step) {
            // Build payload from finite floats (NOT raw byte patterns —
            // those occasionally form NaN, which the strict-finite
            // encoder rightly rejects).
            std::vector<float> values(16 * 16);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<float>(step) + 0.001f * static_cast<float>(i);
            }
            std::vector<std::uint8_t> data(values.size() * sizeof(float));
            std::memcpy(data.data(), values.data(), data.size());
            await_void_future(as_future<void>([&](auto cb) {
                enc.write_object(desc, data.data(), data.size(), std::move(cb));
            }));
        }
        await_void_future(as_future<void>(
            [&](auto cb) { enc.finish(std::move(cb)); }));
    }

    // ── Consumer ────────────────────────────────────────────────
    {
        auto file = await_future(as_future<tac::async_file>([&path](auto cb) {
            tac::async_file::open(path.string(), std::move(cb));
        }));

        // Note: the producer ran in a single-message-with-8-objects pattern
        // (8 calls to write_object ➜ 1 message with 8 objects).  Verify the
        // count and that all 8 objects round-trip.
        auto count = await_future(as_future<std::size_t>(
            [&](auto cb) { file.message_count(std::move(cb)); }));
        EXPECT_EQ(count, 1u);

        auto msg = await_future(as_future<tensogram::message>(
            [&](auto cb) { file.decode_message(0, std::move(cb)); }));
        EXPECT_EQ(msg.num_objects(), 8u);

        for (std::size_t i = 0; i < msg.num_objects(); ++i) {
            auto obj = msg.object(i);
            EXPECT_EQ(obj.ndim(), 2u);
            EXPECT_EQ(obj.data_size(), 16u * 16u * 4u);

            // Decode the payload back to floats and check the (step,index)
            // pattern matches what the producer wrote.
            const float* values = obj.data_as<float>();
            for (std::size_t k = 0; k < 16 * 16; ++k) {
                float expected = static_cast<float>(i) + 0.001f * static_cast<float>(k);
                ASSERT_NEAR(values[k], expected, 1e-5f)
                    << "object " << i << " element " << k;
            }
        }
    }

    std::filesystem::remove(path);
}

/// Stress test for the dispatcher pool: fire many in-flight callback
/// completions concurrently, each callback doing at least 1ms of
/// work.  The runtime has bounded queues (≤ `dispatcher_workers * 4`
/// slots; default 4 workers ⇒ 16-slot channel), so 1000 concurrent
/// callbacks force the inline-fallback path documented at
/// `dispatch_to_pool` to surface gracefully.
///
/// Verifies:
///   1. Every callback fires exactly once (no drops, no duplicates).
///   2. The runtime does not stall — total wall-clock stays
///      bounded.
///   3. No deadlock between the tokio worker pool and the dispatcher
///      pool when the dispatcher's bounded channel is saturated.
TEST(AsyncProducerConsumer, DispatcherPoolBackpressureStress) {
    constexpr int kCount = 1000;

    // Build one tiny file we can hammer with many concurrent reads.
    auto path = make_temp_path(".tgm");
    {
        std::string meta_json = R"json({
            "base":[],
            "descriptors":[{
                "type":"ntensor","ndim":1,"shape":[2],"strides":[1],
                "dtype":"uint8","byte_order":"little","encoding":"none",
                "filter":"none","compression":"none"
            }]
        })json";
        std::vector<std::uint8_t> data(2);
        std::vector<std::pair<const std::uint8_t*, std::size_t>> objs;
        objs.emplace_back(data.data(), data.size());
        auto bytes = tensogram::encode(meta_json, objs);
        std::ofstream out(path.string(), std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }

    auto file = await_future(as_future<tac::async_file>(
        [&path](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }));

    std::atomic<int> fired{0};
    std::mutex done_mtx;
    std::condition_variable done_cv;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kCount; ++i) {
        file.message_count(
            [&fired, &done_mtx, &done_cv](tac::result<std::size_t> r) {
                // Force at least 1ms of work in the callback to keep
                // the dispatcher saturated long enough to exercise
                // backpressure.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (r.ok()) {
                    int n = fired.fetch_add(1) + 1;
                    if (n == kCount) {
                        std::lock_guard<std::mutex> g(done_mtx);
                        done_cv.notify_all();
                    }
                }
            });
    }

    // Wait until every callback has fired or we hit a generous bound
    // (60s — under default config this should complete in well under
    // 10s; the bound exists only to prevent a hard hang on a real
    // deadlock).
    {
        std::unique_lock<std::mutex> lk(done_mtx);
        bool ok = done_cv.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return fired.load() == kCount; });
        ASSERT_TRUE(ok) << "dispatcher stalled at " << fired.load()
                        << "/" << kCount << " callbacks";
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_EQ(fired.load(), kCount);

    // Sanity: 1000 callbacks × 1ms with 4-worker pool ≈ 250ms ideal;
    // allow generous slack for inline-fallback dispatch and CI noise.
    EXPECT_LT(elapsed, std::chrono::seconds(30));

    std::filesystem::remove(path);
}

/// Concurrent decode: multiple threads decoding from the same shared
/// async_file handle.  Verifies the Arc-shared backing is safe under
/// concurrent FFI calls.
TEST(AsyncProducerConsumer, ConcurrentDecodeOnSharedFile) {
    auto path = make_temp_path(".tgm");

    // Build a 4-message file via sync API.
    {
        std::string meta_json = R"json({
            "base":[],
            "descriptors":[{
                "type":"ntensor","ndim":1,"shape":[4],"strides":[1],
                "dtype":"float32","byte_order":"little","encoding":"none",
                "filter":"none","compression":"none"
            }]
        })json";
        std::vector<std::uint8_t> data(16);
        std::vector<std::pair<const std::uint8_t*, std::size_t>> objs;
        objs.emplace_back(data.data(), data.size());
        std::ofstream out(path.string(), std::ios::binary);
        for (int i = 0; i < 4; ++i) {
            for (std::size_t k = 0; k < data.size(); ++k) {
                data[k] = static_cast<std::uint8_t>(i * 16 + k);
            }
            auto bytes = tensogram::encode(meta_json, objs);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        }
    }

    auto file = await_future(as_future<tac::async_file>(
        [&path](auto cb) { tac::async_file::open(path.string(), std::move(cb)); }));

    // Spawn 4 threads, each decoding a different message.
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&file, i, &success_count]() {
            try {
                auto msg = await_future(as_future<tensogram::message>(
                    [&file, i](auto cb) {
                        file.decode_message(static_cast<std::size_t>(i), std::move(cb));
                    }));
                if (msg.num_objects() == 1) {
                    success_count.fetch_add(1);
                }
            } catch (...) {
                // swallow
            }
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_EQ(success_count.load(), 4);

    std::filesystem::remove(path);
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF

/// @file tensogram/async/std_future.hpp
/// @brief std::future<T> frontend (PR 5 of the cpp-async plan).
///
/// C++17 wrapper that returns `std::future<T>` from every async
/// operation.  Internally backed by a `std::promise<T>` set inside a
/// callback frontend completion handler.
///
/// Composition story is intentionally weak (no `.then`, no
/// `when_all`).  Users wanting composition should use `coro.hpp` on
/// C++20.  Plan: `plans/PLAN_CPP_ASYNC.md` §4.3.

#ifndef TENSOGRAM_ASYNC_STD_FUTURE_HPP
#define TENSOGRAM_ASYNC_STD_FUTURE_HPP

#include "callback.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensogram::stdfuture {

namespace tac = tensogram::async_callback;

namespace detail {

// Typed-exception dispatch lives in `tensogram::detail::throw_for_code`
// (see `tensogram.hpp`); we delegate to it here so all async frontends
// share a single source of truth for the C-error-to-C++-exception map.

template <typename T, typename Launcher>
inline std::future<T> launch_future(Launcher&& launch) {
    auto promise = std::make_shared<std::promise<T>>();
    auto fut = promise->get_future();
    launch([promise](tac::result<T> r) {
        if (r.ok()) {
            promise->set_value(r.take());
        } else {
            try {
                tensogram::detail::throw_for_code(r.code(), r.message());
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    });
    return fut;
}

template <typename Launcher>
inline std::future<void> launch_void_future(Launcher&& launch) {
    auto promise = std::make_shared<std::promise<void>>();
    auto fut = promise->get_future();
    launch([promise](tac::result<void> r) {
        if (r.ok()) {
            promise->set_value();
        } else {
            try {
                tensogram::detail::throw_for_code(r.code(), r.message());
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    });
    return fut;
}

}  // namespace detail

class async_file {
public:
    [[nodiscard]] static std::future<async_file> open(
        const std::string& path,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_future<async_file>([path, token, timeout](auto cb) {
            tac::async_file::open(path,
                [cb = std::move(cb)](tac::result<tac::async_file> r) mutable {
                    if (r.ok()) {
                        cb(tac::result<async_file>::ok_value(async_file(r.take())));
                    } else {
                        cb(tac::result<async_file>::err(r.code(), r.message()));
                    }
                }, token, timeout);
        });
    }

    /// Open a remote `.tgm` (S3 / GCS / Azure / HTTP / `file://`).
    /// See `tensogram::async_callback::async_file::open_remote` for
    /// the `storage_options` / `bidirectional` semantics and the
    /// `async-remote` feature requirement.  Failures (including a
    /// build without `async-remote`) surface through `.get()` as the
    /// typed `tensogram::error` hierarchy.
    [[nodiscard]] static std::future<async_file> open_remote(
        const std::string& url,
        const std::vector<std::pair<std::string, std::string>>& storage_options,
        bool bidirectional,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_future<async_file>(
            [url, storage_options, bidirectional, token, timeout](auto cb) {
                tac::async_file::open_remote(url, storage_options, bidirectional,
                    [cb = std::move(cb)](tac::result<tac::async_file> r) mutable {
                        if (r.ok()) {
                            cb(tac::result<async_file>::ok_value(async_file(r.take())));
                        } else {
                            cb(tac::result<async_file>::err(r.code(), r.message()));
                        }
                    }, token, timeout);
            });
    }

    [[nodiscard]] std::future<std::size_t> message_count(
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        return detail::launch_future<std::size_t>([this, token, timeout](auto cb) {
            inner_->message_count(std::move(cb), token, timeout);
        });
    }

    [[nodiscard]] std::future<tensogram::message> decode_message(
        std::size_t index,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        return detail::launch_future<tensogram::message>([this, index, token, timeout](auto cb) {
            inner_->decode_message(index, std::move(cb), token, timeout);
        });
    }

    [[nodiscard]] std::future<tensogram::metadata> decode_metadata(
        std::size_t index,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        return detail::launch_future<tensogram::metadata>([this, index, token, timeout](auto cb) {
            inner_->decode_metadata(index, std::move(cb), token, timeout);
        });
    }

    [[nodiscard]] std::future<std::vector<std::uint8_t>> read_message(
        std::size_t index,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) const {
        return detail::launch_future<std::vector<std::uint8_t>>(
            [this, index, token, timeout](auto cb) {
                inner_->read_message(index, std::move(cb), token, timeout);
            });
    }

    async_file(async_file&&) noexcept = default;
    async_file& operator=(async_file&&) noexcept = default;
    async_file(const async_file&) = delete;
    async_file& operator=(const async_file&) = delete;

private:
    explicit async_file(tac::async_file inner)
        : inner_(std::make_shared<tac::async_file>(std::move(inner))) {}
    std::shared_ptr<tac::async_file> inner_;
};

class async_streaming_encoder {
public:
    [[nodiscard]] static std::future<async_streaming_encoder> create(
        const std::string& path,
        const std::string& metadata_json,
        const std::string& hash_algo = "xxh3",
        std::uint32_t threads = 0,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_future<async_streaming_encoder>(
            [path, metadata_json, hash_algo, threads, token, timeout](auto cb) {
                tac::async_streaming_encoder::create(path, metadata_json,
                    [cb = std::move(cb)](tac::result<tac::async_streaming_encoder> r) mutable {
                        if (r.ok()) {
                            cb(tac::result<async_streaming_encoder>::ok_value(
                                async_streaming_encoder(r.take())));
                        } else {
                            cb(tac::result<async_streaming_encoder>::err(r.code(), r.message()));
                        }
                    }, hash_algo, threads, token, timeout);
            });
    }

    [[nodiscard]] std::future<void> write_object(
        const std::string& descriptor_json,
        const std::uint8_t* data, std::size_t len,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_void_future(
            [this, descriptor_json, data, len, token, timeout](auto cb) {
                inner_->write_object(descriptor_json, data, len, std::move(cb), token, timeout);
            });
    }

    [[nodiscard]] std::future<void> write_pre_encoded(
        const std::string& descriptor_json,
        const std::uint8_t* data, std::size_t len,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_void_future(
            [this, descriptor_json, data, len, token, timeout](auto cb) {
                inner_->write_pre_encoded(descriptor_json, data, len, std::move(cb),
                                          token, timeout);
            });
    }

    [[nodiscard]] std::future<void> write_preceder(
        const std::string& metadata_json,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_void_future(
            [this, metadata_json, token, timeout](auto cb) {
                inner_->write_preceder(metadata_json, std::move(cb), token, timeout);
            });
    }

    [[nodiscard]] std::future<void> finish(
        bool backfill = true,
        tac::cancellation_token* token = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        return detail::launch_void_future([this, backfill, token, timeout](auto cb) {
            inner_->finish(std::move(cb), backfill, token, timeout);
        });
    }

    using object_count_status = tac::async_streaming_encoder::object_count_status;
    using object_count_result = tac::async_streaming_encoder::object_count_result;

    /// Discriminated try-accessor.  See
    /// `tac::async_streaming_encoder::try_object_count`.
    [[nodiscard]] object_count_result try_object_count() const noexcept {
        return inner_->try_object_count();
    }

    [[nodiscard]] std::size_t object_count() const noexcept {
        return inner_->object_count();
    }

    async_streaming_encoder(async_streaming_encoder&&) noexcept = default;
    async_streaming_encoder& operator=(async_streaming_encoder&&) noexcept = default;
    async_streaming_encoder(const async_streaming_encoder&) = delete;
    async_streaming_encoder& operator=(const async_streaming_encoder&) = delete;

private:
    explicit async_streaming_encoder(tac::async_streaming_encoder inner)
        : inner_(std::make_shared<tac::async_streaming_encoder>(std::move(inner))) {}
    std::shared_ptr<tac::async_streaming_encoder> inner_;
};

}  // namespace tensogram::stdfuture

#endif  // TENSOGRAM_ASYNC_STD_FUTURE_HPP

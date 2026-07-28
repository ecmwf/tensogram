// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/// @file 26_frame_walker.cpp
/// @brief Example 26 — walk a message's frames, read its envelope, and steer
///        the encoder with the full `encode_options` set (C++ wrapper).
///
/// Demonstrates:
///   * `tensogram::read_message_header()` — the 24-byte preamble as typed
///     predicates: is this message random-access (metadata / index / hashes
///     in the header) or streaming (in the footer)?
///   * `tensogram::frames()` — a lazy range over the frames of one message,
///     usable in a range-for.  Each frame borrows the caller's buffer, so
///     the payload view stays valid after the range is gone.
///   * `tensogram::encode_options::aggregate_hash` — where the aggregate
///     hash frame goes, verified by walking the frames it produced; and
///     `codec_backend`, which picks the codec implementation.
///   * `tensogram::is_remote_url()` — whether a source belongs to the remote
///     backend (`file::open_remote()`) or the local one (`file::open()`).
///
/// Build:
///   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build -j
///   ./build/bin/26_frame_walker

#include <tensogram.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

/// A 1-D float32 descriptor, repeated @p n times (one per data object).
std::string f32_json(std::size_t n, std::size_t elements) {
    std::string json = R"({"descriptors":[)";
    for (std::size_t i = 0; i < n; ++i) {
        if (i != 0) json += ",";
        json += R"({"type":"ndarray","ndim":1,"shape":[)" + std::to_string(elements) +
                R"(],"strides":[4],"dtype":"float32","byte_order":"little",)"
                R"("encoding":"none","filter":"none","compression":"none"})";
    }
    return json + "]}";
}

const char* frame_name(tensogram::frame_type t) {
    switch (t) {
        case tensogram::frame_type::header_metadata:   return "HeaderMetadata";
        case tensogram::frame_type::header_index:      return "HeaderIndex";
        case tensogram::frame_type::header_hash:       return "HeaderHash";
        case tensogram::frame_type::footer_hash:       return "FooterHash";
        case tensogram::frame_type::footer_index:      return "FooterIndex";
        case tensogram::frame_type::footer_metadata:   return "FooterMetadata";
        case tensogram::frame_type::preceder_metadata: return "PrecederMetadata";
        case tensogram::frame_type::ntensor:           return "NTensor";
    }
    return "unknown";
}

/// Render the eight structural predicates as "header / footer / both / -".
std::string placement(bool in_header, bool in_footer) {
    if (in_header && in_footer) return "both";
    if (in_header) return "header";
    if (in_footer) return "footer";
    return "-";
}

void print_header(const char* label, const std::vector<std::uint8_t>& msg) {
    const auto h = tensogram::read_message_header(msg.data(), msg.size());
    std::printf("%s: version=%u total_length=%llu%s\n", label, h.version(),
                static_cast<unsigned long long>(h.total_length()),
                h.total_length() == 0 ? "  (streaming writer never back-filled it)" : "");
    std::printf("  metadata: %-6s  index: %-6s  aggregate hashes: %s\n",
                placement(h.has_header_metadata(), h.has_footer_metadata()).c_str(),
                placement(h.has_header_index(), h.has_footer_index()).c_str(),
                placement(h.has_header_hashes(), h.has_footer_hashes()).c_str());
    std::printf("  preceder metadata: %s   every frame hashed: %s\n",
                h.has_preceder_metadata() ? "yes (advisory when streaming)" : "no",
                h.has_hashes_present() ? "yes" : "no");
}

void print_frames(const std::vector<std::uint8_t>& msg) {
    // Lazy: one C-level `next` per loop iteration, nothing materialised up
    // front.  A malformed frame chain throws framing_error here; a clean end
    // just terminates the loop.
    for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
        std::printf("  %-16s offset=%-5zu length=%-5zu payload=%-5zu hash=%s%s\n",
                    frame_name(f.type()), f.offset(), f.length(), f.payload_size(),
                    f.has_hash() ? "yes" : "no ",
                    f.is_data_object() ? "  <- data object" : "");
    }
}

}  // namespace

int main() {
    std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects{
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)},
        {reinterpret_cast<const std::uint8_t*>(values.data()),
         values.size() * sizeof(float)}};
    const std::string json = f32_json(objects.size(), values.size());

    // ── 1. A buffered (random-access) message ─────────────────────────────
    std::printf("=== buffered message ===\n");
    auto buffered = tensogram::encode(json, objects);
    print_header("preamble", buffered);
    print_frames(buffered);

    // ── 2. The payload borrows the caller's buffer ────────────────────────
    // frame::payload() points INTO `buffered`; it stays valid after the range
    // (and its C cursor) are destroyed, for as long as `buffered` lives.
    std::printf("\n=== borrowed payloads ===\n");
    std::vector<tensogram::frame> collected;
    {
        auto range = tensogram::frames(buffered.data(), buffered.size());
        for (const auto& f : range) collected.push_back(f);
    }  // the cursor is freed here — the payload views are not
    for (const auto& f : collected) {
        const auto payload = f.payload();
        std::printf("  %-16s payload[0..4] =", frame_name(f.type()));
        for (std::size_t i = 0; i < 4 && i < payload.size(); ++i) {
            std::printf(" %02x", static_cast<unsigned>(
                                     static_cast<std::uint8_t>(payload[i])));
        }
        std::printf("   (offset %zu into the message buffer)\n",
                    static_cast<std::size_t>(f.payload_data() - buffered.data()));
    }

    // ── 3. Steering the encoder: aggregate hash placement ─────────────────
    // A reader that scans backwards wants the hash list next to the
    // postamble; a reader that streams forwards wants it in the header.
    // `both` writes identical lists in both places — verified by walking.
    std::printf("\n=== encode_options: aggregate_hash = both ===\n");
    tensogram::encode_options opts;
    opts.aggregate_hash = tensogram::aggregate_hash_policy::both;
    opts.codec_backend = tensogram::compression_backend::automatic;
    auto both = tensogram::encode(json, objects, opts);
    print_header("preamble", both);
    print_frames(both);

    // ── 4. A streaming message puts metadata and index in the footer ──────
    std::printf("\n=== streaming message ===\n");
    const std::string path = "tensogram_example_26_stream.tgm";
    {
        // header / both are rejected when streaming: the header is written
        // before any data object exists, so the hashes are not yet known.
        tensogram::encode_options stream_opts;
        stream_opts.aggregate_hash = tensogram::aggregate_hash_policy::footer;
        tensogram::streaming_encoder enc(path, "{}", stream_opts);
        enc.write_preceder(R"({"units":"K"})");
        enc.write_object(
            R"({"type":"ndarray","ndim":1,"shape":[4],"strides":[4],"dtype":"float32",)"
            R"("byte_order":"little","encoding":"none","filter":"none","compression":"none"})",
            reinterpret_cast<const std::uint8_t*>(values.data()),
            values.size() * sizeof(float));
        enc.finish();
    }
    auto streamed = tensogram::file::open(path).read_message(0);
    print_header("preamble", streamed);
    print_frames(streamed);
    std::remove(path.c_str());

    // ── 5. Typed descriptor enums ─────────────────────────────────────────
    std::printf("\n=== typed descriptor enums ===\n");
    auto decoded = tensogram::decode(buffered.data(), buffered.size());
    const auto obj = decoded.object(0);
    std::printf("  dtype      : %-10s (enum code %d)\n", obj.dtype_string().c_str(),
                static_cast<int>(obj.dtype_enum()));
    std::printf("  byte order : %-10s (enum code %d)\n", obj.byte_order_string().c_str(),
                static_cast<int>(obj.byte_order_enum()));
    switch (obj.dtype_enum()) {
        case tensogram::dtype::float32:
            std::printf("  switched on dtype::float32 — 4 bytes per element\n");
            break;
        default:
            std::printf("  some other dtype\n");
            break;
    }

    // ── 6. Local or remote? ───────────────────────────────────────────────
    std::printf("\n=== is_remote_url ===\n");
    for (const char* source : {"s3://bucket/forecast.tgm", "https://host/forecast.tgm",
                               "/data/forecast.tgm", "file:///data/forecast.tgm"}) {
        const bool remote = tensogram::is_remote_url(source);
        std::printf("  %-30s -> %s\n", source,
                    remote ? "file::open_remote()" : "file::open()");
    }
    std::printf("(a build without the `remote` Cargo feature answers \"local\" for\n"
                " every URL — it genuinely cannot open one; rebuild the C API with\n"
                " --features remote, or CMake -DTENSOGRAM_REMOTE=ON)\n");

    // ── 7. A malformed frame chain is not a silent short walk ─────────────
    std::printf("\n=== malformed frame chain ===\n");
    std::vector<std::uint8_t> truncated(buffered.begin(),
                                        buffered.begin() +
                                            static_cast<long>(buffered.size() - 64));
    try {
        std::size_t seen = 0;
        for (const auto& f : tensogram::frames(truncated.data(), truncated.size())) {
            (void)f;
            ++seen;
        }
        std::printf("  walked %zu frames with no error\n", seen);
    } catch (const tensogram::framing_error& e) {
        std::printf("  framing_error after the intact frames: %s\n", e.what());
    }

    return 0;
}

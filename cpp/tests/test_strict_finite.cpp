// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// C++ parity tests for encode_options::reject_nan / reject_inf.
// Cross-references:
//   - Rust:   rust/tensogram/tests/strict_finite.rs
//   - Python: python/tests/test_strict_finite.py
//   - TS:     typescript/tests/strict_finite.test.ts
//   - Memo:   plans/RESEARCH_NAN_HANDLING.md §4.1

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

std::string f32_descriptor_json(std::size_t count) {
    return R"({"version":2,"descriptors":[{"type":"ntensor","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[1],"dtype":"float32","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
}

std::string f64_descriptor_json(std::size_t count) {
    return R"({"version":2,"descriptors":[{"type":"ntensor","ndim":1,"shape":[)" +
           std::to_string(count) +
           R"(],"strides":[1],"dtype":"float64","byte_order":"little","encoding":"none","filter":"none","compression":"none"}]})";
}

std::vector<std::pair<const std::uint8_t*, std::size_t>>
bytes_pair(const std::vector<float>& v) {
    return {
        {reinterpret_cast<const std::uint8_t*>(v.data()),
         v.size() * sizeof(float)}
    };
}

std::vector<std::pair<const std::uint8_t*, std::size_t>>
bytes_pair(const std::vector<double>& v) {
    return {
        {reinterpret_cast<const std::uint8_t*>(v.data()),
         v.size() * sizeof(double)}
    };
}

}  // namespace

// ── Defaults preserve current behaviour ─────────────────────────────────

TEST(StrictFinite, DefaultAcceptsNanInFloat32) {
    std::vector<float> values = {1.0f, std::nanf(""), 3.0f};
    EXPECT_NO_THROW(
        tensogram::encode(f32_descriptor_json(values.size()), bytes_pair(values))
    );
}

TEST(StrictFinite, DefaultAcceptsInfInFloat64) {
    std::vector<double> values = {
        1.0,
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        4.0,
    };
    EXPECT_NO_THROW(
        tensogram::encode(f64_descriptor_json(values.size()), bytes_pair(values))
    );
}

// ── reject_nan rejects across float dtypes ─────────────────────────────

TEST(StrictFinite, RejectNanRejectsFloat32) {
    std::vector<float> values = {1.0f, 2.0f, std::nanf(""), 4.0f};
    tensogram::encode_options opts;
    opts.reject_nan = true;
    EXPECT_THROW(
        tensogram::encode(
            f32_descriptor_json(values.size()), bytes_pair(values), opts),
        tensogram::encoding_error
    );
}

TEST(StrictFinite, RejectNanFloat64ErrorMessageMentionsIndexAndDtype) {
    std::vector<double> values = {1.0, 2.0, 3.0, std::nan("")};
    tensogram::encode_options opts;
    opts.reject_nan = true;
    try {
        tensogram::encode(
            f64_descriptor_json(values.size()), bytes_pair(values), opts);
        FAIL() << "expected encoding_error";
    } catch (const tensogram::encoding_error& e) {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("NaN"), std::string::npos);
        EXPECT_NE(msg.find("element 3"), std::string::npos);
        EXPECT_NE(msg.find("float64"), std::string::npos);
    }
}

// ── reject_inf ─────────────────────────────────────────────────────────

TEST(StrictFinite, RejectInfRejectsPositiveInf) {
    std::vector<float> values = {
        1.0f,
        std::numeric_limits<float>::infinity(),
        3.0f,
    };
    tensogram::encode_options opts;
    opts.reject_inf = true;
    EXPECT_THROW(
        tensogram::encode(
            f32_descriptor_json(values.size()), bytes_pair(values), opts),
        tensogram::encoding_error
    );
}

TEST(StrictFinite, RejectInfRejectsNegativeInf) {
    std::vector<double> values = {1.0, -std::numeric_limits<double>::infinity()};
    tensogram::encode_options opts;
    opts.reject_inf = true;
    EXPECT_THROW(
        tensogram::encode(
            f64_descriptor_json(values.size()), bytes_pair(values), opts),
        tensogram::encoding_error
    );
}

// ── Orthogonality ──────────────────────────────────────────────────────

TEST(StrictFinite, RejectInfDoesNotRejectNan) {
    std::vector<float> values = {1.0f, std::nanf("")};
    tensogram::encode_options opts;
    opts.reject_inf = true;
    EXPECT_NO_THROW(
        tensogram::encode(
            f32_descriptor_json(values.size()), bytes_pair(values), opts)
    );
}

TEST(StrictFinite, RejectNanDoesNotRejectInf) {
    std::vector<float> values = {
        1.0f,
        std::numeric_limits<float>::infinity(),
    };
    tensogram::encode_options opts;
    opts.reject_nan = true;
    EXPECT_NO_THROW(
        tensogram::encode(
            f32_descriptor_json(values.size()), bytes_pair(values), opts)
    );
}

TEST(StrictFinite, BothFlagsCatchEither) {
    tensogram::encode_options opts;
    opts.reject_nan = true;
    opts.reject_inf = true;

    std::vector<float> nan_data = {1.0f, std::nanf("")};
    EXPECT_THROW(
        tensogram::encode(
            f32_descriptor_json(nan_data.size()), bytes_pair(nan_data), opts),
        tensogram::encoding_error
    );

    std::vector<float> inf_data = {
        1.0f,
        std::numeric_limits<float>::infinity(),
    };
    EXPECT_THROW(
        tensogram::encode(
            f32_descriptor_json(inf_data.size()), bytes_pair(inf_data), opts),
        tensogram::encoding_error
    );
}

// ── Edge cases ─────────────────────────────────────────────────────────

TEST(StrictFinite, NegativeZeroIsNotRejected) {
    std::vector<double> values = {1.0, -0.0, 2.0};
    tensogram::encode_options opts;
    opts.reject_nan = true;
    opts.reject_inf = true;
    EXPECT_NO_THROW(
        tensogram::encode(
            f64_descriptor_json(values.size()), bytes_pair(values), opts)
    );
}

TEST(StrictFinite, EmptyArrayPasses) {
    std::vector<float> values;
    tensogram::encode_options opts;
    opts.reject_nan = true;
    opts.reject_inf = true;
    EXPECT_NO_THROW(
        tensogram::encode(
            f32_descriptor_json(0), bytes_pair(values), opts)
    );
}

// ── streaming_encoder honours the flags ────────────────────────────────

TEST(StrictFinite, StreamingEncoderRejectsNan) {
    test_helpers::TempFile tf;
    std::vector<float> values = {1.0f, std::nanf("")};
    const std::string desc = R"({
        "type": "ntensor",
        "ndim": 1,
        "shape": [2],
        "strides": [1],
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none"
    })";
    tensogram::encode_options opts;
    opts.reject_nan = true;
    tensogram::streaming_encoder enc(tf.path, R"({"version":2})", opts);
    EXPECT_THROW(
        enc.write_object(desc,
                          reinterpret_cast<const std::uint8_t*>(values.data()),
                          values.size() * sizeof(float)),
        tensogram::encoding_error
    );
}

TEST(StrictFinite, StreamingEncoderDefaultAcceptsNan) {
    test_helpers::TempFile tf;
    std::vector<float> values = {1.0f, std::nanf("")};
    const std::string desc = R"({
        "type": "ntensor",
        "ndim": 1,
        "shape": [2],
        "strides": [1],
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none"
    })";
    tensogram::streaming_encoder enc(tf.path, R"({"version":2})");
    EXPECT_NO_THROW(
        enc.write_object(desc,
                          reinterpret_cast<const std::uint8_t*>(values.data()),
                          values.size() * sizeof(float))
    );
    enc.finish();
}

// ── file.append() honours the flags ────────────────────────────────────

TEST(StrictFinite, FileAppendRejectsNan) {
    test_helpers::TempFile tf;
    tensogram::file f = tensogram::file::create(tf.path);
    std::vector<float> values = {1.0f, std::nanf("")};
    tensogram::encode_options opts;
    opts.reject_nan = true;
    EXPECT_THROW(
        f.append(f32_descriptor_json(values.size()), bytes_pair(values), opts),
        tensogram::encoding_error
    );
}

// ── encode_pre_encoded rejects strict flags ────────────────────────────

TEST(StrictFinite, EncodePreEncodedErrorsWhenRejectNanIsSet) {
    // Mirrors the Rust and Python contracts: pre-encoded bytes are
    // opaque, so the strict flags cannot be meaningfully applied.
    // Setting them on encode_pre_encoded must throw rather than
    // silently discarding the caller's intent.
    std::vector<float> values = {1.0f, 2.0f};
    tensogram::encode_options opts;
    opts.reject_nan = true;
    try {
        tensogram::encode_pre_encoded(
            f32_descriptor_json(values.size()), bytes_pair(values), opts);
        FAIL() << "expected encoding_error";
    } catch (const tensogram::encoding_error& e) {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("reject_nan"), std::string::npos);
        EXPECT_NE(msg.find("encode_pre_encoded"), std::string::npos);
    }
}

TEST(StrictFinite, EncodePreEncodedErrorsWhenRejectInfIsSet) {
    std::vector<float> values = {1.0f, 2.0f};
    tensogram::encode_options opts;
    opts.reject_inf = true;
    EXPECT_THROW(
        tensogram::encode_pre_encoded(
            f32_descriptor_json(values.size()), bytes_pair(values), opts),
        tensogram::encoding_error
    );
}

TEST(StrictFinite, EncodePreEncodedAcceptsDefaultOptions) {
    // Regression: the check must not fire when flags are off (default).
    std::vector<float> values = {1.0f, std::nanf("")};
    EXPECT_NO_THROW(
        tensogram::encode_pre_encoded(
            f32_descriptor_json(values.size()), bytes_pair(values))
    );
}

// ── Interaction with simple_packing: §3.1 gotcha mitigation ───────────

TEST(StrictFinite, RejectInfBlocksSimplePackingSilentCorruption) {
    // Without reject_inf, [1.0, Inf, 3.0] through simple_packing silently
    // decodes to NaN everywhere (binary_scale_factor overflows). With the
    // flag on we catch it cleanly.
    std::vector<double> values = {
        1.0, std::numeric_limits<double>::infinity(), 3.0,
    };
    const std::string json = R"({
        "version": 2,
        "descriptors": [{
            "type": "ntensor",
            "ndim": 1,
            "shape": [3],
            "strides": [1],
            "dtype": "float64",
            "byte_order": "little",
            "encoding": "simple_packing",
            "filter": "none",
            "compression": "none",
            "bits_per_value": 16,
            "reference_value": 1.0,
            "binary_scale_factor": 0,
            "decimal_scale_factor": 0
        }]
    })";
    tensogram::encode_options opts;
    opts.reject_inf = true;
    EXPECT_THROW(
        tensogram::encode(json, bytes_pair(values), opts),
        tensogram::encoding_error
    );
}

// ── Standalone-API safety net — plans/RESEARCH_NAN_HANDLING.md §4.2.3 ────

namespace {

// Build a simple_packing descriptor JSON with hand-crafted params.
std::string simple_packing_json(
    double reference_value,
    long long binary_scale_factor,
    long long bits_per_value = 16
) {
    return std::string{R"({
        "version": 2,
        "descriptors": [{
            "type": "ntensor",
            "ndim": 1,
            "shape": [4],
            "strides": [1],
            "dtype": "float64",
            "byte_order": "little",
            "encoding": "simple_packing",
            "filter": "none",
            "compression": "none",
            "reference_value": )"} + std::to_string(reference_value) +
        R"(, "binary_scale_factor": )" + std::to_string(binary_scale_factor) +
        R"(, "decimal_scale_factor": 0, "bits_per_value": )" +
        std::to_string(bits_per_value) +
        R"(}]})";
}

}  // namespace

TEST(StrictFinite, SafetyNetRejectsHugeBinaryScaleFactor) {
    // i32::MAX is the fingerprint of feeding Inf through compute_params's
    // range arithmetic; the safety net in encode_with_threads catches it.
    std::vector<double> values = {273.15, 283.0, 293.0, 303.0};
    const std::string json = simple_packing_json(273.15, 2147483647LL);
    try {
        tensogram::encode(json, bytes_pair(values));
        FAIL() << "expected encoding_error";
    } catch (const tensogram::encoding_error& e) {
        std::string msg{e.what()};
        EXPECT_NE(msg.find("binary_scale_factor"), std::string::npos);
        EXPECT_NE(msg.find("256"), std::string::npos);
    }
}

TEST(StrictFinite, SafetyNetThresholdIs256) {
    // Threshold is inclusive: 256 accepted, 257 rejected.
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};

    EXPECT_NO_THROW(
        tensogram::encode(simple_packing_json(0.0, 256), bytes_pair(values))
    );
    EXPECT_THROW(
        tensogram::encode(simple_packing_json(0.0, 257), bytes_pair(values)),
        tensogram::encoding_error
    );
}

TEST(StrictFinite, SafetyNetAcceptsRealisticBinaryScaleFactors) {
    // Regression guard: real-world weather-data values must pass.
    std::vector<double> values = {273.15, 283.0, 293.0, 303.0};
    for (long long bsf : {-60LL, -20LL, 0LL, 20LL, 60LL}) {
        EXPECT_NO_THROW(
            tensogram::encode(simple_packing_json(273.15, bsf), bytes_pair(values))
        ) << "realistic bsf " << bsf << " should pass";
    }
}

TEST(StrictFinite, SafetyNetAcceptsConstantFieldEncoding) {
    // bits_per_value=0 is a legitimate constant-field encoding.
    // The safety net must not reject it.
    std::vector<double> values = {42.0, 42.0, 42.0, 42.0};
    const std::string json = simple_packing_json(42.0, 0, /*bits_per_value=*/0);
    EXPECT_NO_THROW(tensogram::encode(json, bytes_pair(values)));
}

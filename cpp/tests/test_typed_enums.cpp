// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 ECMWF
//
// Tests for the typed descriptor enums:
//   * tensogram::dtype       + decoded_object::dtype_enum()  (tgm_object_dtype_enum)
//   * tensogram::byte_order  + decoded_object::byte_order_enum()
//                                                 (tgm_object_byte_order_enum)
//
// The paired string getters (dtype_string() / byte_order_string()) stay the
// unambiguous bounds check: they return "" for exactly the inputs on which the
// C enum accessors fall back to their zero variant.

#include <gtest/gtest.h>
#include <tensogram.hpp>
#include "test_helpers.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

/// Encode one 1-D object of @p dtype_name, then decode it.
///
/// @p elem_bits is the element width in bits (1 for the sub-byte bitmask
/// dtype), @p count the number of elements.  Decoding keeps the wire byte
/// order so that a big-endian object is not silently swapped on the way out.
std::vector<std::uint8_t> encode_one(const std::string& dtype_name,
                                     std::size_t elem_bits,
                                     std::size_t count,
                                     const std::string& byte_order_name) {
    const std::size_t stride = elem_bits >= 8 ? elem_bits / 8 : 1;
    const std::size_t bytes = (elem_bits * count + 7) / 8;
    // encode() consumes the bytes during the call, so a local buffer is fine.
    const std::vector<std::uint8_t> data(bytes, 0x01);
    const std::string json =
        R"({"descriptors":[{"type":"ndarray","ndim":1,"shape":[)" + std::to_string(count) +
        R"(],"strides":[)" + std::to_string(stride) + R"(],"dtype":")" + dtype_name +
        R"(","byte_order":")" + byte_order_name +
        R"(","encoding":"none","filter":"none","compression":"none"}]})";
    std::vector<std::pair<const std::uint8_t*, std::size_t>> objects{{data.data(), data.size()}};
    return tensogram::encode(json, objects);
}

struct dtype_case {
    const char* name;
    std::size_t elem_bits;
    tensogram::dtype expected;
};

/// Every dtype the encoder accepts, paired with the C++ enumerator it must
/// map to.  A new core dtype without a C++ enumerator fails to compile here.
const std::vector<dtype_case>& all_dtypes() {
    using tensogram::dtype;
    static const std::vector<dtype_case> cases{
        {"float16", 16, dtype::float16},
        {"bfloat16", 16, dtype::bfloat16},
        {"float32", 32, dtype::float32},
        {"float64", 64, dtype::float64},
        {"complex64", 64, dtype::complex64},
        {"complex128", 128, dtype::complex128},
        {"int8", 8, dtype::int8},
        {"int16", 16, dtype::int16},
        {"int32", 32, dtype::int32},
        {"int64", 64, dtype::int64},
        {"uint8", 8, dtype::uint8},
        {"uint16", 16, dtype::uint16},
        {"uint32", 32, dtype::uint32},
        {"uint64", 64, dtype::uint64},
        {"bitmask", 1, dtype::bitmask},
    };
    return cases;
}

}  // namespace

// ---------------------------------------------------------------------------
// dtype
// ---------------------------------------------------------------------------

TEST(DtypeEnumTest, MirrorsTheCEnumValues) {
    using tensogram::dtype;
    EXPECT_EQ(static_cast<int>(dtype::float16), static_cast<int>(TGM_DTYPE_FLOAT16));
    EXPECT_EQ(static_cast<int>(dtype::bfloat16), static_cast<int>(TGM_DTYPE_BFLOAT16));
    EXPECT_EQ(static_cast<int>(dtype::float32), static_cast<int>(TGM_DTYPE_FLOAT32));
    EXPECT_EQ(static_cast<int>(dtype::float64), static_cast<int>(TGM_DTYPE_FLOAT64));
    EXPECT_EQ(static_cast<int>(dtype::complex64), static_cast<int>(TGM_DTYPE_COMPLEX64));
    EXPECT_EQ(static_cast<int>(dtype::complex128), static_cast<int>(TGM_DTYPE_COMPLEX128));
    EXPECT_EQ(static_cast<int>(dtype::int8), static_cast<int>(TGM_DTYPE_INT8));
    EXPECT_EQ(static_cast<int>(dtype::int16), static_cast<int>(TGM_DTYPE_INT16));
    EXPECT_EQ(static_cast<int>(dtype::int32), static_cast<int>(TGM_DTYPE_INT32));
    EXPECT_EQ(static_cast<int>(dtype::int64), static_cast<int>(TGM_DTYPE_INT64));
    EXPECT_EQ(static_cast<int>(dtype::uint8), static_cast<int>(TGM_DTYPE_UINT8));
    EXPECT_EQ(static_cast<int>(dtype::uint16), static_cast<int>(TGM_DTYPE_UINT16));
    EXPECT_EQ(static_cast<int>(dtype::uint32), static_cast<int>(TGM_DTYPE_UINT32));
    EXPECT_EQ(static_cast<int>(dtype::uint64), static_cast<int>(TGM_DTYPE_UINT64));
    EXPECT_EQ(static_cast<int>(dtype::bitmask), static_cast<int>(TGM_DTYPE_BITMASK));
}

TEST(DtypeEnumTest, AgreesWithTheStringGetterForEveryDtype) {
    tensogram::decode_options opts;
    opts.native_byte_order = false;  // keep the wire dtype layout untouched
    for (const auto& c : all_dtypes()) {
        auto encoded = encode_one(c.name, c.elem_bits, 8, "little");
        auto msg = tensogram::decode(encoded.data(), encoded.size(), opts);
        ASSERT_EQ(msg.num_objects(), 1u) << c.name;
        const auto obj = msg.object(0);
        EXPECT_EQ(obj.dtype_string(), c.name);
        EXPECT_EQ(obj.dtype_enum(), c.expected) << c.name;
    }
}

TEST(DtypeEnumTest, IsUsableInASwitch) {
    // The point of the typed accessor: switch instead of strcmp.
    auto encoded = encode_one("int32", 32, 4, "little");
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    std::size_t width = 0;
    switch (msg.object(0).dtype_enum()) {
        case tensogram::dtype::int32: width = 4; break;
        default: width = 0; break;
    }
    EXPECT_EQ(width, 4u);
}

TEST(DtypeEnumTest, OutOfRangeIndexThrowsInsteadOfReturningTheZeroVariant) {
    auto encoded = encode_one("float64", 64, 4, "little");
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    ASSERT_EQ(msg.num_objects(), 1u);
    // The C accessor has no spare code for failure — it returns the zero
    // variant (float16) and records the reason.  The C++ wrapper bounds-checks
    // through the paired string getter, which returns "" for exactly those
    // inputs, and raises instead of handing back a plausible-looking dtype.
    EXPECT_EQ(msg.object(1).dtype_string(), "");
    EXPECT_THROW((void)msg.object(1).dtype_enum(), tensogram::invalid_arg_error);
    EXPECT_THROW((void)msg.object(99).dtype_enum(), tensogram::invalid_arg_error);
}

// ---------------------------------------------------------------------------
// byte_order
// ---------------------------------------------------------------------------

TEST(ByteOrderEnumTest, MirrorsTheCEnumValues) {
    EXPECT_EQ(static_cast<int>(tensogram::byte_order::little),
              static_cast<int>(TGM_BYTE_ORDER_LITTLE));
    EXPECT_EQ(static_cast<int>(tensogram::byte_order::big),
              static_cast<int>(TGM_BYTE_ORDER_BIG));
}

TEST(ByteOrderEnumTest, AgreesWithTheStringGetter) {
    tensogram::decode_options opts;
    opts.native_byte_order = false;
    const std::pair<const char*, tensogram::byte_order> cases[]{
        {"little", tensogram::byte_order::little},
        {"big", tensogram::byte_order::big},
    };
    for (const auto& [name, expected] : cases) {
        auto encoded = encode_one("int16", 16, 4, name);
        auto msg = tensogram::decode(encoded.data(), encoded.size(), opts);
        ASSERT_EQ(msg.num_objects(), 1u);
        const auto obj = msg.object(0);
        EXPECT_EQ(obj.byte_order_string(), name);
        EXPECT_EQ(obj.byte_order_enum(), expected) << name;
    }
}

TEST(ByteOrderEnumTest, OutOfRangeIndexThrows) {
    auto encoded = encode_one("uint8", 8, 4, "little");
    auto msg = tensogram::decode(encoded.data(), encoded.size());
    EXPECT_EQ(msg.object(1).byte_order_string(), "");
    EXPECT_THROW((void)msg.object(1).byte_order_enum(), tensogram::invalid_arg_error);
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// Cross-language parity harness, C/C++ (row-major) side. Pairs with
// xlang_fortran.f90. Both agree on a logical field F(i,j) = i*1000 + j
// (1-based). The wire descriptor shape is the Fortran-reversed [NJ, NI]
// (C/row-major), so:
//   * a Fortran a(ni,nj) is read here, in C order, as flat[r*NI + c] = a(c+1, r+1);
//   * a C [NJ, NI] tensor written here is read in Fortran as out(ni,nj) = F.
// Uses lossless zstd compression; values are exact integers-as-floats.
//
// Modes: `cwrite <path>` (encode -> file), `cread <path>` (decode + check).

extern "C" {
#include <tensogram/tensogram.h>
}

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int NI = 5;
constexpr int NJ = 3;

float field(int i, int j) { return static_cast<float>(i * 1000 + j); }  // 1-based

const char* host_byte_order() {
    uint32_t x = 1;
    return (*reinterpret_cast<uint8_t*>(&x) == 1) ? "little" : "big";
}

int do_cwrite(const char* path) {
    // C-order [NJ, NI] tensor representing the same logical field.
    std::vector<float> data(static_cast<size_t>(NJ) * NI);
    for (int r = 0; r < NJ; ++r)
        for (int c = 0; c < NI; ++c)
            data[static_cast<size_t>(r) * NI + c] = field(c + 1, r + 1);

    const std::string meta =
        std::string("{\"descriptors\":[{\"type\":\"ntensor\",\"ndim\":2,") +
        "\"shape\":[" + std::to_string(NJ) + "," + std::to_string(NI) + "]," +
        "\"strides\":[" + std::to_string(NI) + ",1]," +
        "\"dtype\":\"float32\",\"byte_order\":\"" + host_byte_order() + "\"," +
        "\"encoding\":\"none\",\"filter\":\"none\",\"compression\":\"zstd\"}]}";

    const uint8_t* ptrs[1] = {reinterpret_cast<const uint8_t*>(data.data())};
    const size_t lens[1] = {data.size() * sizeof(float)};
    tgm_bytes_t enc = {nullptr, 0};
    if (tgm_encode(meta.c_str(), ptrs, lens, 1, "xxh3", 0, &enc) != TGM_ERROR_OK) {
        std::fprintf(stderr, "cwrite: encode failed: %s\n", tgm_last_error());
        return 1;
    }
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "cwrite: cannot open %s\n", path);
        tgm_bytes_free(enc);
        return 1;
    }
    std::fwrite(enc.data, 1, enc.len, f);
    std::fclose(f);
    std::printf("cwrite: wrote %zu bytes\n", enc.len);
    tgm_bytes_free(enc);
    return 0;
}

int do_cread(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cread: cannot open %s\n", path);
        return 1;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    const size_t got = std::fread(buf.data(), 1, static_cast<size_t>(n), f);
    std::fclose(f);
    if (got != static_cast<size_t>(n)) {
        std::fprintf(stderr, "cread: short read\n");
        return 1;
    }

    tgm_message_t* msg = nullptr;
    if (tgm_decode(buf.data(), buf.size(), /*native=*/1, /*threads=*/0,
                   /*verify=*/0, &msg) != TGM_ERROR_OK) {
        std::fprintf(stderr, "cread: decode failed: %s\n", tgm_last_error());
        return 1;
    }

    int rc = 0;
    if (tgm_message_num_objects(msg) != 1) {
        std::fprintf(stderr, "cread: expected 1 object\n");
        rc = 1;
    }
    if (!rc && tgm_object_ndim(msg, 0) != 2) {
        std::fprintf(stderr, "cread: expected ndim 2\n");
        rc = 1;
    }
    if (!rc) {
        const uint64_t* shp = tgm_object_shape(msg, 0);  // wire (C) order
        if (shp[0] != static_cast<uint64_t>(NJ) || shp[1] != static_cast<uint64_t>(NI)) {
            std::fprintf(stderr, "cread: shape [%llu,%llu] != [%d,%d]\n",
                         static_cast<unsigned long long>(shp[0]),
                         static_cast<unsigned long long>(shp[1]), NJ, NI);
            rc = 1;
        }
    }
    if (!rc) {
        size_t out_len = 0;
        const float* fd = reinterpret_cast<const float*>(tgm_object_data(msg, 0, &out_len));
        if (out_len != static_cast<size_t>(NJ) * NI * sizeof(float)) {
            std::fprintf(stderr, "cread: unexpected byte length %zu\n", out_len);
            rc = 1;
        }
        for (int r = 0; r < NJ && !rc; ++r)
            for (int c = 0; c < NI && !rc; ++c) {
                const float expected = field(c + 1, r + 1);
                const float actual = fd[static_cast<size_t>(r) * NI + c];
                if (actual != expected) {  // exact: integer-valued floats, lossless
                    std::fprintf(stderr, "cread: [%d,%d]=%g != %g\n", r, c, actual, expected);
                    rc = 1;
                }
            }
    }
    tgm_message_free(msg);
    if (rc == 0)
        std::printf("cread: PASS (Fortran a(ni,nj) seen as C [nj,ni] transpose)\n");
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xlang_c <cwrite|cread> <path>\n");
        return 2;
    }
    if (std::strcmp(argv[1], "cwrite") == 0) return do_cwrite(argv[2]);
    if (std::strcmp(argv[1], "cread") == 0) return do_cread(argv[2]);
    std::fprintf(stderr, "xlang_c: unknown mode %s\n", argv[1]);
    return 2;
}

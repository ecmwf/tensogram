cmake_minimum_required(VERSION 3.20)
project(tensogram-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build the Rust static library via cargo
# ---------------------------------------------------------------------------
set(CARGO_TARGET_DIR "${CMAKE_SOURCE_DIR}/../target")
set(RUST_LIB "${CARGO_TARGET_DIR}/release/libtensogram_ffi.a")

# Always invoke cargo. Cargo's own incremental cache decides whether to
# actually rebuild — sub-second when nothing changed, full rebuild when any
# Rust source moved. add_custom_target (without OUTPUT) avoids CMake's
# mtime "skip if output exists" heuristic, which can otherwise serve a
# days-old static library to today's test binary if the .a is unchanged on
# disk while the Rust source has moved on (cargo had no reason to retouch
# the file mtime, so CMake's check passes).
add_custom_target(rust_lib ALL
    BYPRODUCTS "${RUST_LIB}"
    COMMAND cargo build --release -p tensogram-ffi
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    COMMENT "Invoking cargo build -p tensogram-ffi (cargo decides freshness)"
)

# Import the Rust static library as a CMake target
add_library(tensogram_ffi STATIC IMPORTED GLOBAL)
set_target_properties(tensogram_ffi PROPERTIES
    IMPORTED_LOCATION "${RUST_LIB}")
add_dependencies(tensogram_ffi rust_lib)

# ---------------------------------------------------------------------------
# Header-only C++ wrapper library (INTERFACE target)
# ---------------------------------------------------------------------------
add_library(tensogram INTERFACE)
# `cpp/include`     for the header-only C++ wrapper `tensogram.hpp`.
# `rust/tensogram-ffi` for the cbindgen-generated `tensogram.h` (regenerated
# on every cargo build, single source of truth for the C API).
target_include_directories(tensogram INTERFACE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/../rust/tensogram-ffi")
target_link_libraries(tensogram INTERFACE tensogram_ffi)

# Platform-specific system libraries required by the Rust static library
if(APPLE)
    target_link_libraries(tensogram INTERFACE
        "-framework CoreFoundation"
        "-framework Security"
        "-framework SystemConfiguration"
        "-lc++"
        "-lm")
elseif(UNIX)
    target_link_libraries(tensogram INTERFACE
        dl pthread m stdc++)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
option(BUILD_EXAMPLES "Build C++ examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../examples/cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/examples_build"
    )
endif()

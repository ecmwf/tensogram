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

add_custom_command(
    OUTPUT "${RUST_LIB}"
    COMMAND cargo build --release -p tensogram-ffi
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    COMMENT "Building Rust tensogram-ffi static library"
)
add_custom_target(rust_lib ALL DEPENDS "${RUST_LIB}")

# Import the Rust static library as a CMake target
add_library(tensogram_ffi STATIC IMPORTED GLOBAL)
set_target_properties(tensogram_ffi PROPERTIES
    IMPORTED_LOCATION "${RUST_LIB}")
add_dependencies(tensogram_ffi rust_lib)

# ---------------------------------------------------------------------------
# Header-only C++ wrapper library (INTERFACE target)
# ---------------------------------------------------------------------------
add_library(tensogram INTERFACE)
target_include_directories(tensogram INTERFACE
    "${CMAKE_SOURCE_DIR}/include")
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

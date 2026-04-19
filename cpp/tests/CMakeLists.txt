include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(tensogram_tests
    test_encode_decode.cpp
    test_metadata.cpp
    test_descriptor.cpp
    test_file.cpp
    test_iterators.cpp
    test_error.cpp
    test_edge_cases.cpp
    test_simple_packing.cpp
    test_streaming.cpp
    test_multi_dtype.cpp
    test_encode_pre_encoded.cpp
    test_validate.cpp
    test_allow_nan_inf.cpp
)

target_include_directories(tensogram_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
    target_compile_options(tensogram_tests PRIVATE /W4)
else()
    target_compile_options(tensogram_tests PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

target_link_libraries(tensogram_tests PRIVATE
    tensogram
    GTest::gtest
    GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tensogram_tests)

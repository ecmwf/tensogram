include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TENSOGRAM_TEST_SOURCES
    test_encode_decode.cpp
    test_metadata.cpp
    test_meta_value.cpp
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
    test_decode_verify_hash.cpp
    test_doctor.cpp
)

if(TENSOGRAM_ASYNC)
    list(APPEND TENSOGRAM_TEST_SOURCES test_async_callback.cpp)
    list(APPEND TENSOGRAM_TEST_SOURCES test_async_stdfuture.cpp)
    list(APPEND TENSOGRAM_TEST_SOURCES test_async_producer_consumer.cpp)
    list(APPEND TENSOGRAM_TEST_SOURCES test_async_decode_object_range.cpp)
    list(APPEND TENSOGRAM_TEST_SOURCES test_async_task_control.cpp)
endif()

# Test executable
add_executable(tensogram_tests ${TENSOGRAM_TEST_SOURCES})

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

# Async runtime-shutdown test.  Built as its own executable because
# tgm_runtime_shutdown_blocking permanently tears down the process-global
# async runtime — running it in the shared `tensogram_tests` binary would
# poison the other async tests when the suite is run as a single process.
if(TENSOGRAM_ASYNC)
    add_executable(tensogram_async_shutdown_tests
        test_async_shutdown_during_flight.cpp)
    target_include_directories(tensogram_async_shutdown_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(tensogram_async_shutdown_tests PRIVATE /W4)
    else()
        target_compile_options(tensogram_async_shutdown_tests PRIVATE
            -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    endif()
    target_link_libraries(tensogram_async_shutdown_tests PRIVATE
        tensogram GTest::gtest GTest::gtest_main)
    gtest_discover_tests(tensogram_async_shutdown_tests)
endif()

# Optional C++20 coroutine test executable.  Compiled separately because
# the rest of the suite is C++17.  Set TENSOGRAM_ASYNC_CORO_TESTS=OFF to
# skip if your compiler doesn't support C++20 coroutines.
option(TENSOGRAM_ASYNC_CORO_TESTS "Build the C++20 coroutine async tests" ON)
if(TENSOGRAM_ASYNC AND TENSOGRAM_ASYNC_CORO_TESTS)
    add_executable(tensogram_async_coro_tests test_async_coro.cpp)
    set_target_properties(tensogram_async_coro_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
    target_compile_options(tensogram_async_coro_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    target_link_libraries(tensogram_async_coro_tests PRIVATE
        tensogram GTest::gtest GTest::gtest_main)
    gtest_discover_tests(tensogram_async_coro_tests)
endif()

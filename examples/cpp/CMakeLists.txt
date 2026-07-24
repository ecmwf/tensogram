# Synchronous examples (C++17, no async dependencies).
set(EXAMPLES
    01_encode_decode
    02_mars_metadata
    02b_generic_metadata
    03_simple_packing
    04_file_api
    05_iterators
    06_hash_and_errors
    07_scan_buffer
    08_decode_variants
    09_streaming_consumer
    10_doctor
    11_encode_pre_encoded
    13_validate
    16_multi_threaded_pipeline
)

# Async examples on the callback / std::future frontends (C++17, header-only).
# Available wherever the FFI is built with --features=async.
if(TENSOGRAM_ASYNC)
    list(APPEND EXAMPLES
        22_async_callback
        23_async_stdfuture
        24_async_cancellation
        25_async_decode_object_range
    )
endif()

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE tensogram)
    if(MSVC)
        target_compile_options(${example} PRIVATE /W4)
    else()
        target_compile_options(${example} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    set_target_properties(${example} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()

# Async examples on the C++20 coroutine frontend.  Compiled as separate
# targets at C++20 (the rest of the examples are C++17).  Disable with
# -DTENSOGRAM_ASYNC_CORO_EXAMPLES=OFF if your compiler lacks C++20 coroutines.
option(TENSOGRAM_ASYNC_CORO_EXAMPLES "Build the C++20 coroutine async examples" ON)
if(TENSOGRAM_ASYNC AND TENSOGRAM_ASYNC_CORO_EXAMPLES)
    set(CORO_EXAMPLES
        20_async_producer
        21_async_consumer
    )
    # The remote-decode example needs the object-store backend, which is
    # only linked in when the FFI is built with --features=async-remote.
    if(TENSOGRAM_ASYNC_REMOTE)
        list(APPEND CORO_EXAMPLES 19_async_decode_remote)
    endif()
    foreach(example ${CORO_EXAMPLES})
        add_executable(${example} ${example}.cpp)
        target_link_libraries(${example} PRIVATE tensogram)
        set_target_properties(${example} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        if(MSVC)
            target_compile_options(${example} PRIVATE /W4)
        else()
            target_compile_options(${example} PRIVATE
                -Wall -Wextra -Wpedantic -Wno-unused-parameter)
        endif()
    endforeach()
endif()

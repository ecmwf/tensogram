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
    11_encode_pre_encoded
    13_validate
    16_multi_threaded_pipeline
)

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

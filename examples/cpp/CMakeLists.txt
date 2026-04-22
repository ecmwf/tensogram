set(EXAMPLES
    01_encode_decode
    02_mars_metadata
    02b_generic_metadata
    03_simple_packing
    04_file_api
    05_iterators
    11_encode_pre_encoded
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

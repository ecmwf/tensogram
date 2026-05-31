# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

# CMake build for the Tensogram Fortran binding (the system of record;
# PLAN_FORTRAN.md §5.6). Links the system `libtensogram` discovered through
# pkg-config (the cargo-c / release-tarball install that ships tensogram.pc).
#
#   cmake -S fortran -B build/fortran
#   cmake --build build/fortran
#   ctest --test-dir build/fortran --output-on-failure
#
# For a non-default FFI prefix, point pkg-config at it first, e.g.:
#   export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"

cmake_minimum_required(VERSION 3.16)
project(tensogram_fortran VERSION 0.21.0 LANGUAGES Fortran)

option(TENSOGRAM_FORTRAN_EXAMPLES "Build the Fortran examples" ON)
option(TENSOGRAM_FORTRAN_TESTS    "Build the Fortran tests"    ON)

# Discover libtensogram (the C ABI) via pkg-config. `tensogram.pc` is shipped
# by `cargo cinstall -p tensogram-ffi` and by the release tarballs.
find_package(PkgConfig REQUIRED)
pkg_check_modules(TENSOGRAM REQUIRED IMPORTED_TARGET tensogram)

# Standard-conformance + warnings-as-errors on GNU (the supported compiler for
# the initial arc; PLAN_FORTRAN.md §10).
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-std=f2018 -Wall -Wextra -Werror)
endif()

# Emit .mod files into a known directory and expose it to consumers.
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod)

add_library(tensogram_f STATIC src/tensogram.f90)
target_link_libraries(tensogram_f PUBLIC PkgConfig::TENSOGRAM)
target_include_directories(tensogram_f PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
  $<INSTALL_INTERFACE:include>)

install(TARGETS tensogram_f ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod/
        DESTINATION include FILES_MATCHING PATTERN "*.mod")

if(TENSOGRAM_FORTRAN_EXAMPLES)
  add_executable(fortran_encode_decode
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/fortran/encode_decode.f90)
  target_link_libraries(fortran_encode_decode PRIVATE tensogram_f)

  add_executable(fortran_file_api
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/fortran/file_api.f90)
  target_link_libraries(fortran_file_api PRIVATE tensogram_f)
endif()

if(TENSOGRAM_FORTRAN_TESTS)
  enable_testing()

  # Positive tests: succeed (exit 0).
  foreach(t test_roundtrip test_dtype_rank test_file_api test_errors)
    add_executable(${t} test/${t}.f90)
    target_link_libraries(${t} PRIVATE tensogram_f)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()

  # Negative test: the non-copyable handle guard must abort (non-zero exit).
  # WILL_FAIL inverts the pass/fail sense, so a clean exit 0 (guard silently
  # gone) is reported as a regression. PLAN_FORTRAN.md §5.4.
  add_executable(test_copy_guard test/test_copy_guard.f90)
  target_link_libraries(test_copy_guard PRIVATE tensogram_f)
  add_test(NAME test_copy_guard COMMAND test_copy_guard)
  set_tests_properties(test_copy_guard PROPERTIES WILL_FAIL TRUE)

  # Cross-language parity (optional): Fortran encodes, Python decodes and
  # asserts the column-major contract. Skipped unless a Python with the
  # `tensogram` and `numpy` packages is discoverable. PLAN_FORTRAN.md §8.
  find_program(TENSOGRAM_PYTHON NAMES python3 python)
  if(TENSOGRAM_PYTHON)
    execute_process(
      COMMAND ${TENSOGRAM_PYTHON} -c "import tensogram, numpy"
      RESULT_VARIABLE _tgm_py_ok OUTPUT_QUIET ERROR_QUIET)
    if(_tgm_py_ok EQUAL 0)
      add_executable(parity_write test/parity_write.f90)
      target_link_libraries(parity_write PRIVATE tensogram_f)
      add_test(NAME parity_fortran_to_python COMMAND ${CMAKE_COMMAND}
        -DWRITER=$<TARGET_FILE:parity_write>
        -DPY=${TENSOGRAM_PYTHON}
        -DSCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/test/parity_check.py
        -DTGM=${CMAKE_CURRENT_BINARY_DIR}/parity.tgm
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_parity.cmake)
    else()
      message(STATUS "tensogram Python package not importable; skipping Fortran<->Python parity test")
    endif()
  endif()
endif()

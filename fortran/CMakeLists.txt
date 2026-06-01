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
# the initial arc; PLAN_FORTRAN.md §10). Scoped to Fortran so the optional
# C++ cross-language harness (below) is not given Fortran flags.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:Fortran>:-std=f2018>"
    "$<$<COMPILE_LANGUAGE:Fortran>:-Wall>"
    "$<$<COMPILE_LANGUAGE:Fortran>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:Fortran>:-Werror>")
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

  add_executable(fortran_streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/fortran/streaming.f90)
  target_link_libraries(fortran_streaming PRIVATE tensogram_f)
endif()

if(TENSOGRAM_FORTRAN_TESTS)
  enable_testing()

  # Positive tests: succeed (exit 0).
  foreach(t test_roundtrip test_dtype_rank test_file_api test_metadata test_streaming test_edge_cases test_errors)
    add_executable(${t} test/${t}.f90)
    target_link_libraries(${t} PRIVATE tensogram_f)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()

  # Run the examples as smoke tests when they are built — otherwise they are
  # only compile-checked, never executed. Each must exit 0.
  if(TENSOGRAM_FORTRAN_EXAMPLES)
    foreach(ex fortran_encode_decode fortran_file_api fortran_streaming)
      add_test(NAME smoke_${ex} COMMAND ${ex})
    endforeach()
  endif()

  # Negative test: the non-copyable handle guard must abort (non-zero exit).
  # WILL_FAIL inverts the pass/fail sense, so a clean exit 0 (guard silently
  # gone) is reported as a regression. PLAN_FORTRAN.md §5.4.
  add_executable(test_copy_guard test/test_copy_guard.f90)
  target_link_libraries(test_copy_guard PRIVATE tensogram_f)
  add_test(NAME test_copy_guard COMMAND test_copy_guard)
  set_tests_properties(test_copy_guard PROPERTIES WILL_FAIL TRUE)

  # Canonical Fortran side of the cross-language parity harness
  # (`xlang_fortran fwrite` / `fread`). Pure Fortran, so it is built whether
  # or not a C++ compiler is present, and is reused by both the Python and
  # C/C++ parity directions below. The shared driver run_xlang.cmake runs a
  # writer then a reader; both agree on F(i,j)=i*1000+j and a lossless (zstd)
  # pipeline. PLAN_FORTRAN.md §8.
  add_executable(xlang_fortran test/xlang_fortran.f90)
  target_link_libraries(xlang_fortran PRIVATE tensogram_f)

  # Cross-language parity with Python (NumPy, row-major), BOTH directions.
  # Skipped unless a Python with `tensogram` + `numpy` is discoverable.
  find_program(TENSOGRAM_PYTHON NAMES python3 python)
  if(TENSOGRAM_PYTHON)
    execute_process(
      COMMAND ${TENSOGRAM_PYTHON} -c "import tensogram, numpy"
      RESULT_VARIABLE _tgm_py_ok OUTPUT_QUIET ERROR_QUIET)
    if(_tgm_py_ok EQUAL 0)
      add_test(NAME parity_fortran_to_python COMMAND ${CMAKE_COMMAND}
        -DWRITER=$<TARGET_FILE:xlang_fortran> -DWMODE=fwrite
        -DREADER=${TENSOGRAM_PYTHON} -DRMODE=${CMAKE_CURRENT_SOURCE_DIR}/test/parity_check.py
        -DTGM=${CMAKE_CURRENT_BINARY_DIR}/parity_f2p.tgm
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_xlang.cmake)
      add_test(NAME parity_python_to_fortran COMMAND ${CMAKE_COMMAND}
        -DWRITER=${TENSOGRAM_PYTHON} -DWMODE=${CMAKE_CURRENT_SOURCE_DIR}/test/parity_py_write.py
        -DREADER=$<TARGET_FILE:xlang_fortran> -DRMODE=fread
        -DTGM=${CMAKE_CURRENT_BINARY_DIR}/parity_p2f.tgm
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_xlang.cmake)
    else()
      message(STATUS "tensogram Python package not importable; skipping Fortran<->Python parity tests")
    endif()
  endif()

  # Cross-language parity with C/C++ (row-major), BOTH directions. Skipped if
  # no C++ compiler is available.
  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    add_executable(xlang_c test/xlang_c.cpp)
    target_link_libraries(xlang_c PRIVATE PkgConfig::TENSOGRAM)

    add_test(NAME xlang_fortran_to_c COMMAND ${CMAKE_COMMAND}
      -DWRITER=$<TARGET_FILE:xlang_fortran> -DWMODE=fwrite
      -DREADER=$<TARGET_FILE:xlang_c>       -DRMODE=cread
      -DTGM=${CMAKE_CURRENT_BINARY_DIR}/xlang_f2c.tgm
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_xlang.cmake)

    add_test(NAME xlang_c_to_fortran COMMAND ${CMAKE_COMMAND}
      -DWRITER=$<TARGET_FILE:xlang_c>       -DWMODE=cwrite
      -DREADER=$<TARGET_FILE:xlang_fortran> -DRMODE=fread
      -DTGM=${CMAKE_CURRENT_BINARY_DIR}/xlang_c2f.tgm
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_xlang.cmake)
  else()
    message(STATUS "No C++ compiler; skipping Fortran<->C++ cross-language tests")
  endif()

  # Single-source-of-truth guard: the Fortran TGM_ERROR_* parameter mirror
  # must match the C `tgm_error` enum in the in-tree tensogram.h (AGENTS.md).
  # Skipped when the in-tree header is not reachable (standalone checkout) or
  # bash is unavailable.
  set(_tgm_header ${CMAKE_CURRENT_SOURCE_DIR}/../rust/tensogram-ffi/tensogram.h)
  find_program(BASH_PROGRAM bash)
  if(EXISTS ${_tgm_header} AND BASH_PROGRAM)
    add_test(NAME error_enum_consistency
      COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/test/check_error_enum.sh
              ${CMAKE_CURRENT_SOURCE_DIR}/src/tensogram.f90 ${_tgm_header})
  endif()
endif()

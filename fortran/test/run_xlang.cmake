# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

# CTest driver for a cross-language round-trip: run WRITER WMODE TGM, then
# READER RMODE TGM; fail if either step returns non-zero. Used for both
# directions (Fortran->C and C->Fortran).

execute_process(COMMAND ${WRITER} ${WMODE} ${TGM} RESULT_VARIABLE rc_write)
if(NOT rc_write EQUAL 0)
  message(FATAL_ERROR "writer (${WMODE}) failed with code ${rc_write}")
endif()

execute_process(COMMAND ${READER} ${RMODE} ${TGM} RESULT_VARIABLE rc_read)
if(NOT rc_read EQUAL 0)
  message(FATAL_ERROR "reader (${RMODE}) failed with code ${rc_read}")
endif()

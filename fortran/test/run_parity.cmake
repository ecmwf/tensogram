# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

# CTest driver: run the Fortran parity writer, then the Python checker.
# Fails the test if either step returns non-zero. Invoked with -D defines for
# WRITER (Fortran exe), PY (python interpreter), SCRIPT (parity_check.py),
# and TGM (the intermediate .tgm path).

execute_process(COMMAND ${WRITER} ${TGM} RESULT_VARIABLE rc_write)
if(NOT rc_write EQUAL 0)
  message(FATAL_ERROR "parity_write failed with code ${rc_write}")
endif()

execute_process(COMMAND ${PY} ${SCRIPT} ${TGM} RESULT_VARIABLE rc_check)
if(NOT rc_check EQUAL 0)
  message(FATAL_ERROR "parity_check.py failed with code ${rc_check}")
endif()

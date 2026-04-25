/* (C) Copyright 2026- ECMWF and individual contributors.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/* Expose the libaec compile-time version macro as a linkable C symbol so that
 * Rust code can read the version of the libaec library that was used when
 * building tensogram-encodings.  The macro AEC_VERSION_STR is defined in the
 * libaec public header <libaec.h> (built from source by libaec-sys).
 */

#include <libaec.h>

const char *tensogram_libaec_version(void) {
    return AEC_VERSION_STR;
}

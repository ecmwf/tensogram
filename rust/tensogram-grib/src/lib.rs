// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! GRIB to Tensogram format converter.
//!
//! Uses ECMWF's ecCodes library (via the `eccodes` Rust crate) to read GRIB
//! messages and convert them to Tensogram wire format.
//!
//! # System requirement
//!
//! The ecCodes C library must be installed:
//! ```bash
//! brew install eccodes       # macOS
//! apt install libeccodes-dev # Debian/Ubuntu
//! ```

pub mod converter;
pub mod error;
pub mod metadata;

pub use converter::{convert_grib_file, ConvertOptions, Grouping};
pub use error::GribError;
// `DataPipeline` lives in `tensogram-core::pipeline` — both GRIB and
// NetCDF converters share the same type so they cannot drift. Re-export
// it here so existing `use tensogram_grib::DataPipeline` callers keep
// compiling.
pub use tensogram_core::DataPipeline;

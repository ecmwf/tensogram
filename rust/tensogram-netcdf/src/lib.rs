// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! NetCDF to Tensogram format converter.
//!
//! Reads NetCDF files (classic and NetCDF-4) and converts variables to
//! Tensogram wire format messages. One-way conversion only (read NetCDF,
//! write Tensogram).
//!
//! # System requirement
//!
//! The NetCDF C library must be installed:
//! ```bash
//! brew install netcdf       # macOS
//! apt install libnetcdf-dev # Debian/Ubuntu
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use tensogram_netcdf::{convert_netcdf_file, ConvertOptions};
//!
//! let options = ConvertOptions::default();
//! let messages = convert_netcdf_file(Path::new("data.nc"), &options).unwrap();
//! ```

pub mod converter;
pub mod error;
pub mod metadata;

pub use converter::{ConvertOptions, SplitBy, convert_netcdf_file};
pub use error::NetcdfError;
// `DataPipeline` lives in `tensogram::pipeline` — both GRIB and
// NetCDF converters share the same type so they cannot drift. Re-export
// it here so existing `use tensogram_netcdf::DataPipeline` callers keep
// compiling.
pub use tensogram::DataPipeline;

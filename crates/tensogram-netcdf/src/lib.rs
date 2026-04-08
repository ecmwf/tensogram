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

pub use converter::{convert_netcdf_file, ConvertOptions, DataPipeline, SplitBy};
pub use error::NetcdfError;

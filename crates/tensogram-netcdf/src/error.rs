use thiserror::Error;

/// Errors produced during NetCDF to Tensogram conversion.
#[derive(Debug, Error)]
pub enum NetcdfError {
    /// Error from the underlying netcdf C library.
    #[error("netcdf error: {0}")]
    Netcdf(#[from] netcdf::Error),

    /// No variables found in the input file.
    #[error("no variables found in input")]
    NoVariables,

    /// Encode error from tensogram-core.
    #[error("encode error: {0}")]
    Encode(String),

    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid or unsupported data in the input file.
    #[error("invalid netcdf data: {0}")]
    InvalidData(String),

    /// Unsupported data type (e.g. string, vlen, compound).
    #[error("unsupported variable type for '{name}': {reason}")]
    UnsupportedType { name: String, reason: String },

    /// --split-by=record used on a file with no unlimited dimension.
    #[error("--split-by=record requires an unlimited dimension, none found in {file}")]
    NoUnlimitedDimension { file: String },
}

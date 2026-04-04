use thiserror::Error;

#[derive(Debug, Error)]
pub enum GribError {
    #[error("ecCodes error: {0}")]
    EcCodes(#[from] eccodes::errors::CodesError),

    #[error("no GRIB messages found in input")]
    NoMessages,

    #[error("encode error: {0}")]
    Encode(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid GRIB data: {0}")]
    InvalidData(String),
}

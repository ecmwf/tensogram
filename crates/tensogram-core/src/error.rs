use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensogramError {
    #[error("framing error: {0}")]
    Framing(String),
    #[error("metadata error: {0}")]
    Metadata(String),
    #[error("encoding error: {0}")]
    Encoding(String),
    #[error("compression error: {0}")]
    Compression(String),
    #[error("object error: {0}")]
    Object(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
}

pub type Result<T> = std::result::Result<T, TensogramError>;

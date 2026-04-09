//! Error types for the pure-Rust AEC/SZIP codec.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AecError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("data error: {0}")]
    Data(String),
    #[error("buffer too small: {0}")]
    Buffer(String),
}

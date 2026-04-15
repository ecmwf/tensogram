// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

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
    #[error("remote error: {0}")]
    Remote(String),
}

pub type Result<T> = std::result::Result<T, TensogramError>;

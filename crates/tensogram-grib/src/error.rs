// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

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

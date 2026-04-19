// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

pub mod bitmask;
pub mod compression;
#[cfg(feature = "szip")]
pub mod libaec;
pub mod pipeline;
pub mod shuffle;
pub mod simple_packing;
#[cfg(feature = "zfp")]
pub mod zfp_ffi;

pub use pipeline::{
    ByteOrder, CompressionBackend, CompressionType, EncodingType, FilterType, PipelineConfig,
    PipelineResult,
};

#[cfg(feature = "blosc2")]
pub use pipeline::Blosc2Codec;
#[cfg(feature = "sz3")]
pub use pipeline::Sz3ErrorBound;
#[cfg(feature = "zfp")]
pub use pipeline::ZfpMode;

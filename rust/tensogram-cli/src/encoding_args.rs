// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use clap::Args;

/// Encoding/filter/compression flags shared by all converter commands.
///
/// Use `#[command(flatten)]` to embed these in a converter's argument struct.
///
/// Every flag that accepts a fixed set of values uses clap's
/// `PossibleValuesParser` so invalid values fail at argument-parse time
/// with a helpful "did you mean?" suggestion, rather than propagating
/// into the converter as an `InvalidData` error at run time.
#[derive(Debug, Clone, Args)]
pub struct PipelineArgs {
    /// Encoding to apply before storage: none (default) or simple_packing.
    #[arg(
        long,
        default_value = "none",
        value_name = "ENC",
        value_parser = clap::builder::PossibleValuesParser::new(["none", "simple_packing"])
    )]
    pub encoding: String,

    /// Bits per value for simple_packing encoding.  Default
    /// `tensogram::pipeline::DEFAULT_SIMPLE_PACKING_BITS` (16) when
    /// `--encoding simple_packing` is set without `--bits`.
    #[arg(long, value_name = "N")]
    pub bits: Option<u32>,

    /// Byte-shuffle filter: none (default) or shuffle.
    #[arg(
        long,
        default_value = "none",
        value_name = "FILTER",
        value_parser = clap::builder::PossibleValuesParser::new(["none", "shuffle"])
    )]
    pub filter: String,

    /// Compression codec: none (default), szip, zstd, lz4, or blosc2.
    /// Codec must be compiled in; otherwise an error is returned at conversion time.
    #[arg(
        long,
        default_value = "none",
        value_name = "CODEC",
        value_parser = clap::builder::PossibleValuesParser::new([
            "none",
            "szip",
            "zstd",
            "lz4",
            "blosc2",
        ])
    )]
    pub compression: String,

    /// Compression level (used by zstd and blosc2; ignored by other codecs).
    #[arg(long, value_name = "LEVEL")]
    pub compression_level: Option<i32>,
}

impl Default for PipelineArgs {
    fn default() -> Self {
        Self {
            encoding: "none".to_string(),
            bits: None,
            filter: "none".to_string(),
            compression: "none".to_string(),
            compression_level: None,
        }
    }
}

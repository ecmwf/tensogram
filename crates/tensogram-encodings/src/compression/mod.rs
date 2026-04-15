// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

#[cfg(feature = "blosc2")]
mod blosc2;
#[cfg(feature = "lz4")]
mod lz4;
#[cfg(feature = "sz3")]
mod sz3;
#[cfg(feature = "szip")]
mod szip;
#[cfg(feature = "szip-pure")]
mod szip_pure;
#[cfg(feature = "zfp")]
mod zfp;
#[cfg(feature = "zstd")]
mod zstd;
#[cfg(feature = "zstd-pure")]
mod zstd_pure;

// Both FFI and pure-Rust backends can be compiled together; the caller
// selects which one to use at runtime via `CompressionBackend` in the
// `PipelineConfig`.  See `pipeline::build_compressor()`.

#[cfg(feature = "blosc2")]
pub use self::blosc2::Blosc2Compressor;
#[cfg(feature = "lz4")]
pub use self::lz4::Lz4Compressor;
#[cfg(feature = "sz3")]
pub use self::sz3::Sz3Compressor;
#[cfg(feature = "szip")]
pub use self::szip::SzipCompressor;
#[cfg(feature = "szip-pure")]
pub use self::szip_pure::SzipPureCompressor;
#[cfg(feature = "zfp")]
pub use self::zfp::ZfpCompressor;
#[cfg(feature = "zstd")]
pub use self::zstd::ZstdCompressor;
#[cfg(feature = "zstd-pure")]
pub use self::zstd_pure::ZstdPureCompressor;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("szip error: {0}")]
    Szip(String),
    #[error("zstd error: {0}")]
    Zstd(String),
    #[error("lz4 error: {0}")]
    Lz4(String),
    #[error("blosc2 error: {0}")]
    Blosc2(String),
    #[error("zfp error: {0}")]
    Zfp(String),
    #[error("sz3 error: {0}")]
    Sz3(String),
    #[error("range decode not supported for this compressor")]
    RangeNotSupported,
    #[error("unknown compression: {0}")]
    Unknown(String),
    #[error("compression not available: {0} (feature not enabled at compile time)")]
    NotAvailable(String),
}

pub struct CompressResult {
    pub data: Vec<u8>,
    pub block_offsets: Option<Vec<u64>>,
}

pub trait Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError>;
    fn decompress(&self, data: &[u8], expected_size: usize) -> Result<Vec<u8>, CompressionError>;
    fn decompress_range(
        &self,
        data: &[u8],
        block_offsets: &[u64],
        byte_pos: usize,
        byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError>;
}

pub struct NoopCompressor;

impl Compressor for NoopCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        Ok(CompressResult {
            data: data.to_vec(),
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }

    fn decompress_range(
        &self,
        data: &[u8],
        _block_offsets: &[u64],
        byte_pos: usize,
        byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        let end = byte_pos
            .checked_add(byte_size)
            .ok_or_else(|| CompressionError::Szip("byte range overflow".to_string()))?;
        if end > data.len() {
            return Err(CompressionError::Szip(format!(
                "range ({byte_pos}, {byte_size}) exceeds data length {}",
                data.len()
            )));
        }
        Ok(data[byte_pos..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_compressor_range_decode() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;

        let partial = compressor.decompress_range(&data, &[], 10, 20).unwrap();
        assert_eq!(&partial[..], &data[10..30]);
    }

    #[test]
    fn noop_compressor_range_out_of_bounds() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;
        assert!(compressor.decompress_range(&data, &[], 90, 20).is_err());
    }

    // ── NoopCompressor compress roundtrip ────────────────────────────────

    #[test]
    fn noop_compressor_compress_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressor = NoopCompressor;

        let result = compressor.compress(&data).unwrap();
        assert_eq!(result.data, data);
        assert!(result.block_offsets.is_none());
    }

    #[test]
    fn noop_compressor_decompress() {
        let data: Vec<u8> = (0..64).collect();
        let compressor = NoopCompressor;

        let decompressed = compressor.decompress(&data, 64).unwrap();
        assert_eq!(decompressed, data);
    }

    // ── NoopCompressor edge cases ────────────────────────────────────────

    #[test]
    fn noop_compressor_empty_data() {
        let compressor = NoopCompressor;

        let result = compressor.compress(&[]).unwrap();
        assert!(result.data.is_empty());

        let decompressed = compressor.decompress(&[], 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn noop_compressor_range_at_start() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;

        let partial = compressor.decompress_range(&data, &[], 0, 10).unwrap();
        assert_eq!(&partial[..], &data[0..10]);
    }

    #[test]
    fn noop_compressor_range_at_end() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;

        let partial = compressor.decompress_range(&data, &[], 90, 10).unwrap();
        assert_eq!(&partial[..], &data[90..100]);
    }

    #[test]
    fn noop_compressor_range_exact_end() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;

        // Exactly at the boundary — should succeed
        let partial = compressor.decompress_range(&data, &[], 0, 100).unwrap();
        assert_eq!(partial, data);
    }

    #[test]
    fn noop_compressor_range_one_past_end() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;

        // One byte past the end
        let result = compressor.decompress_range(&data, &[], 0, 101);
        assert!(result.is_err());
    }

    #[test]
    fn noop_compressor_range_overflow() {
        let data: Vec<u8> = (0..100).collect();
        let compressor = NoopCompressor;

        // byte_pos + byte_size would overflow usize
        let result = compressor.decompress_range(&data, &[], usize::MAX, 1);
        assert!(result.is_err(), "overflow should produce error");
    }

    // ── CompressionError Display coverage ────────────────────────────────

    #[test]
    fn compression_error_display() {
        let err = CompressionError::Szip("test szip error".to_string());
        assert!(err.to_string().contains("szip error"));

        let err = CompressionError::Zstd("test zstd error".to_string());
        assert!(err.to_string().contains("zstd error"));

        let err = CompressionError::Lz4("test lz4 error".to_string());
        assert!(err.to_string().contains("lz4 error"));

        let err = CompressionError::Blosc2("test blosc2 error".to_string());
        assert!(err.to_string().contains("blosc2 error"));

        let err = CompressionError::Zfp("test zfp error".to_string());
        assert!(err.to_string().contains("zfp error"));

        let err = CompressionError::Sz3("test sz3 error".to_string());
        assert!(err.to_string().contains("sz3 error"));

        let err = CompressionError::RangeNotSupported;
        assert!(err.to_string().contains("range decode not supported"));

        let err = CompressionError::Unknown("mystery".to_string());
        assert!(err.to_string().contains("unknown compression"));
        assert!(err.to_string().contains("mystery"));

        let err = CompressionError::NotAvailable("szip".to_string());
        assert!(err.to_string().contains("not available"));
        assert!(err.to_string().contains("szip"));
    }
}

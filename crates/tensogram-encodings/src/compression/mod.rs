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

// Mutual exclusion guards — exactly one backend per codec
#[cfg(all(feature = "szip", feature = "szip-pure"))]
compile_error!("features 'szip' and 'szip-pure' are mutually exclusive — choose one");

#[cfg(all(feature = "zstd", feature = "zstd-pure"))]
compile_error!("features 'zstd' and 'zstd-pure' are mutually exclusive — choose one");

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
}

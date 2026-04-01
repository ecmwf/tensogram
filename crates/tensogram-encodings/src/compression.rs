use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("szip compression not available (libaec not linked)")]
    SzipNotAvailable,
    #[error("unknown compression: {0}")]
    Unknown(String),
}

pub struct CompressResult {
    pub data: Vec<u8>,
    pub block_offsets: Option<Vec<u64>>,
}

pub trait Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError>;
    fn decompress(&self, data: &[u8], expected_size: usize) -> Result<Vec<u8>, CompressionError>;
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
}

pub struct SzipCompressor {
    pub rsi: u32,
    pub block_size: u32,
    pub flags: u32,
}

impl Compressor for SzipCompressor {
    fn compress(&self, _data: &[u8]) -> Result<CompressResult, CompressionError> {
        Err(CompressionError::SzipNotAvailable)
    }

    fn decompress(&self, _data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::SzipNotAvailable)
    }
}

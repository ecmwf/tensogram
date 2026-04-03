use crate::libaec::{self, AecParams};

use super::{CompressResult, CompressionError, Compressor};

pub struct SzipCompressor {
    pub rsi: u32,
    pub block_size: u32,
    pub flags: u32,
    pub bits_per_sample: u32,
}

impl Compressor for SzipCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        let params = self.aec_params();
        let (compressed, offsets) = libaec::aec_compress(data, &params)?;
        Ok(CompressResult {
            data: compressed,
            block_offsets: Some(offsets),
        })
    }

    fn decompress(&self, data: &[u8], expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        let params = self.aec_params();
        libaec::aec_decompress(data, expected_size, &params)
    }

    fn decompress_range(
        &self,
        data: &[u8],
        block_offsets: &[u64],
        byte_pos: usize,
        byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        let params = self.aec_params();
        libaec::aec_decompress_range(data, block_offsets, byte_pos, byte_size, &params)
    }
}

impl SzipCompressor {
    fn aec_params(&self) -> AecParams {
        AecParams {
            bits_per_sample: self.bits_per_sample,
            block_size: self.block_size,
            rsi: self.rsi,
            flags: self.flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn szip_compressor_round_trip() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let compressor = SzipCompressor {
            rsi: 128,
            block_size: 16,
            flags: libaec_sys::AEC_DATA_PREPROCESS,
            bits_per_sample: 8,
        };

        let result = compressor.compress(&data).unwrap();
        assert!(result.block_offsets.is_some());

        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn szip_compressor_range_decode() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = SzipCompressor {
            rsi: 128,
            block_size: 16,
            flags: libaec_sys::AEC_DATA_PREPROCESS,
            bits_per_sample: 8,
        };

        let result = compressor.compress(&data).unwrap();
        let offsets = result.block_offsets.as_ref().unwrap();

        let partial = compressor
            .decompress_range(&result.data, offsets, 200, 500)
            .unwrap();
        assert_eq!(partial.len(), 500);
        assert_eq!(&partial[..], &data[200..700]);
    }
}

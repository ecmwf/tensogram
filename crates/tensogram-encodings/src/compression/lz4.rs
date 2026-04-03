use super::{CompressResult, CompressionError, Compressor};

pub struct Lz4Compressor;

impl Compressor for Lz4Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        let compressed = lz4_flex::compress_prepend_size(data);
        Ok(CompressResult {
            data: compressed,
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        lz4_flex::decompress_size_prepended(data).map_err(|e| CompressionError::Lz4(e.to_string()))
    }

    fn decompress_range(
        &self,
        _data: &[u8],
        _block_offsets: &[u64],
        _byte_pos: usize,
        _byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::RangeNotSupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_round_trip() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Lz4Compressor;

        let result = compressor.compress(&data).unwrap();
        assert!(result.block_offsets.is_none());

        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_range_not_supported() {
        let compressor = Lz4Compressor;
        let result = compressor.decompress_range(&[0], &[], 0, 1);
        assert!(matches!(result, Err(CompressionError::RangeNotSupported)));
    }
}

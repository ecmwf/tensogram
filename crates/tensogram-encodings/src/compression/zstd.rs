// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use super::{CompressResult, CompressionError, Compressor};

pub struct ZstdCompressor {
    pub level: i32,
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        let compressed = zstd::encode_all(data, self.level)
            .map_err(|e| CompressionError::Zstd(e.to_string()))?;
        Ok(CompressResult {
            data: compressed,
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        zstd::decode_all(data).map_err(|e| CompressionError::Zstd(e.to_string()))
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
    fn zstd_round_trip() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = ZstdCompressor { level: 3 };

        let result = compressor.compress(&data).unwrap();
        assert!(result.block_offsets.is_none());
        assert!(result.data.len() < data.len());

        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn zstd_range_not_supported() {
        let compressor = ZstdCompressor { level: 3 };
        let result = compressor.decompress_range(&[0], &[], 0, 1);
        assert!(matches!(result, Err(CompressionError::RangeNotSupported)));
    }
}

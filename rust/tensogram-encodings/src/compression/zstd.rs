// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use super::{CompressResult, CompressionError, Compressor};

pub struct ZstdCompressor {
    pub level: i32,
    /// Number of worker threads for zstd compression.
    ///
    /// Forwarded to `zstd_safe::CParameter::NbWorkers`.  `0` means
    /// "sequential compression" (no workers spawned) and produces the
    /// pre-0.13.0 byte layout.  Values `>= 1` enable multi-threaded
    /// compression inside libzstd — note that the compressed bytes may
    /// differ from the `nbWorkers=0` output (blocks land in the frame
    /// in worker completion order) while always round-tripping
    /// losslessly.  This field is a no-op on decompress.
    pub nb_workers: u32,
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        // Fast path: no workers requested → use the one-shot helper
        // which is guaranteed byte-identical to previous releases.
        if self.nb_workers == 0 {
            let compressed = zstd::encode_all(data, self.level)
                .map_err(|e| CompressionError::Zstd(e.to_string()))?;
            return Ok(CompressResult {
                data: compressed,
                block_offsets: None,
            });
        }

        // Multi-threaded path: use bulk::Compressor and set NbWorkers.
        let mut cctx = zstd::bulk::Compressor::new(self.level)
            .map_err(|e| CompressionError::Zstd(e.to_string()))?;
        cctx.set_parameter(zstd::zstd_safe::CParameter::NbWorkers(self.nb_workers))
            .map_err(|e| CompressionError::Zstd(e.to_string()))?;
        let compressed = cctx
            .compress(data)
            .map_err(|e| CompressionError::Zstd(e.to_string()))?;
        Ok(CompressResult {
            data: compressed,
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        // Decompression is not affected by NbWorkers — zstd always
        // decodes frames sequentially regardless of how they were
        // encoded.  Use the simple helper.
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
        let compressor = ZstdCompressor {
            level: 3,
            nb_workers: 0,
        };

        let result = compressor.compress(&data).unwrap();
        assert!(result.block_offsets.is_none());
        assert!(result.data.len() < data.len());

        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn zstd_range_not_supported() {
        let compressor = ZstdCompressor {
            level: 3,
            nb_workers: 0,
        };
        let result = compressor.decompress_range(&[0], &[], 0, 1);
        assert!(matches!(result, Err(CompressionError::RangeNotSupported)));
    }

    /// `nb_workers=0` (default) must match the pre-0.13.0 one-shot
    /// output bit-for-bit.  This lets golden files continue to work.
    #[test]
    fn zstd_nb_workers_zero_matches_encode_all() {
        let data: Vec<u8> = (0..32 * 1024).map(|i| ((i * 7) % 256) as u8).collect();
        let via_struct = ZstdCompressor {
            level: 3,
            nb_workers: 0,
        }
        .compress(&data)
        .unwrap()
        .data;
        let via_helper = zstd::encode_all(data.as_slice(), 3).unwrap();
        assert_eq!(via_struct, via_helper);
    }

    /// `nb_workers > 0` must round-trip losslessly.
    /// Compressed bytes may differ; decoded data must not.
    #[test]
    fn zstd_nb_workers_round_trip_lossless() {
        let data: Vec<u8> = (0..256 * 1024).map(|i| ((i * 31) % 256) as u8).collect();
        for n in [1u32, 2, 4, 8] {
            let c = ZstdCompressor {
                level: 3,
                nb_workers: n,
            };
            let out = c.compress(&data).unwrap();
            let rt = c.decompress(&out.data, data.len()).unwrap();
            assert_eq!(rt, data, "zstd nb_workers={n} round-trip failure");
        }
    }
}

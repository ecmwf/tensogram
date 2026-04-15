// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::sync::Once;

use blosc2::chunk::SChunk;
use blosc2::{CParams, CompressAlgo, DParams};

use super::{CompressResult, CompressionError, Compressor};
use crate::pipeline::Blosc2Codec;

/// Ensure the blosc2 C library is initialized.
///
/// Workaround: the `blosc2` crate (v0.2.2) calls `blosc2_init()` inside
/// `SChunk::new()` but not `SChunk::from_buffer()`. Decode-only processes
/// that never compress will hit an uninitialized library and fail.
fn ensure_blosc2_init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: blosc2_init() has no preconditions and is safe to call multiple times.
        unsafe { blosc2_sys::blosc2_init() };
    });
}

fn map_err(e: blosc2::Error) -> CompressionError {
    CompressionError::Blosc2(format!("{e:?}"))
}

fn codec_to_algo(codec: &Blosc2Codec) -> CompressAlgo {
    match codec {
        Blosc2Codec::Blosclz => CompressAlgo::Blosclz,
        Blosc2Codec::Lz4 => CompressAlgo::Lz4,
        Blosc2Codec::Lz4hc => CompressAlgo::Lz4hc,
        Blosc2Codec::Zlib => CompressAlgo::Zlib,
        Blosc2Codec::Zstd => CompressAlgo::Zstd,
    }
}

pub struct Blosc2Compressor {
    pub codec: Blosc2Codec,
    pub clevel: i32,
    pub typesize: usize,
}

impl Compressor for Blosc2Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        ensure_blosc2_init();
        let algo = codec_to_algo(&self.codec);
        let mut cparams = CParams::default();
        cparams
            .compressor(algo)
            .clevel(self.clevel as u32)
            .typesize(self.typesize)
            .map_err(map_err)?;

        let mut schunk = SChunk::new(cparams.clone(), DParams::default()).map_err(map_err)?;
        schunk.append(data).map_err(map_err)?;

        let buf = schunk.to_buffer().map_err(map_err)?;
        Ok(CompressResult {
            data: buf.as_slice().to_vec(),
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        ensure_blosc2_init();
        let schunk = SChunk::from_buffer(data.into()).map_err(map_err)?;
        let num_items = schunk.items_num();
        if num_items == 0 {
            return Ok(Vec::new());
        }
        schunk.items(0..num_items).map_err(map_err)
    }

    fn decompress_range(
        &self,
        data: &[u8],
        _block_offsets: &[u64],
        byte_pos: usize,
        byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        ensure_blosc2_init();
        let schunk = SChunk::from_buffer(data.into()).map_err(map_err)?;
        let ts = schunk.typesize();
        if ts == 0 {
            return Err(CompressionError::Blosc2("typesize is 0".to_string()));
        }

        // Convert byte range to item range
        let item_start = byte_pos / ts;
        let item_end = (byte_pos + byte_size).div_ceil(ts);

        let items = schunk.items(item_start..item_end).map_err(map_err)?;

        // Trim to exact byte range within the item-aligned result
        let offset_in_items = byte_pos % ts;
        let end = offset_in_items
            .checked_add(byte_size)
            .ok_or_else(|| CompressionError::Blosc2("range overflow".to_string()))?;
        if end > items.len() {
            return Err(CompressionError::Blosc2(format!(
                "range exceeds decompressed data: need {end} bytes, got {}",
                items.len()
            )));
        }
        Ok(items[offset_in_items..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blosc2_round_trip() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
        };

        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn blosc2_round_trip_4byte() {
        let data: Vec<u8> = (0..4000).flat_map(|i: u32| i.to_ne_bytes()).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Blosclz,
            clevel: 5,
            typesize: 4,
        };

        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn blosc2_range_decode() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
        };

        let result = compressor.compress(&data).unwrap();
        let partial = compressor
            .decompress_range(&result.data, &[], 200, 500)
            .unwrap();
        assert_eq!(partial.len(), 500);
        assert_eq!(&partial[..], &data[200..700]);
    }

    #[test]
    fn blosc2_round_trip_zstd() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Zstd,
            clevel: 3,
            typesize: 1,
        };
        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn blosc2_round_trip_zlib() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Zlib,
            clevel: 5,
            typesize: 1,
        };
        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Regression guard for the decompress path.
    ///
    /// The real bug (missing blosc2_init in decode-only processes) cannot be
    /// fully reproduced in a unit test because blosc2_init is global state —
    /// once called by compress(), it stays initialized. This test still guards
    /// against regressions in the decompress wiring itself.
    #[test]
    fn blosc2_decompress_path() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressor = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 5,
            typesize: 1,
        };
        let compressed = compressor.compress(&data).unwrap().data;

        let decoder = Blosc2Compressor {
            codec: Blosc2Codec::Lz4,
            clevel: 0,
            typesize: 1,
        };
        let decompressed = decoder.decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}

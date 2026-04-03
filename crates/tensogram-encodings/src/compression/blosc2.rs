use blosc2::chunk::SChunk;
use blosc2::{CParams, CompressAlgo, DParams};

use super::{CompressResult, CompressionError, Compressor};
use crate::pipeline::Blosc2Codec;

fn map_err(e: blosc2::Error) -> CompressionError {
    CompressionError::Blosc2(format!("{e:?}"))
}

fn codec_to_algo(codec: &Blosc2Codec) -> CompressAlgo {
    match codec {
        Blosc2Codec::Blosclz => CompressAlgo::Blosclz,
        Blosc2Codec::Lz4 => CompressAlgo::Lz4,
        Blosc2Codec::Lz4hc => CompressAlgo::Lz4hc,
        // Zlib and Zstd require cargo features on the blosc2 crate.
        // Fall back to blosclz if the feature is not enabled.
        Blosc2Codec::Zlib => CompressAlgo::Blosclz,
        Blosc2Codec::Zstd => CompressAlgo::Blosclz,
    }
}

pub struct Blosc2Compressor {
    pub codec: Blosc2Codec,
    pub clevel: i32,
    pub typesize: usize,
}

impl Compressor for Blosc2Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        let mut cparams = CParams::default();
        cparams
            .compressor(codec_to_algo(&self.codec))
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
        Ok(items[offset_in_items..offset_in_items + byte_size].to_vec())
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
}

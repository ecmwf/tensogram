use crate::compression::{
    CompressResult, CompressionError, Compressor, NoopCompressor, SzipCompressor,
};
use crate::shuffle;
use crate::simple_packing::{self, PackingError, SimplePackingParams};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ByteOrder {
    Big,
    Little,
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("encoding error: {0}")]
    Encoding(#[from] PackingError),
    #[error("compression error: {0}")]
    Compression(#[from] CompressionError),
    #[error("shuffle error: {0}")]
    Shuffle(String),
    #[error("unknown encoding: {0}")]
    UnknownEncoding(String),
    #[error("unknown filter: {0}")]
    UnknownFilter(String),
    #[error("unknown compression: {0}")]
    UnknownCompression(String),
}

#[derive(Debug, Clone)]
pub enum EncodingType {
    None,
    SimplePacking(SimplePackingParams),
}

#[derive(Debug, Clone)]
pub enum FilterType {
    None,
    Shuffle { element_size: usize },
}

#[derive(Debug, Clone)]
pub enum CompressionType {
    None,
    Szip {
        rsi: u32,
        block_size: u32,
        flags: u32,
        bits_per_sample: u32,
    },
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub encoding: EncodingType,
    pub filter: FilterType,
    pub compression: CompressionType,
    pub num_values: usize,
    pub byte_order: ByteOrder,
    pub dtype_byte_width: usize,
}

pub struct PipelineResult {
    pub encoded_bytes: Vec<u8>,
    pub szip_block_offsets: Option<Vec<u64>>,
}

/// Full forward pipeline: encode → filter → compress
pub fn encode_pipeline(
    data: &[u8],
    config: &PipelineConfig,
) -> Result<PipelineResult, PipelineError> {
    // Step 1: Encoding
    let encoded = match &config.encoding {
        EncodingType::None => data.to_vec(),
        EncodingType::SimplePacking(params) => {
            let values = bytes_to_f64(data, config.byte_order);
            simple_packing::encode(&values, params)?
        }
    };

    // Step 2: Filter
    let filtered = match &config.filter {
        FilterType::None => encoded,
        FilterType::Shuffle { element_size } => shuffle::shuffle(&encoded, *element_size)
            .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
    };

    // Step 3: Compression
    let compressor: Box<dyn Compressor> = match &config.compression {
        CompressionType::None => Box::new(NoopCompressor),
        CompressionType::Szip {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        } => Box::new(SzipCompressor {
            rsi: *rsi,
            block_size: *block_size,
            flags: *flags,
            bits_per_sample: *bits_per_sample,
        }),
    };

    let CompressResult {
        data: compressed,
        block_offsets,
    } = compressor.compress(&filtered)?;

    Ok(PipelineResult {
        encoded_bytes: compressed,
        szip_block_offsets: block_offsets,
    })
}

/// Full reverse pipeline: decompress → unshuffle → decode
pub fn decode_pipeline(encoded: &[u8], config: &PipelineConfig) -> Result<Vec<u8>, PipelineError> {
    // Step 1: Decompress
    let decompressor: Box<dyn Compressor> = match &config.compression {
        CompressionType::None => Box::new(NoopCompressor),
        CompressionType::Szip {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        } => Box::new(SzipCompressor {
            rsi: *rsi,
            block_size: *block_size,
            flags: *flags,
            bits_per_sample: *bits_per_sample,
        }),
    };

    // Estimate expected size for decompression
    let expected_size = estimate_decompressed_size(config);
    let decompressed = decompressor.decompress(encoded, expected_size)?;

    // Step 2: Unshuffle
    let unfiltered = match &config.filter {
        FilterType::None => decompressed,
        FilterType::Shuffle { element_size } => shuffle::unshuffle(&decompressed, *element_size)
            .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
    };

    // Step 3: Decode
    let decoded = match &config.encoding {
        EncodingType::None => unfiltered,
        EncodingType::SimplePacking(params) => {
            let values = simple_packing::decode(&unfiltered, config.num_values, params)?;
            f64_to_bytes(&values, config.byte_order)
        }
    };

    Ok(decoded)
}

/// Decode a partial sample range from a compressed+encoded pipeline.
///
/// Only supports `simple_packing+szip` and `none+none` (uncompressed) pipelines.
/// Shuffle filter is not supported with range decode.
///
/// `sample_offset` and `sample_count` are in logical element units.
/// `block_offsets` are bit offsets of RSI block boundaries from encoding.
pub fn decode_range_pipeline(
    encoded: &[u8],
    config: &PipelineConfig,
    block_offsets: &[u64],
    sample_offset: u64,
    sample_count: u64,
) -> Result<Vec<u8>, PipelineError> {
    match &config.filter {
        FilterType::None => {}
        FilterType::Shuffle { .. } => {
            return Err(PipelineError::Shuffle(
                "partial range decode is not supported with shuffle filter".to_string(),
            ));
        }
    }

    match (&config.encoding, &config.compression) {
        // Uncompressed, unencoded: direct byte slice
        (EncodingType::None, CompressionType::None) => {
            let elem_size = config.dtype_byte_width;
            let byte_start = (sample_offset as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| PipelineError::Shuffle("byte offset overflow".to_string()))?;
            let byte_count = (sample_count as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| PipelineError::Shuffle("byte count overflow".to_string()))?;
            let byte_end = byte_start
                .checked_add(byte_count)
                .ok_or_else(|| PipelineError::Shuffle("byte end overflow".to_string()))?;
            if byte_end > encoded.len() {
                return Err(PipelineError::Shuffle(format!(
                    "range ({sample_offset}, {sample_count}) exceeds payload size"
                )));
            }
            Ok(encoded[byte_start..byte_end].to_vec())
        }

        // simple_packing + szip: decompress range, then unpack
        (
            EncodingType::SimplePacking(params),
            CompressionType::Szip {
                rsi,
                block_size,
                flags,
                bits_per_sample,
            },
        ) => {
            let compressor = SzipCompressor {
                rsi: *rsi,
                block_size: *block_size,
                flags: *flags,
                bits_per_sample: *bits_per_sample,
            };

            // Compute byte range in the packed (pre-compression) stream.
            // Each sample is bits_per_value bits, packed MSB-first.
            let bit_start = sample_offset * params.bits_per_value as u64;
            let bit_count = sample_count * params.bits_per_value as u64;
            // We need to decompress at byte-aligned boundaries that cover our bit range
            let byte_start = (bit_start / 8) as usize;
            let byte_end = (bit_start + bit_count).div_ceil(8) as usize;
            let byte_size = byte_end - byte_start;

            let packed_bytes =
                compressor.decompress_range(encoded, block_offsets, byte_start, byte_size)?;

            // Unpack the sample range from the decompressed packed bytes
            // The bit offset within our decompressed chunk
            let bit_offset_in_chunk = (bit_start % 8) as usize;
            let values = simple_packing::decode_range(
                &packed_bytes,
                bit_offset_in_chunk,
                sample_count as usize,
                params,
            )?;

            Ok(f64_to_bytes(&values, config.byte_order))
        }

        // simple_packing without compression: decompress range from packed bits
        (EncodingType::SimplePacking(params), CompressionType::None) => {
            let bit_start = sample_offset * params.bits_per_value as u64;
            let bit_count = sample_count * params.bits_per_value as u64;
            let byte_start = (bit_start / 8) as usize;
            let byte_end = (bit_start + bit_count).div_ceil(8) as usize;
            if byte_end > encoded.len() {
                return Err(PipelineError::Shuffle(format!(
                    "range ({sample_offset}, {sample_count}) exceeds packed data size"
                )));
            }
            let packed_bytes = &encoded[byte_start..byte_end];
            let bit_offset_in_chunk = (bit_start % 8) as usize;
            let values = simple_packing::decode_range(
                packed_bytes,
                bit_offset_in_chunk,
                sample_count as usize,
                params,
            )?;
            Ok(f64_to_bytes(&values, config.byte_order))
        }

        // none encoding + szip: decompress range directly
        (
            EncodingType::None,
            CompressionType::Szip {
                rsi,
                block_size,
                flags,
                bits_per_sample,
            },
        ) => {
            let compressor = SzipCompressor {
                rsi: *rsi,
                block_size: *block_size,
                flags: *flags,
                bits_per_sample: *bits_per_sample,
            };
            let elem_size = config.dtype_byte_width;
            let byte_pos = (sample_offset as usize) * elem_size;
            let byte_size = (sample_count as usize) * elem_size;
            let decompressed =
                compressor.decompress_range(encoded, block_offsets, byte_pos, byte_size)?;
            Ok(decompressed)
        }
    }
}

fn estimate_decompressed_size(config: &PipelineConfig) -> usize {
    match &config.encoding {
        EncodingType::None => config.num_values.saturating_mul(config.dtype_byte_width),
        EncodingType::SimplePacking(params) => {
            let total_bits =
                (config.num_values as u128).saturating_mul(params.bits_per_value as u128);
            total_bits.div_ceil(8).min(usize::MAX as u128) as usize
        }
    }
}

fn bytes_to_f64(data: &[u8], byte_order: ByteOrder) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            match byte_order {
                ByteOrder::Big => f64::from_be_bytes(arr),
                ByteOrder::Little => f64::from_le_bytes(arr),
            }
        })
        .collect()
}

fn f64_to_bytes(values: &[f64], byte_order: ByteOrder) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match byte_order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_pipeline() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 1,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        };
        let result = encode_pipeline(&data, &config).unwrap();
        assert_eq!(result.encoded_bytes, data);
        let decoded = decode_pipeline(&result.encoded_bytes, &config).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_simple_packing_pipeline() {
        let values: Vec<f64> = (0..50).map(|i| 200.0 + i as f64 * 0.1).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params = simple_packing::compute_params(&values, 16, 0).unwrap();

        let config = PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        let decoded = decode_pipeline(&result.encoded_bytes, &config).unwrap();
        let decoded_values = bytes_to_f64(&decoded, ByteOrder::Little);

        for (orig, dec) in values.iter().zip(decoded_values.iter()) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_shuffle_pipeline() {
        let data: Vec<u8> = (0..16).collect();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::Shuffle { element_size: 4 },
            compression: CompressionType::None,
            num_values: 4,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 4,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        assert_ne!(result.encoded_bytes, data); // shuffled should differ
        let decoded = decode_pipeline(&result.encoded_bytes, &config).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_szip_round_trip_pipeline() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: libaec_sys::AEC_DATA_PREPROCESS,
                bits_per_sample: 8,
            },
            num_values: 2048,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 1,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        assert!(result.szip_block_offsets.is_some());

        let decoded = decode_pipeline(&result.encoded_bytes, &config).unwrap();
        assert_eq!(decoded, data);
    }
}

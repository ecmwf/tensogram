use std::borrow::Cow;

#[cfg(feature = "blosc2")]
use crate::compression::Blosc2Compressor;
#[cfg(feature = "lz4")]
use crate::compression::Lz4Compressor;
#[cfg(feature = "sz3")]
use crate::compression::Sz3Compressor;
#[cfg(feature = "szip")]
use crate::compression::SzipCompressor;
#[cfg(feature = "zfp")]
use crate::compression::ZfpCompressor;
#[cfg(feature = "zstd")]
use crate::compression::ZstdCompressor;
use crate::compression::{CompressResult, CompressionError, Compressor};
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
    #[error("range error: {0}")]
    Range(String),
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

impl std::fmt::Display for EncodingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodingType::None => write!(f, "none"),
            EncodingType::SimplePacking(_) => write!(f, "simple_packing"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FilterType {
    None,
    Shuffle { element_size: usize },
}

#[cfg(feature = "blosc2")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Blosc2Codec {
    Blosclz,
    Lz4,
    Lz4hc,
    Zlib,
    Zstd,
}

#[cfg(feature = "zfp")]
#[derive(Debug, Clone)]
pub enum ZfpMode {
    FixedRate { rate: f64 },
    FixedPrecision { precision: u32 },
    FixedAccuracy { tolerance: f64 },
}

#[cfg(feature = "sz3")]
#[derive(Debug, Clone)]
pub enum Sz3ErrorBound {
    Absolute(f64),
    Relative(f64),
    Psnr(f64),
}

#[derive(Debug, Clone)]
pub enum CompressionType {
    None,
    #[cfg(feature = "szip")]
    Szip {
        rsi: u32,
        block_size: u32,
        flags: u32,
        bits_per_sample: u32,
    },
    #[cfg(feature = "zstd")]
    Zstd {
        level: i32,
    },
    #[cfg(feature = "lz4")]
    Lz4,
    #[cfg(feature = "blosc2")]
    Blosc2 {
        codec: Blosc2Codec,
        clevel: i32,
        typesize: usize,
    },
    #[cfg(feature = "zfp")]
    Zfp {
        mode: ZfpMode,
    },
    #[cfg(feature = "sz3")]
    Sz3 {
        error_bound: Sz3ErrorBound,
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
    /// Block offsets produced by compressors that support random access (szip, blosc2).
    pub block_offsets: Option<Vec<u64>>,
}

/// Build a boxed compressor from a CompressionType variant.
fn build_compressor(
    compression: &CompressionType,
    #[allow(unused_variables)] config: &PipelineConfig,
) -> Result<Option<Box<dyn Compressor>>, CompressionError> {
    match compression {
        CompressionType::None => Ok(None),
        #[cfg(feature = "szip")]
        CompressionType::Szip {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        } => Ok(Some(Box::new(SzipCompressor {
            rsi: *rsi,
            block_size: *block_size,
            flags: *flags,
            bits_per_sample: *bits_per_sample,
        }))),
        #[cfg(feature = "zstd")]
        CompressionType::Zstd { level } => Ok(Some(Box::new(ZstdCompressor { level: *level }))),
        #[cfg(feature = "lz4")]
        CompressionType::Lz4 => Ok(Some(Box::new(Lz4Compressor))),
        #[cfg(feature = "blosc2")]
        CompressionType::Blosc2 {
            codec,
            clevel,
            typesize,
        } => Ok(Some(Box::new(Blosc2Compressor {
            codec: *codec,
            clevel: *clevel,
            typesize: *typesize,
        }))),
        #[cfg(feature = "zfp")]
        CompressionType::Zfp { mode } => Ok(Some(Box::new(ZfpCompressor {
            mode: mode.clone(),
            num_values: config.num_values,
        }))),
        #[cfg(feature = "sz3")]
        CompressionType::Sz3 { error_bound } => Ok(Some(Box::new(Sz3Compressor {
            error_bound: error_bound.clone(),
            num_values: config.num_values,
        }))),
    }
}

/// Full forward pipeline: encode → filter → compress
#[tracing::instrument(skip(data, config), fields(data_len = data.len(), encoding = %config.encoding))]
pub fn encode_pipeline(
    data: &[u8],
    config: &PipelineConfig,
) -> Result<PipelineResult, PipelineError> {
    // Step 1: Encoding — Cow avoids cloning when encoding is None
    let encoded: Cow<'_, [u8]> = match &config.encoding {
        EncodingType::None => Cow::Borrowed(data),
        EncodingType::SimplePacking(params) => {
            let values = bytes_to_f64(data, config.byte_order);
            Cow::Owned(simple_packing::encode(&values, params)?)
        }
    };

    // Step 2: Filter
    let filtered: Cow<'_, [u8]> = match &config.filter {
        FilterType::None => encoded,
        FilterType::Shuffle { element_size } => Cow::Owned(
            shuffle::shuffle(&encoded, *element_size)
                .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
        ),
    };

    // Step 3: Compression
    match build_compressor(&config.compression, config)? {
        None => Ok(PipelineResult {
            encoded_bytes: filtered.into_owned(),
            block_offsets: None,
        }),
        Some(compressor) => {
            let CompressResult {
                data: compressed,
                block_offsets,
            } = compressor.compress(&filtered)?;
            Ok(PipelineResult {
                encoded_bytes: compressed,
                block_offsets,
            })
        }
    }
}

/// Full reverse pipeline: decompress → unshuffle → decode
#[tracing::instrument(skip(encoded, config), fields(encoded_len = encoded.len()))]
pub fn decode_pipeline(encoded: &[u8], config: &PipelineConfig) -> Result<Vec<u8>, PipelineError> {
    // Step 1: Decompress — Cow avoids cloning when no compression
    let decompressed: Cow<'_, [u8]> = match build_compressor(&config.compression, config)? {
        None => Cow::Borrowed(encoded),
        Some(compressor) => {
            let expected_size = estimate_decompressed_size(config);
            Cow::Owned(compressor.decompress(encoded, expected_size)?)
        }
    };

    // Step 2: Unshuffle
    let unfiltered: Cow<'_, [u8]> = match &config.filter {
        FilterType::None => decompressed,
        FilterType::Shuffle { element_size } => Cow::Owned(
            shuffle::unshuffle(&decompressed, *element_size)
                .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
        ),
    };

    // Step 3: Decode
    let decoded = match &config.encoding {
        EncodingType::None => unfiltered.into_owned(),
        EncodingType::SimplePacking(params) => {
            let values = simple_packing::decode(&unfiltered, config.num_values, params)?;
            f64_to_bytes(&values, config.byte_order)
        }
    };

    Ok(decoded)
}

/// Decode a partial sample range from a compressed+encoded pipeline.
///
/// Supports compressors with random access (szip, blosc2, zfp fixed-rate).
/// Shuffle filter is not supported with range decode.
///
/// `sample_offset` and `sample_count` are in logical element units.
/// `block_offsets` are block boundary offsets from encoding (compressor-specific).
pub fn decode_range_pipeline(
    encoded: &[u8],
    config: &PipelineConfig,
    block_offsets: &[u64],
    sample_offset: u64,
    sample_count: u64,
) -> Result<Vec<u8>, PipelineError> {
    if matches!(config.filter, FilterType::Shuffle { .. }) {
        return Err(PipelineError::Shuffle(
            "partial range decode is not supported with shuffle filter".to_string(),
        ));
    }

    // Phase 1: Compute byte range needed from the (possibly compressed) stream
    let (byte_start, byte_size, bit_offset_in_chunk) = match &config.encoding {
        EncodingType::SimplePacking(params) => {
            let bit_start = sample_offset * params.bits_per_value as u64;
            let bit_count = sample_count * params.bits_per_value as u64;
            let bs = (bit_start / 8) as usize;
            let be = (bit_start + bit_count).div_ceil(8) as usize;
            (bs, be - bs, Some((bit_start % 8) as usize))
        }
        EncodingType::None => {
            let elem_size = config.dtype_byte_width;
            let bs = (sample_offset as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| PipelineError::Range("byte offset overflow".to_string()))?;
            let sz = (sample_count as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| PipelineError::Range("byte count overflow".to_string()))?;
            (bs, sz, None)
        }
    };

    // Phase 2: Get decompressed bytes for the range
    let decompressed = match build_compressor(&config.compression, config)? {
        None => {
            // No compression: slice directly from encoded buffer
            let byte_end = byte_start
                .checked_add(byte_size)
                .ok_or_else(|| PipelineError::Range("byte end overflow".to_string()))?;
            if byte_end > encoded.len() {
                return Err(PipelineError::Range(format!(
                    "range ({sample_offset}, {sample_count}) exceeds payload size"
                )));
            }
            encoded[byte_start..byte_end].to_vec()
        }
        Some(compressor) => {
            compressor.decompress_range(encoded, block_offsets, byte_start, byte_size)?
        }
    };

    // Phase 3: Decode encoding from decompressed bytes
    match &config.encoding {
        EncodingType::None => Ok(decompressed),
        EncodingType::SimplePacking(params) => {
            let values = simple_packing::decode_range(
                &decompressed,
                bit_offset_in_chunk.unwrap_or(0),
                sample_count as usize,
                params,
            )?;
            Ok(f64_to_bytes(&values, config.byte_order))
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

    #[cfg(feature = "szip")]
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
        assert!(result.block_offsets.is_some());

        let decoded = decode_pipeline(&result.encoded_bytes, &config).unwrap();
        assert_eq!(decoded, data);
    }
}

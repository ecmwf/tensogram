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
        } => Box::new(SzipCompressor {
            rsi: *rsi,
            block_size: *block_size,
            flags: *flags,
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
        } => Box::new(SzipCompressor {
            rsi: *rsi,
            block_size: *block_size,
            flags: *flags,
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
    fn test_szip_returns_error() {
        let data = vec![0u8; 100];
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Szip {
                rsi: 128,
                block_size: 32,
                flags: 0,
            },
            num_values: 100,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 1,
        };
        assert!(encode_pipeline(&data, &config).is_err());
    }
}

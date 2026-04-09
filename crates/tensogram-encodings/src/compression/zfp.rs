use super::{CompressResult, CompressionError, Compressor};
use crate::pipeline::{ByteOrder, ZfpMode};
use crate::zfp_ffi;

/// ZFP compressor for floating-point data.
///
/// ZFP operates directly on typed floating-point arrays, so it replaces
/// both encoding and compression in the pipeline. Use with
/// `encoding: "none"`, `filter: "none"`, `compression: "zfp"`.
///
/// In fixed-rate mode, each block compresses to a fixed number of bits,
/// enabling random access via `decompress_range`.
pub struct ZfpCompressor {
    pub mode: ZfpMode,
    pub num_values: usize,
    /// Byte order to use when serialising decompressed f64 values back to
    /// bytes.  This ensures the decompressed output matches the wire byte
    /// order declared in the descriptor, so the pipeline's uniform
    /// native-endian byteswap step works identically for lossy and
    /// lossless codecs.
    pub byte_order: ByteOrder,
}

impl Compressor for ZfpCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        // Compress side: input bytes are always in the caller's native byte
        // order (per design: "always encode in the endianness of the caller").
        // This intentionally ignores `self.byte_order` — it only governs
        // the decompress output format.  Using the declared byte_order here
        // would produce garbage if the caller provides native-endian data
        // (which is the contract for all encode paths).
        let values = bytes_to_f64_native(data)?;
        let compressed = zfp_ffi::zfp_compress_f64(&values, &self.mode)?;
        Ok(CompressResult {
            data: compressed,
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], _expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        let values = zfp_ffi::zfp_decompress_f64(data, self.num_values, &self.mode)?;
        // Write in the wire byte order so the pipeline's byteswap step can
        // uniformly convert wire → native without special-casing lossy codecs.
        Ok(f64_to_bytes(&values, self.byte_order))
    }

    fn decompress_range(
        &self,
        data: &[u8],
        _block_offsets: &[u64],
        byte_pos: usize,
        byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        // Convert byte range to sample range (f64 = 8 bytes)
        let sample_offset = byte_pos / 8;
        let sample_count = byte_size / 8;

        let values = zfp_ffi::zfp_decompress_range_f64(
            data,
            self.num_values,
            &self.mode,
            sample_offset,
            sample_count,
        )?;

        Ok(f64_to_bytes(&values, self.byte_order))
    }
}

/// Interpret raw bytes as native-endian f64 values (used on the compress side
/// where the caller always provides native-endian data).
fn bytes_to_f64_native(data: &[u8]) -> Result<Vec<f64>, CompressionError> {
    if !data.len().is_multiple_of(8) {
        return Err(CompressionError::Zfp(format!(
            "data length {} is not a multiple of 8",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            f64::from_ne_bytes(arr)
        })
        .collect())
}

/// Serialise f64 values to bytes in the specified byte order.
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

    fn smooth_data(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i as f64 / n as f64 * std::f64::consts::PI).sin())
            .flat_map(|v| v.to_ne_bytes())
            .collect()
    }

    #[test]
    fn zfp_compressor_round_trip() {
        let data = smooth_data(512);
        let compressor = ZfpCompressor {
            mode: ZfpMode::FixedRate { rate: 16.0 },
            num_values: 512,
            byte_order: ByteOrder::native(),
        };

        let result = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&result.data, data.len()).unwrap();
        assert_eq!(decompressed.len(), data.len());
    }

    #[test]
    fn zfp_compressor_range_decode() {
        let data = smooth_data(512);
        let compressor = ZfpCompressor {
            mode: ZfpMode::FixedRate { rate: 16.0 },
            num_values: 512,
            byte_order: ByteOrder::native(),
        };

        let result = compressor.compress(&data).unwrap();
        let full = compressor.decompress(&result.data, data.len()).unwrap();

        // Decode samples 100..200 (byte range 800..1600)
        let partial = compressor
            .decompress_range(&result.data, &[], 800, 800)
            .unwrap();
        assert_eq!(partial.len(), 800);
        assert_eq!(&partial[..], &full[800..1600]);
    }
}

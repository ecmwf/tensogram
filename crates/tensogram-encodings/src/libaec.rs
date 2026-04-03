//! Safe Rust wrapper around libaec (CCSDS 121.0-B-3) compression.
//!
//! Provides compress/decompress with RSI block offset tracking for partial
//! range decode support.

use crate::compression::CompressionError;

/// Parameters for AEC encoding/decoding.
#[derive(Debug, Clone)]
pub struct AecParams {
    pub bits_per_sample: u32,
    pub block_size: u32,
    pub rsi: u32,
    pub flags: u32,
}

/// Compress data using libaec, returning compressed bytes and RSI block bit offsets.
pub fn aec_compress(
    data: &[u8],
    params: &AecParams,
) -> Result<(Vec<u8>, Vec<u64>), CompressionError> {
    use libaec_sys::*;

    if data.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let sample_bytes = sample_byte_width(params.bits_per_sample);
    if !data.len().is_multiple_of(sample_bytes) {
        return Err(CompressionError::Szip(format!(
            "data length {} is not a multiple of sample byte width {}",
            data.len(),
            sample_bytes
        )));
    }

    let num_samples = data.len() / sample_bytes;
    // Worst case: compressed could be slightly larger than input
    let out_capacity = data.len() + data.len() / 4 + 256;
    let mut out = vec![0u8; out_capacity];

    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = out.len();
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = params.flags;

        check_aec(aec_encode_init(&mut strm), "aec_encode_init")?;

        // Enable RSI offset tracking
        check_aec(
            aec_encode_enable_offsets(&mut strm),
            "aec_encode_enable_offsets",
        )?;

        let rc = aec_encode(&mut strm, AEC_FLUSH as _);
        if rc != AEC_OK as _ {
            aec_encode_end(&mut strm);
            return Err(CompressionError::Szip(format!(
                "aec_encode failed with code {rc}"
            )));
        }

        let compressed_len = out.len() - strm.avail_out;

        // Retrieve RSI block offsets (bit offsets)
        let mut offset_count: usize = 0;
        check_aec_cleanup(
            aec_encode_count_offsets(&mut strm, &mut offset_count),
            "aec_encode_count_offsets",
            || {
                aec_encode_end(&mut strm);
            },
        )?;

        let mut offsets = vec![0usize; offset_count];
        if offset_count > 0 {
            check_aec_cleanup(
                aec_encode_get_offsets(&mut strm, offsets.as_mut_ptr(), offset_count),
                "aec_encode_get_offsets",
                || {
                    aec_encode_end(&mut strm);
                },
            )?;
        }

        aec_encode_end(&mut strm);

        out.truncate(compressed_len);
        let bit_offsets: Vec<u64> = offsets.iter().map(|&o| o as u64).collect();
        // Filter out trailing zeros for streams with fewer RSI blocks than reported
        let samples_per_rsi = params.rsi as usize * params.block_size as usize;
        let num_rsi_blocks = num_samples.div_ceil(samples_per_rsi);
        let bit_offsets = if bit_offsets.len() > num_rsi_blocks {
            bit_offsets[..num_rsi_blocks].to_vec()
        } else {
            bit_offsets
        };

        Ok((out, bit_offsets))
    }
}

/// Decompress an entire AEC-compressed stream.
pub fn aec_decompress(
    data: &[u8],
    expected_size: usize,
    params: &AecParams,
) -> Result<Vec<u8>, CompressionError> {
    use libaec_sys::*;

    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = vec![0u8; expected_size];

    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = out.len();
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = params.flags;

        check_aec(aec_decode_init(&mut strm), "aec_decode_init")?;

        let rc = aec_decode(&mut strm, AEC_FLUSH as _);
        if rc != AEC_OK as _ {
            aec_decode_end(&mut strm);
            return Err(CompressionError::Szip(format!(
                "aec_decode failed with code {rc}"
            )));
        }

        let decoded_len = out.len() - strm.avail_out;
        aec_decode_end(&mut strm);

        out.truncate(decoded_len);
        Ok(out)
    }
}

/// Decode a partial range from AEC-compressed data using pre-computed RSI block offsets.
///
/// `block_offsets` are bit offsets of RSI block boundaries in the compressed stream.
/// `byte_pos` and `byte_size` specify the byte range within the decompressed output to extract.
pub fn aec_decompress_range(
    data: &[u8],
    block_offsets: &[u64],
    byte_pos: usize,
    byte_size: usize,
    params: &AecParams,
) -> Result<Vec<u8>, CompressionError> {
    use libaec_sys::*;

    if byte_size == 0 {
        return Ok(Vec::new());
    }
    if data.is_empty() {
        return Err(CompressionError::Szip(
            "cannot decompress range from empty data".to_string(),
        ));
    }

    let mut out = vec![0u8; byte_size];
    let offsets_usize: Vec<usize> = block_offsets.iter().map(|&o| o as usize).collect();

    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = out.len();
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = params.flags;

        check_aec(aec_decode_init(&mut strm), "aec_decode_init")?;

        let rc = aec_decode_range(
            &mut strm,
            offsets_usize.as_ptr(),
            offsets_usize.len(),
            byte_pos,
            byte_size,
        );
        if rc != AEC_OK as _ {
            aec_decode_end(&mut strm);
            return Err(CompressionError::Szip(format!(
                "aec_decode_range failed with code {rc}"
            )));
        }

        let decoded_len = out.len() - strm.avail_out;
        aec_decode_end(&mut strm);

        out.truncate(decoded_len);
        Ok(out)
    }
}

/// Compute the byte width of a sample given bits_per_sample.
/// 3-byte samples are promoted to 4 bytes (libaec convention).
fn sample_byte_width(bits_per_sample: u32) -> usize {
    let nbytes = (bits_per_sample as usize).div_ceil(8);
    if nbytes == 3 {
        4
    } else {
        nbytes
    }
}

fn check_aec(rc: i32, context: &str) -> Result<(), CompressionError> {
    if rc != libaec_sys::AEC_OK as i32 {
        Err(CompressionError::Szip(format!(
            "{context} failed with code {rc}"
        )))
    } else {
        Ok(())
    }
}

fn check_aec_cleanup(
    rc: i32,
    context: &str,
    cleanup: impl FnOnce(),
) -> Result<(), CompressionError> {
    if rc != libaec_sys::AEC_OK as i32 {
        cleanup();
        Err(CompressionError::Szip(format!(
            "{context} failed with code {rc}"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params(bits_per_sample: u32) -> AecParams {
        AecParams {
            bits_per_sample,
            block_size: 16,
            rsi: 128,
            flags: libaec_sys::AEC_DATA_PREPROCESS,
        }
    }

    #[test]
    fn round_trip_u8_data() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_u16_data() {
        let values: Vec<u16> = (0..2048).map(|i| (i * 7 % 65536) as u16).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params = default_params(16);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_u32_data() {
        let values: Vec<u32> = (0..4096).map(|i| i * 13).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params = default_params(32);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn empty_input_returns_empty() {
        let params = default_params(8);
        let (compressed, offsets) = aec_compress(&[], &params).unwrap();
        assert!(compressed.is_empty());
        assert!(offsets.is_empty());

        let decompressed = aec_decompress(&[], 0, &params).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn misaligned_data_returns_error() {
        // 3 bytes is not a multiple of u16 sample width (2 bytes)
        let data = vec![1u8, 2, 3];
        let params = default_params(16);
        assert!(aec_compress(&data, &params).is_err());
    }

    #[test]
    fn offsets_match_rsi_block_count() {
        // 4096 samples / (128 rsi * 16 block_size) = 2 RSI blocks
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);
        let (_, offsets) = aec_compress(&data, &params).unwrap();
        let expected_blocks = 4096_usize.div_ceil(128 * 16);
        assert_eq!(offsets.len(), expected_blocks);
    }

    #[test]
    fn range_decode_matches_full_decode_slice() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();

        // Decode a range from the middle
        let pos = 100;
        let size = 200;
        let partial = aec_decompress_range(&compressed, &offsets, pos, size, &params).unwrap();

        assert_eq!(partial.len(), size);
        assert_eq!(&partial[..], &full[pos..pos + size]);
    }

    #[test]
    fn range_decode_first_block() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();

        // First 50 bytes
        let partial = aec_decompress_range(&compressed, &offsets, 0, 50, &params).unwrap();
        assert_eq!(&partial[..], &full[..50]);
    }

    #[test]
    fn range_decode_last_block() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();

        // Last 100 bytes
        let pos = data.len() - 100;
        let partial = aec_decompress_range(&compressed, &offsets, pos, 100, &params).unwrap();
        assert_eq!(&partial[..], &full[pos..]);
    }

    #[test]
    fn range_decode_entire_stream() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();

        let partial = aec_decompress_range(&compressed, &offsets, 0, data.len(), &params).unwrap();
        assert_eq!(partial, data);
    }

    #[test]
    fn range_decode_zero_size_returns_empty() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();

        let partial = aec_decompress_range(&compressed, &offsets, 0, 0, &params).unwrap();
        assert!(partial.is_empty());
    }

    #[test]
    fn corrupted_data_produces_wrong_output() {
        // libaec may not always return an error for corrupt data, but the
        // decompressed output will differ from the original — this is caught
        // by the hash verification layer in tensogram-core, not the compressor
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);
        let (mut compressed, _) = aec_compress(&data, &params).unwrap();
        // Flip some bytes in the middle of the compressed stream
        for b in compressed[10..20].iter_mut() {
            *b ^= 0xFF;
        }
        // Decompression may succeed but produce wrong data, or may fail
        let result = aec_decompress(&compressed, data.len(), &params);
        match result {
            Err(_) => {} // Error is acceptable
            Ok(decompressed) => assert_ne!(decompressed, data, "corruption should change output"),
        }
    }
}

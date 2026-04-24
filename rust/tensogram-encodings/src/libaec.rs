// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

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
    aec_compress_impl(data, params, true)
}

pub fn aec_compress_no_offsets(
    data: &[u8],
    params: &AecParams,
) -> Result<Vec<u8>, CompressionError> {
    let (bytes, _) = aec_compress_impl(data, params, false)?;
    Ok(bytes)
}

/// Reject AEC parameter combinations that libaec would fail on, or that
/// would drive the decoder into divide-by-zero / infinite-loop territory
/// (`block_size == 0`, `rsi == 0`).  Mirrors the pure-Rust backend's
/// `params::validate` so both szip implementations reject the same
/// malformed input up front, before any FFI boundary is crossed.
fn validate_params(params: &AecParams) -> Result<(), CompressionError> {
    use libaec_sys::*;

    if params.bits_per_sample == 0 || params.bits_per_sample > 32 {
        return Err(CompressionError::Szip(format!(
            "bits_per_sample must be 1..=32, got {}",
            params.bits_per_sample
        )));
    }
    if params.block_size == 0 {
        return Err(CompressionError::Szip(
            "block_size must be non-zero".to_string(),
        ));
    }
    if params.flags & AEC_NOT_ENFORCE != 0 {
        if params.block_size & 1 != 0 {
            return Err(CompressionError::Szip(format!(
                "block_size must be even, got {}",
                params.block_size
            )));
        }
    } else if !matches!(params.block_size, 8 | 16 | 32 | 64) {
        return Err(CompressionError::Szip(format!(
            "block_size must be 8, 16, 32, or 64, got {}",
            params.block_size
        )));
    }
    if params.rsi == 0 || params.rsi > 4096 {
        return Err(CompressionError::Szip(format!(
            "rsi must be 1..=4096, got {}",
            params.rsi
        )));
    }
    if params.flags & AEC_RESTRICTED != 0 && params.bits_per_sample > 4 {
        return Err(CompressionError::Szip(format!(
            "AEC_RESTRICTED requires bits_per_sample <= 4, got {}",
            params.bits_per_sample
        )));
    }
    Ok(())
}

fn aec_compress_impl(
    data: &[u8],
    params: &AecParams,
    track_offsets: bool,
) -> Result<(Vec<u8>, Vec<u64>), CompressionError> {
    use libaec_sys::*;

    validate_params(params)?;

    if data.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let flags = effective_flags(params);
    let sample_bytes = sample_byte_width(params.bits_per_sample, flags);
    if !data.len().is_multiple_of(sample_bytes) {
        return Err(CompressionError::Szip(format!(
            "data length {} is not a multiple of sample byte width {}",
            data.len(),
            sample_bytes
        )));
    }

    let num_samples = data.len() / sample_bytes;
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
        strm.flags = flags;

        check_aec(aec_encode_init(&mut strm), "aec_encode_init")?;

        if track_offsets {
            check_aec(
                aec_encode_enable_offsets(&mut strm),
                "aec_encode_enable_offsets",
            )?;
        }

        let rc = aec_encode(&mut strm, AEC_FLUSH as _);
        if rc != AEC_OK as _ {
            aec_encode_end(&mut strm);
            return Err(CompressionError::Szip(format!(
                "aec_encode failed with code {rc}"
            )));
        }

        let compressed_len = out.len() - strm.avail_out;

        let bit_offsets = if track_offsets {
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

            let bit_offsets: Vec<u64> = offsets.iter().map(|&o| o as u64).collect();
            let samples_per_rsi = params.rsi as usize * params.block_size as usize;
            let num_rsi_blocks = num_samples.div_ceil(samples_per_rsi);
            if bit_offsets.len() > num_rsi_blocks {
                bit_offsets[..num_rsi_blocks].to_vec()
            } else {
                bit_offsets
            }
        } else {
            Vec::new()
        };

        aec_encode_end(&mut strm);

        out.truncate(compressed_len);
        Ok((out, bit_offsets))
    }
}

/// Decompress an entire AEC-compressed stream.
///
/// # Trust model
///
/// `expected_size` is treated as **untrusted**. It originates from the
/// tensor descriptor in the wire format (via
/// `estimate_decompressed_size`) and a malformed `.tgm` file can inflate
/// it to an arbitrary value. The buffer is therefore reserved with
/// [`Vec::try_reserve_exact`]; an oversized request surfaces as
/// [`CompressionError::Szip`] with a `"failed to reserve"` prefix rather
/// than aborting the process.
pub fn aec_decompress(
    data: &[u8],
    expected_size: usize,
    params: &AecParams,
) -> Result<Vec<u8>, CompressionError> {
    use libaec_sys::*;

    validate_params(params)?;

    // Reject the two malformed pairings that would otherwise silently
    // return `Ok(empty)`:
    //   * `expected_size == 0` with non-empty compressed data would
    //     discard whatever the data decodes to.
    //   * `expected_size > 0` with empty compressed data would claim a
    //     successful decode of a truncated/missing payload.
    // The only honest case is both sides empty, which round-trips as
    // an empty Vec.
    match (expected_size, data.is_empty()) {
        (0, true) => return Ok(Vec::new()),
        (0, false) => {
            return Err(CompressionError::Szip(
                "expected_size=0 with non-empty compressed stream (malformed descriptor)"
                    .to_string(),
            ));
        }
        (_, true) => {
            return Err(CompressionError::Szip(format!(
                "expected_size={expected_size} with empty compressed stream (truncated or malformed payload)"
            )));
        }
        _ => {}
    }

    let flags = effective_flags(params);

    // Reserve capacity fallibly — untrusted `expected_size` must not be
    // allowed to abort the process via an infallible allocator panic.
    // We deliberately do NOT zero-initialise the buffer: libaec writes
    // every byte it produces (see SAFETY note below), and the previous
    // `vec![0u8; N]` form forced the kernel to commit every page up
    // front, which was both slower and the mechanism that made hostile
    // sizes lethal under Linux overcommit.
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(expected_size).map_err(|e| {
        CompressionError::Szip(format!(
            "failed to reserve {expected_size} bytes for szip decompression: {e}"
        ))
    })?;

    // SAFETY contract for the FFI call below:
    //   * The `expected_size == 0` early return above means we reach
    //     this point with `expected_size >= 1`, so the preceding
    //     `try_reserve_exact` gave us `capacity() >= 1` — `as_mut_ptr`
    //     is therefore a valid pointer into the backing allocation
    //     rather than the `NonNull::dangling()` that would be returned
    //     for a zero-capacity `Vec`.
    //   * libaec writes forward into `next_out` for at most `avail_out`
    //     bytes and never reads from it.
    //   * `out.len()` remains `0` during the entire FFI call, so no
    //     uninitialised bytes are ever logically "inside" the `Vec`.
    //   * After decode we `set_len(decoded_len)`, where `decoded_len` is
    //     `checked_sub`'d from `expected_size` against the reported
    //     `avail_out`. Every byte in `0..decoded_len` is initialised by
    //     libaec.
    //   * On any error path we return before `set_len`, so the `Vec` is
    //     dropped with `len == 0` and no destructor runs on
    //     uninitialised memory.
    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = expected_size;
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = flags;

        check_aec(aec_decode_init(&mut strm), "aec_decode_init")?;

        let rc = aec_decode(&mut strm, AEC_FLUSH as _);
        if rc != AEC_OK as _ {
            aec_decode_end(&mut strm);
            return Err(CompressionError::Szip(format!(
                "aec_decode failed with code {rc}"
            )));
        }

        // `checked_sub` guards against a misbehaving libaec leaving
        // `avail_out > expected_size` (not legal per the libaec contract,
        // but cheap insurance against the wrapping arithmetic that would
        // otherwise feed `set_len` an absurd length).  Call
        // `aec_decode_end` unconditionally on this error path too so
        // the stream state is never leaked.
        let decoded_len = match expected_size.checked_sub(strm.avail_out) {
            Some(n) => n,
            None => {
                let avail = strm.avail_out;
                aec_decode_end(&mut strm);
                return Err(CompressionError::Szip(format!(
                    "aec_decode reported avail_out={avail} > expected_size={expected_size}"
                )));
            }
        };
        aec_decode_end(&mut strm);

        out.set_len(decoded_len);
        Ok(out)
    }
}

/// Decode a partial range from AEC-compressed data using pre-computed RSI block offsets.
///
/// `block_offsets` are bit offsets of RSI block boundaries in the compressed stream.
/// `byte_pos` and `byte_size` specify the byte range within the decompressed output to extract.
///
/// # Trust model
///
/// `byte_size` is treated as **untrusted**. For honest callers it is the
/// requested range length, but an attacker-supplied descriptor can make
/// upstream range-math produce arbitrary values. The output buffer is
/// reserved with [`Vec::try_reserve_exact`]; an oversized request
/// surfaces as [`CompressionError::Szip`] with a `"failed to reserve"`
/// prefix rather than aborting the process.
pub fn aec_decompress_range(
    data: &[u8],
    block_offsets: &[u64],
    byte_pos: usize,
    byte_size: usize,
    params: &AecParams,
) -> Result<Vec<u8>, CompressionError> {
    use libaec_sys::*;

    validate_params(params)?;

    if byte_size == 0 {
        return Ok(Vec::new());
    }
    if data.is_empty() {
        return Err(CompressionError::Szip(
            "cannot decompress range from empty data".to_string(),
        ));
    }

    let flags = effective_flags(params);

    // Fallible reservation — mirrors `aec_decompress` above; see its
    // trust-model doc comment for the full rationale.
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(byte_size).map_err(|e| {
        CompressionError::Szip(format!(
            "failed to reserve {byte_size} bytes for szip range decode: {e}"
        ))
    })?;
    // Build the usize offsets fallibly: `block_offsets.len()` is
    // small and attacker-bounded by the number of RSI blocks, but
    // `u64 -> usize` can truncate on 32-bit targets, so use
    // `usize::try_from` to surface the overflow as a typed error.
    let mut offsets_usize: Vec<usize> = Vec::new();
    offsets_usize
        .try_reserve_exact(block_offsets.len())
        .map_err(|e| {
            CompressionError::Szip(format!(
                "failed to reserve {} offsets for szip range decode: {e}",
                block_offsets.len(),
            ))
        })?;
    for &o in block_offsets {
        offsets_usize.push(usize::try_from(o).map_err(|_| {
            CompressionError::Szip(format!("RSI block offset {o} exceeds usize on this target"))
        })?);
    }

    // SAFETY: identical reasoning to `aec_decompress`; `out.len() == 0`
    // throughout the FFI call, `capacity >= byte_size`, libaec only
    // writes forward into `next_out`, and `set_len(decoded_len)` runs
    // only on success with `decoded_len = byte_size - strm.avail_out`
    // bytes all initialised by libaec.
    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = byte_size;
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = flags;

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

        // `checked_sub` guards against a misbehaving libaec (see the
        // matching note in `aec_decompress`).  Call `aec_decode_end`
        // unconditionally on this error path so the stream state is
        // never leaked.
        let decoded_len = match byte_size.checked_sub(strm.avail_out) {
            Some(n) => n,
            None => {
                let avail = strm.avail_out;
                aec_decode_end(&mut strm);
                return Err(CompressionError::Szip(format!(
                    "aec_decode_range reported avail_out={avail} > byte_size={byte_size}"
                )));
            }
        };
        aec_decode_end(&mut strm);

        out.set_len(decoded_len);
        Ok(out)
    }
}

/// Ensure AEC_DATA_3BYTE is set for 17-24 bit samples so libaec reads
/// 3-byte containers instead of defaulting to 4-byte.
fn effective_flags(params: &AecParams) -> u32 {
    let mut flags = params.flags;
    if params.bits_per_sample > 16 && params.bits_per_sample <= 24 {
        flags |= libaec_sys::AEC_DATA_3BYTE;
    }
    flags
}

fn sample_byte_width(bits_per_sample: u32, flags: u32) -> usize {
    let nbytes = (bits_per_sample as usize).div_ceil(8);
    if nbytes == 3 && flags & libaec_sys::AEC_DATA_3BYTE == 0 {
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
    fn round_trip_u24_data() {
        // 24-bit samples in 3-byte containers — requires AEC_DATA_3BYTE
        // (automatically set by effective_flags for bits_per_sample 17-24).
        let n = 4096;
        let data: Vec<u8> = (0..n * 3).map(|i| (i % 256) as u8).collect();
        let params = default_params(24);

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
        // by the hash verification layer in tensogram, not the compressor
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

    // ── Preallocation-DoS hardening ──────────────────────────────────────
    //
    // Regression tests for the cross-codec `expected_size` preallocation
    // hardening.  Before the fix, a malicious descriptor whose shape
    // product was close to `usize::MAX` could drive `aec_decompress` (or
    // `aec_decompress_range`) into an infallible `vec![0u8; N]` that
    // aborted the process.  After the fix, the untrusted size is
    // reserved fallibly and rejected cleanly as `CompressionError::Szip`.

    fn small_real_compressed_blob() -> (Vec<u8>, AecParams) {
        let params = default_params(8);
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let (compressed, _) = aec_compress(&data, &params).unwrap();
        (compressed, params)
    }

    #[test]
    fn aec_decompress_rejects_pathological_expected_size() {
        let (compressed, params) = small_real_compressed_blob();

        let err = aec_decompress(&compressed, usize::MAX, &params)
            .expect_err("expected allocation failure, not success nor abort");
        let msg = format!("{err}");
        assert!(
            msg.contains("failed to reserve"),
            "error should report allocation failure, got: {msg}"
        );
    }

    #[test]
    fn aec_decompress_range_rejects_pathological_byte_size() {
        let (compressed, params) = small_real_compressed_blob();
        let offsets: Vec<u64> = Vec::new();

        let err = aec_decompress_range(&compressed, &offsets, 0, usize::MAX, &params)
            .expect_err("expected allocation failure, not success nor abort");
        let msg = format!("{err}");
        assert!(
            msg.contains("failed to reserve"),
            "error should report allocation failure, got: {msg}"
        );
    }

    // ── FFI parameter validator parity ──────────────────────────────────
    //
    // The FFI backend must reject the same invalid `AecParams` as the
    // pure-Rust backend's `params::validate`. These tests assert the
    // parity contract — without them the FFI path could drift from the
    // pure-Rust path unnoticed.

    fn tiny_compressed_blob() -> Vec<u8> {
        let data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let (compressed, _) = aec_compress(&data, &default_params(8)).unwrap();
        compressed
    }

    #[test]
    fn validate_rejects_bits_per_sample_zero() {
        let mut params = default_params(8);
        params.bits_per_sample = 0;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("bits_per_sample=0 must be rejected");
        assert!(format!("{err}").contains("bits_per_sample"));
    }

    #[test]
    fn validate_rejects_bits_per_sample_over_32() {
        let mut params = default_params(8);
        params.bits_per_sample = 33;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("bits_per_sample=33 must be rejected");
        assert!(format!("{err}").contains("bits_per_sample"));
    }

    #[test]
    fn validate_rejects_block_size_zero() {
        let mut params = default_params(8);
        params.block_size = 0;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("block_size=0 must be rejected");
        assert!(format!("{err}").contains("block_size"));
    }

    #[test]
    fn validate_rejects_invalid_block_size_without_not_enforce() {
        let mut params = default_params(8);
        params.block_size = 7;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("block_size=7 must be rejected without AEC_NOT_ENFORCE");
        assert!(format!("{err}").contains("block_size"));
    }

    #[test]
    fn validate_rejects_rsi_zero() {
        let mut params = default_params(8);
        params.rsi = 0;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("rsi=0 must be rejected");
        assert!(format!("{err}").contains("rsi"));
    }

    #[test]
    fn validate_rejects_rsi_over_4096() {
        let mut params = default_params(8);
        params.rsi = 4097;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("rsi=4097 must be rejected");
        assert!(format!("{err}").contains("rsi"));
    }

    #[test]
    fn validate_rejects_restricted_with_bps_over_4() {
        let mut params = default_params(8);
        params.flags |= libaec_sys::AEC_RESTRICTED;
        let err = aec_decompress(&tiny_compressed_blob(), 32, &params)
            .expect_err("AEC_RESTRICTED with bits_per_sample>4 must be rejected");
        assert!(format!("{err}").contains("RESTRICTED"));
    }

    #[test]
    fn validate_accepts_not_enforce_even_block_size() {
        // Even block sizes outside {8,16,32,64} are valid under
        // AEC_NOT_ENFORCE. Round-trip must succeed.
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 6,
            rsi: 64,
            flags: libaec_sys::AEC_DATA_PREPROCESS | libaec_sys::AEC_NOT_ENFORCE,
        };
        let data: Vec<u8> = (0..96).map(|i| i as u8).collect();
        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decoded = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decoded, data);
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PackingError {
    #[error("NaN value encountered at index {0}")]
    NanValue(usize),
    #[error("bits_per_value {0} exceeds maximum of 64")]
    BitsPerValueTooLarge(u32),
    #[error("insufficient data: expected at least {expected} bytes, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
}

#[derive(Debug, Clone)]
pub struct SimplePackingParams {
    pub reference_value: f64,
    pub binary_scale_factor: i32,
    pub decimal_scale_factor: i32,
    pub bits_per_value: u32,
}

pub fn compute_params(
    values: &[f64],
    bits_per_value: u32,
    decimal_scale_factor: i32,
) -> Result<SimplePackingParams, PackingError> {
    if bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(bits_per_value));
    }

    if let Some(i) = values.iter().position(|v| v.is_nan()) {
        return Err(PackingError::NanValue(i));
    }

    if values.is_empty() || bits_per_value == 0 {
        let reference_value = values.first().copied().unwrap_or(0.0);
        return Ok(SimplePackingParams {
            reference_value,
            binary_scale_factor: 0,
            decimal_scale_factor,
            bits_per_value,
        });
    }

    let d_scale = 10f64.powi(decimal_scale_factor);

    // Scale values by decimal factor
    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let reference_value = min_val;
    let range = (max_val - min_val) * d_scale;

    let max_packed = if bits_per_value >= 64 {
        u64::MAX as f64
    } else {
        ((1u64 << bits_per_value) - 1) as f64
    };

    // binary_scale_factor E: range = 2^E * max_packed, so E = log2(range / max_packed)
    let binary_scale_factor = if range == 0.0 || max_packed == 0.0 {
        0
    } else {
        (range / max_packed).log2().ceil() as i32
    };

    Ok(SimplePackingParams {
        reference_value,
        binary_scale_factor,
        decimal_scale_factor,
        bits_per_value,
    })
}

/// Encode f64 values to packed integer bytes (MSB-first bit packing).
/// Rejects NaN inputs.
pub fn encode(values: &[f64], params: &SimplePackingParams) -> Result<Vec<u8>, PackingError> {
    if params.bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(params.bits_per_value));
    }

    if let Some(i) = values.iter().position(|v| v.is_nan()) {
        return Err(PackingError::NanValue(i));
    }

    // bits_per_value == 0: constant field, empty payload
    if params.bits_per_value == 0 {
        return Ok(Vec::new());
    }

    let bpv = params.bits_per_value;
    let d_factor = 10f64.powi(params.decimal_scale_factor);
    let e_factor = 2f64.powi(params.binary_scale_factor);

    // Pack values to integers
    // Forward: packed_int = round((value - reference_value) * 10^D / 2^E)
    let max_packed: u64 = if bpv >= 64 {
        u64::MAX
    } else {
        (1u64 << bpv) - 1
    };

    let total_bits = values.len() as u64 * bpv as u64;
    let total_bytes =
        usize::try_from(total_bits.div_ceil(8)).map_err(|_| PackingError::InsufficientData {
            expected: usize::MAX,
            actual: 0,
        })?;
    let mut output = vec![0u8; total_bytes];

    let mut bit_pos: u64 = 0;
    for &value in values {
        let scaled = (value - params.reference_value) * d_factor / e_factor;
        let packed = scaled.round().max(0.0).min(max_packed as f64) as u64;

        // Write `bpv` bits at bit_pos, MSB-first
        write_bits(&mut output, bit_pos, packed, bpv);
        bit_pos += bpv as u64;
    }

    Ok(output)
}

/// Decode packed bytes back to f64 values.
pub fn decode(
    packed: &[u8],
    num_values: usize,
    params: &SimplePackingParams,
) -> Result<Vec<f64>, PackingError> {
    if params.bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(params.bits_per_value));
    }

    // bits_per_value == 0: all values equal reference_value
    if params.bits_per_value == 0 {
        return Ok(vec![params.reference_value; num_values]);
    }

    let bpv = params.bits_per_value;
    let total_bits = num_values as u64 * bpv as u64;
    let required_bytes = total_bits.div_ceil(8) as usize;
    if packed.len() < required_bytes {
        return Err(PackingError::InsufficientData {
            expected: required_bytes,
            actual: packed.len(),
        });
    }

    let d_factor = 10f64.powi(-params.decimal_scale_factor);
    let e_factor = 2f64.powi(params.binary_scale_factor);

    let mut values = Vec::with_capacity(num_values);
    let mut bit_pos: u64 = 0;

    for _ in 0..num_values {
        let packed_int = read_bits(packed, bit_pos, bpv);
        // Reverse: value = reference_value + 2^E * 10^(-D) * packed_int
        let value = params.reference_value + e_factor * d_factor * packed_int as f64;
        values.push(value);
        bit_pos += bpv as u64;
    }

    Ok(values)
}

/// Decode a range of packed values starting at an arbitrary bit offset.
///
/// `packed` is a byte slice containing the packed bits.
/// `bit_offset` is the starting bit position within `packed`.
/// `num_values` is how many values to unpack.
pub fn decode_range(
    packed: &[u8],
    bit_offset: usize,
    num_values: usize,
    params: &SimplePackingParams,
) -> Result<Vec<f64>, PackingError> {
    if params.bits_per_value > 64 {
        return Err(PackingError::BitsPerValueTooLarge(params.bits_per_value));
    }

    if params.bits_per_value == 0 {
        return Ok(vec![params.reference_value; num_values]);
    }

    let bpv = params.bits_per_value;

    let end_bit = bit_offset as u64 + num_values as u64 * bpv as u64;
    let required_bytes = end_bit.div_ceil(8) as usize;
    if packed.len() < required_bytes {
        return Err(PackingError::InsufficientData {
            expected: required_bytes,
            actual: packed.len(),
        });
    }

    let d_factor = 10f64.powi(-params.decimal_scale_factor);
    let e_factor = 2f64.powi(params.binary_scale_factor);

    let mut values = Vec::with_capacity(num_values);
    let mut bit_pos = bit_offset as u64;

    for _ in 0..num_values {
        let packed_int = read_bits(packed, bit_pos, bpv);
        let value = params.reference_value + e_factor * d_factor * packed_int as f64;
        values.push(value);
        bit_pos += bpv as u64;
    }

    Ok(values)
}

/// Write `nbits` bits of `value` at `bit_offset` in `buf`, MSB-first.
fn write_bits(buf: &mut [u8], bit_offset: u64, value: u64, nbits: u32) {
    if nbits == 0 {
        return;
    }
    let mut remaining = nbits;
    let mut pos = bit_offset as usize;
    let mut val = value;

    // Bits available in the first byte
    let first_avail = 8 - (pos % 8);
    if remaining <= first_avail as u32 {
        // Entire value fits in one byte
        let byte_idx = pos / 8;
        let shift = first_avail as u32 - remaining;
        let mask = ((1u64 << remaining) - 1) as u8;
        buf[byte_idx] |= (val as u8 & mask) << shift;
        return;
    }

    // Write partial first byte
    if !pos.is_multiple_of(8) {
        let bits_in_first = first_avail as u32;
        let byte_idx = pos / 8;
        let top_bits = (val >> (remaining - bits_in_first)) as u8;
        let mask = (1u8 << bits_in_first) - 1;
        buf[byte_idx] |= top_bits & mask;
        remaining -= bits_in_first;
        pos += bits_in_first as usize;
        val &= (1u64 << remaining) - 1;
    }

    // Write full bytes
    while remaining >= 8 {
        remaining -= 8;
        buf[pos / 8] = (val >> remaining) as u8;
        pos += 8;
        if remaining > 0 {
            val &= (1u64 << remaining) - 1;
        }
    }

    // Write partial last byte
    if remaining > 0 {
        let shift = 8 - remaining;
        buf[pos / 8] |= (val as u8) << shift;
    }
}

/// Read `nbits` bits at `bit_offset` from `buf`, MSB-first.
fn read_bits(buf: &[u8], bit_offset: u64, nbits: u32) -> u64 {
    if nbits == 0 {
        return 0;
    }
    let mut remaining = nbits;
    let mut pos = bit_offset as usize;
    let mut value: u64 = 0;

    // Bits available in the first byte
    let first_avail = (8 - (pos % 8)) as u32;
    if remaining <= first_avail {
        // Entire value in one byte
        let byte_idx = pos / 8;
        let shift = first_avail - remaining;
        let mask = (1u64 << remaining) - 1;
        return ((buf[byte_idx] >> shift) as u64) & mask;
    }

    // Read partial first byte
    if !pos.is_multiple_of(8) {
        let bits_in_first = first_avail;
        let byte_idx = pos / 8;
        let mask = (1u8 << bits_in_first) - 1;
        value = (buf[byte_idx] & mask) as u64;
        remaining -= bits_in_first;
        pos += bits_in_first as usize;
    }

    // Read full bytes
    while remaining >= 8 {
        value = (value << 8) | buf[pos / 8] as u64;
        remaining -= 8;
        pos += 8;
    }

    // Read partial last byte
    if remaining > 0 {
        let shift = 8 - remaining;
        value = (value << remaining) | ((buf[pos / 8] >> shift) as u64);
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_field() {
        let values = vec![42.0; 100];
        let params = SimplePackingParams {
            reference_value: 42.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        let encoded = encode(&values, &params).unwrap();
        assert!(encoded.is_empty());
        let decoded = decode(&encoded, 100, &params).unwrap();
        assert_eq!(decoded.len(), 100);
        for v in decoded {
            assert!((v - 42.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nan_rejection() {
        let values = vec![1.0, f64::NAN, 3.0];
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 16,
        };
        assert!(matches!(
            encode(&values, &params),
            Err(PackingError::NanValue(1))
        ));
    }

    #[test]
    fn test_round_trip_16bit() {
        let values: Vec<f64> = (0..100).map(|i| 230.0 + i as f64 * 0.5).collect();
        let params = compute_params(&values, 16, 0).unwrap();
        let encoded = encode(&values, &params).unwrap();
        let decoded = decode(&encoded, values.len(), &params).unwrap();
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_round_trip_12bit() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let params = compute_params(&values, 12, 0).unwrap();
        let encoded = encode(&values, &params).unwrap();
        // 10 values * 12 bits = 120 bits = 15 bytes
        assert_eq!(encoded.len(), 15);
        let decoded = decode(&encoded, values.len(), &params).unwrap();
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_round_trip_1bit() {
        let values = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 1,
        };
        let encoded = encode(&values, &params).unwrap();
        // 9 bits = 2 bytes
        assert_eq!(encoded.len(), 2);
        let decoded = decode(&encoded, values.len(), &params).unwrap();
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bits_per_value_too_large() {
        let values = vec![1.0];
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(encode(&values, &params).is_err());
    }

    // ── Coverage: edge cases ─────────────────────────────────────────

    #[test]
    fn test_compute_params_nan_detection() {
        let values = vec![1.0, f64::NAN, 3.0];
        let result = compute_params(&values, 16, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            PackingError::NanValue(idx) => assert_eq!(idx, 1),
            other => panic!("expected NanValue, got {other:?}"),
        }
    }

    #[test]
    fn test_compute_params_nan_at_start() {
        let values = vec![f64::NAN, 1.0, 2.0];
        match compute_params(&values, 16, 0).unwrap_err() {
            PackingError::NanValue(idx) => assert_eq!(idx, 0),
            other => panic!("expected NanValue(0), got {other:?}"),
        }
    }

    #[test]
    fn test_compute_params_empty_array() {
        let params = compute_params(&[], 8, 0).unwrap();
        assert_eq!(params.reference_value, 0.0);
        assert_eq!(params.binary_scale_factor, 0);
    }

    #[test]
    fn test_compute_params_zero_bits() {
        let params = compute_params(&[1.5, 2.5], 0, 0).unwrap();
        assert_eq!(params.bits_per_value, 0);
        assert_eq!(params.reference_value, 1.5);
    }

    #[test]
    fn test_compute_params_bpv_too_large() {
        match compute_params(&[1.0], 65, 0).unwrap_err() {
            PackingError::BitsPerValueTooLarge(bpv) => assert_eq!(bpv, 65),
            other => panic!("expected BitsPerValueTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn test_compute_params_constant_field() {
        // All values identical → range = 0
        let params = compute_params(&[42.0; 100], 16, 0).unwrap();
        assert_eq!(params.binary_scale_factor, 0);
        assert!((params.reference_value - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_params_64_bit() {
        // bits_per_value = 64 → uses u64::MAX branch
        let params = compute_params(&[0.0, 1e18], 64, 0).unwrap();
        assert_eq!(params.bits_per_value, 64);
    }

    #[test]
    fn test_encode_zero_bits() {
        let params = SimplePackingParams {
            reference_value: 5.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        let encoded = encode(&[5.0, 5.0, 5.0], &params).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_decode_zero_bits() {
        let params = SimplePackingParams {
            reference_value: 7.5,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        let decoded = decode(&[], 5, &params).unwrap();
        assert_eq!(decoded.len(), 5);
        for v in &decoded {
            assert!((v - 7.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decode_insufficient_data() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 8,
        };
        // Need 100 bytes for 100 values at 8 bpv, provide only 10
        match decode(&[0u8; 10], 100, &params).unwrap_err() {
            PackingError::InsufficientData { expected, actual } => {
                assert_eq!(expected, 100);
                assert_eq!(actual, 10);
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_range_zero_bits() {
        let params = SimplePackingParams {
            reference_value: 3.125,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 0,
        };
        // decode_range(packed, bit_offset, num_values, params)
        let decoded = decode_range(&[], 0, 3, &params).unwrap();
        assert_eq!(decoded.len(), 3);
        for v in &decoded {
            assert!((v - 3.125).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decode_range_insufficient_data() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 16,
        };
        // decode_range(packed, bit_offset, num_values, params)
        // bit_offset=80, num_values=3, bpv=16 → need (80+48)/8 = 16 bytes, provide 8
        match decode_range(&[0u8; 8], 80, 3, &params).unwrap_err() {
            PackingError::InsufficientData { .. } => {}
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn test_encode_bpv_too_large() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(encode(&[1.0], &params).is_err());
    }

    #[test]
    fn test_decode_bpv_too_large() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(decode(&[0u8; 16], 1, &params).is_err());
    }

    #[test]
    fn test_decode_range_bpv_too_large() {
        let params = SimplePackingParams {
            reference_value: 0.0,
            binary_scale_factor: 0,
            decimal_scale_factor: 0,
            bits_per_value: 65,
        };
        assert!(decode_range(&[0u8; 16], 0, 1, &params).is_err());
    }

    #[test]
    fn test_write_read_bits_zero() {
        // write_bits and read_bits with 0 bits should be no-ops
        let mut buf = vec![0u8; 4];
        write_bits(&mut buf, 0, 0, 0);
        assert_eq!(read_bits(&buf, 0, 0), 0);
    }
}

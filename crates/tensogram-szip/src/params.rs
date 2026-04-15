// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! AEC parameters and flag constants matching the libaec C API.
//!
//! These constants are byte-compatible with `libaec_sys` so that
//! `tensogram-encodings` can swap implementations at build time.

/// Parameters for AEC encoding/decoding.
#[derive(Debug, Clone)]
pub struct AecParams {
    /// Bits per input sample (1–32).
    pub bits_per_sample: u32,
    /// Samples per coding block (8, 16, 32, or 64).
    pub block_size: u32,
    /// Number of blocks per Reference Sample Interval.
    pub rsi: u32,
    /// Combination of `AEC_DATA_*` flags.
    pub flags: u32,
}

// ── Flag constants (matching libaec.h values exactly) ────────────────────────

/// Input samples are signed (two's complement).
pub const AEC_DATA_SIGNED: u32 = 1;
/// Samples are stored in 3-byte containers (for 17–24 bit).
pub const AEC_DATA_3BYTE: u32 = 2;
/// Samples are stored MSB-first (big-endian).
pub const AEC_DATA_MSB: u32 = 4;
/// Enable preprocessor (unit-delay delta coding).
pub const AEC_DATA_PREPROCESS: u32 = 8;
/// Allow k=13 extension (reserved, unused in this implementation).
pub const AEC_ALLOW_K13: u32 = 16;
/// Pad each RSI to byte boundary.
pub const AEC_PAD_RSI: u32 = 32;
/// Allow non-standard block sizes (any even number).
pub const AEC_NOT_ENFORCE: u32 = 64;
/// Use restricted set of code options (for ≤4 bit samples).
pub const AEC_RESTRICTED: u32 = 128;

// ── Derived helpers ──────────────────────────────────────────────────────────

/// Compute the byte width of each sample container in the input/output buffer.
pub(crate) fn sample_byte_width(bits_per_sample: u32, flags: u32) -> usize {
    let nbytes = (bits_per_sample as usize).div_ceil(8);
    if nbytes == 3 && flags & AEC_DATA_3BYTE == 0 {
        4 // 17–24 bit in 4-byte containers unless 3BYTE flag is set
    } else {
        nbytes
    }
}

/// Apply automatic flag adjustments (mirrors libaec `effective_flags`).
///
/// - Sets `AEC_DATA_3BYTE` for 17–24 bit samples so the codec reads
///   3-byte containers instead of defaulting to 4-byte.
pub(crate) fn effective_flags(params: &AecParams) -> u32 {
    let mut flags = params.flags;
    if params.bits_per_sample > 16 && params.bits_per_sample <= 24 {
        flags |= AEC_DATA_3BYTE;
    }
    flags
}

/// Compute the ID length (number of bits for the coding option identifier).
///
/// Matches libaec exactly:
/// - bits_per_sample > 16 → 5
/// - bits_per_sample > 8  → 4
/// - bits_per_sample ≤ 8 + RESTRICTED:
///     - ≤ 2 → 1
///     - ≤ 4 → 2
///     - > 4 → error (should be caught during validation)
/// - bits_per_sample ≤ 8 (normal) → 3
pub(crate) fn id_len(bits_per_sample: u32, flags: u32) -> u32 {
    if bits_per_sample > 16 {
        5
    } else if bits_per_sample > 8 {
        4
    } else if flags & AEC_RESTRICTED != 0 {
        if bits_per_sample <= 2 {
            1
        } else {
            2 // bits_per_sample 3 or 4
        }
    } else {
        3
    }
}

/// Maximum split parameter k = (1 << id_len) - 3.
pub(crate) fn kmax(id_len: u32) -> u32 {
    (1u32 << id_len).saturating_sub(3)
}

/// Validate AEC parameters and return an error if invalid.
pub(crate) fn validate(params: &AecParams) -> Result<(), crate::AecError> {
    use crate::AecError;

    if params.bits_per_sample == 0 || params.bits_per_sample > 32 {
        return Err(AecError::Config(format!(
            "bits_per_sample must be 1–32, got {}",
            params.bits_per_sample
        )));
    }

    if params.flags & AEC_NOT_ENFORCE != 0 {
        if params.block_size & 1 != 0 {
            return Err(AecError::Config(format!(
                "block_size must be even, got {}",
                params.block_size
            )));
        }
    } else if !matches!(params.block_size, 8 | 16 | 32 | 64) {
        return Err(AecError::Config(format!(
            "block_size must be 8, 16, 32, or 64, got {}",
            params.block_size
        )));
    }

    if params.rsi == 0 || params.rsi > 4096 {
        return Err(AecError::Config(format!(
            "rsi must be 1–4096, got {}",
            params.rsi
        )));
    }

    if params.flags & AEC_RESTRICTED != 0 && params.bits_per_sample > 4 {
        return Err(AecError::Config(format!(
            "AEC_RESTRICTED requires bits_per_sample ≤ 4, got {}",
            params.bits_per_sample
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_len_values() {
        assert_eq!(id_len(32, 0), 5);
        assert_eq!(id_len(24, 0), 5);
        assert_eq!(id_len(17, 0), 5);
        assert_eq!(id_len(16, 0), 4);
        assert_eq!(id_len(9, 0), 4);
        assert_eq!(id_len(8, 0), 3);
        assert_eq!(id_len(1, 0), 3);
        assert_eq!(id_len(2, AEC_RESTRICTED), 1);
        assert_eq!(id_len(4, AEC_RESTRICTED), 2);
    }

    #[test]
    fn test_kmax_values() {
        assert_eq!(kmax(5), 29);
        assert_eq!(kmax(4), 13);
        assert_eq!(kmax(3), 5);
        assert_eq!(kmax(2), 1);
        assert_eq!(kmax(1), 0); // restricted, ≤2 bit: no split options, only uncomp/zero/se
    }

    #[test]
    fn test_sample_byte_width() {
        assert_eq!(sample_byte_width(8, 0), 1);
        assert_eq!(sample_byte_width(16, 0), 2);
        assert_eq!(sample_byte_width(24, 0), 4); // no 3BYTE flag → 4-byte container
        assert_eq!(sample_byte_width(24, AEC_DATA_3BYTE), 3);
        assert_eq!(sample_byte_width(32, 0), 4);
    }

    #[test]
    fn test_effective_flags_auto_3byte() {
        let p = AecParams {
            bits_per_sample: 24,
            block_size: 16,
            rsi: 128,
            flags: AEC_DATA_PREPROCESS,
        };
        let f = effective_flags(&p);
        assert!(f & AEC_DATA_3BYTE != 0);
        assert!(f & AEC_DATA_PREPROCESS != 0);
    }

    #[test]
    fn test_validate_ok() {
        let p = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 128,
            flags: AEC_DATA_PREPROCESS,
        };
        assert!(validate(&p).is_ok());
    }

    #[test]
    fn test_validate_bad_bps() {
        let p = AecParams {
            bits_per_sample: 0,
            block_size: 16,
            rsi: 128,
            flags: 0,
        };
        assert!(validate(&p).is_err());

        let p2 = AecParams {
            bits_per_sample: 33,
            block_size: 16,
            rsi: 128,
            flags: 0,
        };
        assert!(validate(&p2).is_err());
    }

    #[test]
    fn test_validate_bad_block_size() {
        let p = AecParams {
            bits_per_sample: 8,
            block_size: 12,
            rsi: 128,
            flags: 0,
        };
        assert!(validate(&p).is_err());

        // With NOT_ENFORCE, even block sizes are allowed
        let p2 = AecParams {
            bits_per_sample: 8,
            block_size: 12,
            rsi: 128,
            flags: AEC_NOT_ENFORCE,
        };
        assert!(validate(&p2).is_ok());
    }
}

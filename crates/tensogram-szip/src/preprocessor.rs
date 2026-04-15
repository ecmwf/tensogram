// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Unit-delay preprocessor and post-processor for the AEC codec.
//!
//! The preprocessor converts sample values into mapped (non-negative)
//! differences that are more compressible. The post-processor inverts
//! this transformation during decoding.
//!
//! The mapping follows CCSDS 121.0-B-3 Section 4.2.1 exactly.

/// Preprocess an RSI of unsigned samples in-place.
///
/// Returns `(reference_sample, mapped_values)` where `mapped_values`
/// is a Vec of non-negative mapped differences for the samples after
/// the reference. The reference sample is the first sample of the RSI
/// and is stored raw.
pub(crate) fn preprocess_unsigned(samples: &[u32], xmax: u32) -> (u32, Vec<u32>) {
    if samples.is_empty() {
        return (0, Vec::new());
    }

    let reference = samples[0];
    let mut mapped = Vec::with_capacity(samples.len() - 1);

    for i in 0..samples.len() - 1 {
        let x_prev = samples[i];
        let x_curr = samples[i + 1];

        let d = if x_curr >= x_prev {
            let delta = x_curr - x_prev;
            if delta <= x_prev {
                2 * delta
            } else {
                x_curr
            }
        } else {
            let delta = x_prev - x_curr;
            if delta <= xmax - x_prev {
                2 * delta - 1
            } else {
                xmax - x_curr
            }
        };
        mapped.push(d);
    }

    (reference, mapped)
}

/// Preprocess an RSI of signed samples in-place.
///
/// Returns `(reference_sample, mapped_values)`. The reference sample
/// is stored as its raw unsigned representation (before sign extension).
pub(crate) fn preprocess_signed(
    samples: &[u32],
    bits_per_sample: u32,
    xmax: u32,
) -> (u32, Vec<u32>) {
    if samples.is_empty() {
        return (0, Vec::new());
    }

    let m = 1u32 << (bits_per_sample - 1);
    let reference = samples[0]; // raw unsigned representation

    let mut mapped = Vec::with_capacity(samples.len() - 1);

    // Follow libaec's exact signed preprocessing logic (preprocess_signed in encode.c)
    for i in 0..samples.len() - 1 {
        // Sign extend both
        let x_prev = ((samples[i] ^ m).wrapping_sub(m)) as i32;
        let x_curr = ((samples[i + 1] ^ m).wrapping_sub(m)) as i32;

        // libaec formulas (encode.c preprocess_signed):
        //   xmin = -(xmax+1)
        //   For x_curr < x_prev:  delta = x_prev - x_curr
        //     if delta <= xmax - x_prev → mapped = 2*delta - 1
        //     else → mapped = xmax - x_curr  (= -(x_curr + xmin) - 1)
        //   For x_curr >= x_prev: delta = x_curr - x_prev
        //     if delta <= x_prev - xmin → mapped = 2*delta
        //     else → mapped = x_curr - xmin  (= x_curr + xmax + 1)
        // Use i64 throughout to avoid any intermediate overflow or sign-cast
        // ambiguity.  All intermediate values fit comfortably in i64 since
        // x_prev, x_curr ∈ [-(xmax+1), xmax] and xmax ≤ 2^31-1.
        let xmax_i = xmax as i64;
        let xp = x_prev as i64;
        let xc = x_curr as i64;

        let d = if xc < xp {
            let delta = (xp - xc) as u64;
            // Threshold: xmax - x_prev (always ≥ 0 since x_prev ≤ xmax)
            if delta <= (xmax_i - xp) as u64 {
                (2 * delta - 1) as u32
            } else {
                // xmax - x_curr (= -(x_curr + xmin) - 1)
                (xmax_i - xc) as u32
            }
        } else {
            let delta = (xc - xp) as u64;
            // Threshold: x_prev - xmin = x_prev + xmax + 1
            if delta <= (xp + xmax_i + 1) as u64 {
                (2 * delta) as u32
            } else {
                // x_curr - xmin = x_curr + xmax + 1
                (xc + xmax_i + 1) as u32
            }
        };
        mapped.push(d);
    }

    (reference, mapped)
}

/// Post-process (undo preprocessing) for unsigned samples.
///
/// Given a reference sample and mapped values, reconstruct the
/// original sample stream.
pub(crate) fn postprocess_unsigned(reference: u32, mapped: &[u32], xmax: u32) -> Vec<u32> {
    let mut output = Vec::with_capacity(mapped.len() + 1);
    output.push(reference);

    let med = xmax / 2 + 1;
    let mut last = reference;

    for &d in mapped {
        let half_d = (d >> 1) + (d & 1);

        // Determine mask based on whether last is in upper half
        let mask = if last >= med { xmax } else { 0 };

        let next = if half_d <= (mask ^ last) {
            // Apply signed delta: d>>1 if d is even (positive), -(d>>1+1) if d is odd (negative)
            if d & 1 == 0 {
                last.wrapping_add(d >> 1)
            } else {
                last.wrapping_sub((d >> 1) + 1)
            }
        } else {
            mask ^ d
        };

        output.push(next);
        last = next;
    }

    output
}

/// Post-process (undo preprocessing) for signed samples.
///
/// Inverts [`preprocess_signed`]: given the raw unsigned reference and the
/// mapped delta values, reconstructs the original signed sample stream.
/// The reference is sign-extended from its raw unsigned representation
/// before the inverse mapping is applied.
///
/// Returns samples as raw unsigned representations (same bit pattern as
/// the original input to `preprocess_signed`).
pub(crate) fn postprocess_signed(
    reference_raw: u32,
    mapped: &[u32],
    bits_per_sample: u32,
    xmax: u32,
) -> Vec<u32> {
    let m = 1u32 << (bits_per_sample - 1);
    let mut output = Vec::with_capacity(mapped.len() + 1);

    // Sign extend the reference sample
    let ref_signed = ((reference_raw ^ m).wrapping_sub(m)) as i32;
    output.push(reference_raw);

    let mut last = ref_signed;
    let xmax_s = xmax as i32;

    for &d in mapped {
        let half_d = ((d >> 1) + (d & 1)) as i32;

        let next = if last < 0 {
            if half_d <= xmax_s + last + 1 {
                // Apply signed delta
                if d & 1 == 0 {
                    last + (d >> 1) as i32
                } else {
                    last - (d >> 1) as i32 - 1
                }
            } else {
                d as i32 - xmax_s - 1
            }
        } else {
            if half_d <= xmax_s - last {
                if d & 1 == 0 {
                    last + (d >> 1) as i32
                } else {
                    last - (d >> 1) as i32 - 1
                }
            } else {
                xmax_s - d as i32
            }
        };

        // Store as unsigned representation (masking to bits_per_sample).
        // For bps=32 the mask is u32::MAX; for bps<32 it is (1<<bps)-1.
        let mask = if bits_per_sample >= 32 {
            u32::MAX
        } else {
            (1u32 << bits_per_sample) - 1
        };
        let raw = (next as u32) & mask;
        output.push(raw);
        last = next;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_postprocess_unsigned_round_trip() {
        let samples = vec![100u32, 102, 101, 105, 103, 100, 110, 108];
        let xmax = 255u32;

        let (reference, mapped) = preprocess_unsigned(&samples, xmax);
        let reconstructed = postprocess_unsigned(reference, &mapped, xmax);

        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_postprocess_unsigned_monotonic() {
        let samples: Vec<u32> = (0..64).collect();
        let xmax = 255u32;

        let (reference, mapped) = preprocess_unsigned(&samples, xmax);
        let reconstructed = postprocess_unsigned(reference, &mapped, xmax);

        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_postprocess_unsigned_constant() {
        let samples = vec![42u32; 32];
        let xmax = 255u32;

        let (reference, mapped) = preprocess_unsigned(&samples, xmax);
        // All deltas should be zero
        assert!(mapped.iter().all(|&d| d == 0));
        let reconstructed = postprocess_unsigned(reference, &mapped, xmax);
        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_postprocess_unsigned_edge_values() {
        let samples = vec![0u32, 255, 0, 255, 128, 0, 255];
        let xmax = 255u32;

        let (reference, mapped) = preprocess_unsigned(&samples, xmax);
        let reconstructed = postprocess_unsigned(reference, &mapped, xmax);

        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_empty() {
        let (reference, mapped) = preprocess_unsigned(&[], 255);
        assert_eq!(reference, 0);
        assert!(mapped.is_empty());
    }

    #[test]
    fn preprocess_single() {
        let (reference, mapped) = preprocess_unsigned(&[42], 255);
        assert_eq!(reference, 42);
        assert!(mapped.is_empty());
    }

    #[test]
    fn preprocess_postprocess_unsigned_16bit() {
        let samples: Vec<u32> = (0..128).map(|i| (i * 511) % 65536).collect();
        let xmax = 65535u32;

        let (reference, mapped) = preprocess_unsigned(&samples, xmax);
        let reconstructed = postprocess_unsigned(reference, &mapped, xmax);

        assert_eq!(reconstructed, samples);
    }

    // ── Signed preprocessing tests ──────────────────────────────────────

    #[test]
    fn preprocess_postprocess_signed_round_trip() {
        // 8-bit signed: values in [-128, 127]
        let samples: Vec<u32> = vec![128, 130, 126, 135, 120, 128]; // as unsigned repr
        let bps = 8;
        let xmax = 127u32; // (1 << (bps-1)) - 1

        let (reference, mapped) = preprocess_signed(&samples, bps, xmax);
        let reconstructed = postprocess_signed(reference, &mapped, bps, xmax);

        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_postprocess_signed_16bit() {
        // 16-bit signed: values as unsigned u16 repr
        let bps = 16;
        let xmax = 32767u32; // (1 << 15) - 1
        let samples: Vec<u32> = (0..64).map(|i| (i * 1031) % 65536).collect();

        let (reference, mapped) = preprocess_signed(&samples, bps, xmax);
        let reconstructed = postprocess_signed(reference, &mapped, bps, xmax);

        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_postprocess_signed_constant() {
        let bps = 8;
        let xmax = 127u32;
        let samples = vec![0u32; 32]; // constant zero (signed)

        let (reference, mapped) = preprocess_signed(&samples, bps, xmax);
        assert!(
            mapped.iter().all(|&d| d == 0),
            "constant should map to all-zero deltas"
        );
        let reconstructed = postprocess_signed(reference, &mapped, bps, xmax);
        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn preprocess_postprocess_signed_single() {
        let bps = 8;
        let xmax = 127u32;
        let (reference, mapped) = preprocess_signed(&[42], bps, xmax);
        assert_eq!(reference, 42);
        assert!(mapped.is_empty());
    }

    #[test]
    fn preprocess_postprocess_signed_empty() {
        let bps = 8;
        let xmax = 127u32;
        let (reference, mapped) = preprocess_signed(&[], bps, xmax);
        assert_eq!(reference, 0);
        assert!(mapped.is_empty());
    }
}

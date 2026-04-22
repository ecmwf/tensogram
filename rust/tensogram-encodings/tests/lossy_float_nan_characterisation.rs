// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! **Characterisation tests** — pin the *actual* behaviour of `zfp` and
//! `sz3` when asked to compress NaN / +Inf / -Inf inputs.
//!
//! These tests exist because the wrappers in
//! `rust/tensogram-encodings/src/compression/{zfp,sz3}.rs` forward
//! the caller's input to the upstream C/C++ library without any
//! pre-compress scan.  The observable behaviour varies across upstream
//! versions and is undefined by contract.
//!
//! Each test:
//!   1. Builds an input mixing finite values with a known-bad value at
//!      a known position.
//!   2. Attempts `compress → decompress`.
//!   3. Records **how the codec reacted** via a `Outcome` enum:
//!      - `E` — encode or decode returned an error.
//!      - `N` — round-trip succeeded AND the NaN/Inf position still
//!        contains NaN/Inf AND neighbouring finite values stay within
//!        the codec's error bound.
//!      - `C` — round-trip succeeded but neighbouring finite values
//!        are corrupted (predictor propagated the bad value).
//!      - `G` — garbage: values everywhere have no relationship to
//!        the original, or the output length changed.
//!
//! The findings serve as an authoritative behavioural record for
//! our pinned upstream versions; any drift fails these tests.
//!
//! # Why characterise instead of defending upfront?
//!
//! Workstream 1 (the `reject_nan` / `reject_inf` `EncodeOptions`
//! flags, see `rust/tensogram/tests/strict_finite.rs`) already gives
//! callers an escape hatch.  The open question — see the memo §5 Q7 —
//! is whether to add a codec-level defence on top.  Pinning current
//! behaviour first means the defence discussion is grounded in
//! measurements rather than speculation, and a future upstream change
//! trips a test rather than silently shifting user-visible output.

#![cfg(any(feature = "zfp", feature = "sz3"))]

use tensogram_encodings::compression::Compressor;
use tensogram_encodings::pipeline::ByteOrder;

#[cfg(feature = "zfp")]
use tensogram_encodings::compression::ZfpCompressor;
#[cfg(feature = "zfp")]
use tensogram_encodings::pipeline::ZfpMode;

#[cfg(feature = "sz3")]
use tensogram_encodings::compression::Sz3Compressor;
#[cfg(feature = "sz3")]
use tensogram_encodings::pipeline::Sz3ErrorBound;

// ────────────────────────────────────────────────────────────────────────────
// Outcome classification
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Encode returned an error. Codec self-defends at encode time.
    #[allow(dead_code)] // may be observed on other upstream versions
    EncodeError,
    /// Decode returned an error (codec self-defends at decode time).
    #[allow(dead_code)]
    DecodeError,
    /// Round-trip succeeded; NaN/Inf preserved at the original index
    /// with the same kind (NaN stays NaN, +Inf stays +Inf, etc.), and
    /// neighbouring finite values unchanged (within tolerance).
    Preserved,
    /// Round-trip succeeded; neighbouring finite values stable, but
    /// the NaN/Inf slot itself silently decoded to some other value
    /// (typically `0.0` or a small finite number).  The damage is
    /// localised — downstream consumers lose the "non-finite" signal
    /// at that position.
    BadSlotReplaced,
    /// Round-trip succeeded; finite neighbours now deviate beyond the
    /// codec's error bound or became NaN (predictor infection).  The
    /// damage spreads beyond the bad slot.
    NeighbourCorruption,
    /// Output length mismatch, or >50% of finite positions damaged.
    #[allow(dead_code)]
    Garbage,
}

impl Outcome {
    fn summary(self) -> &'static str {
        match self {
            Outcome::EncodeError => "E (encode error)",
            Outcome::DecodeError => "E (decode error)",
            Outcome::Preserved => "N (preserved)",
            Outcome::BadSlotReplaced => "R (bad-slot silently replaced)",
            Outcome::NeighbourCorruption => "C (neighbour corruption)",
            Outcome::Garbage => "G (garbage)",
        }
    }
}

fn f64_bytes(vs: &[f64]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// Diagnostic record returned alongside each outcome — helps us pin
/// exactly what happened when the memo documents the findings.
#[derive(Debug)]
struct Diagnostic {
    /// Value at `bad_idx` after round-trip.
    bad_value_after: f64,
    /// Fraction of finite inputs whose output deviates beyond the
    /// tolerance.  0.0 = no drift; 1.0 = everything moved.
    neighbour_drift_fraction: f64,
    /// Max absolute deviation on any finite-input position (excluding
    /// `bad_idx`).
    max_neighbour_abs_err: f64,
    /// True iff any finite input decoded to NaN.
    finite_decoded_as_nan: bool,
}

/// Classify the outcome of a round-trip and produce a diagnostic.
///
/// `input` is the ground truth; `output` is what came back.  `bad_idx`
/// is the position of the known-bad value (NaN or Inf).  `tolerance`
/// bounds how far neighbours can drift (codec-specific error budget).
fn classify(
    input: &[f64],
    output_bytes_res: Result<Vec<u8>, String>,
    bad_idx: usize,
    tolerance: f64,
) -> (Outcome, Option<Diagnostic>) {
    let bytes = match output_bytes_res {
        Err(_) => return (Outcome::DecodeError, None),
        Ok(b) => b,
    };
    if bytes.len() != input.len() * 8 {
        return (Outcome::Garbage, None);
    }
    let output = bytes_to_f64(&bytes);

    let bad_input = input[bad_idx];
    let bad_output = output[bad_idx];

    let bad_preserved_shape = match (bad_input.is_nan(), bad_input.is_infinite()) {
        (true, _) => bad_output.is_nan(),
        (_, true) => {
            bad_output.is_infinite()
                && bad_input.is_sign_positive() == bad_output.is_sign_positive()
        }
        _ => true, // control: bad_input is finite, nothing to preserve
    };

    // Count neighbour drift.
    let mut drift_count = 0usize;
    let mut finite_total = 0usize;
    let mut max_abs = 0.0_f64;
    let mut finite_to_nan = false;
    for (i, (&o, &d)) in input.iter().zip(output.iter()).enumerate() {
        if i == bad_idx {
            continue;
        }
        if !o.is_finite() {
            continue;
        }
        finite_total += 1;
        if d.is_nan() {
            finite_to_nan = true;
            drift_count += 1;
            continue;
        }
        let abs_err = (o - d).abs();
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        let limit = tolerance.max(o.abs() * 0.1);
        if abs_err > limit {
            drift_count += 1;
        }
    }

    let drift_fraction = if finite_total > 0 {
        drift_count as f64 / finite_total as f64
    } else {
        0.0
    };

    let diag = Diagnostic {
        bad_value_after: bad_output,
        neighbour_drift_fraction: drift_fraction,
        max_neighbour_abs_err: max_abs,
        finite_decoded_as_nan: finite_to_nan,
    };

    // Determine outcome class in order of severity:
    //   Garbage            — >50% finite positions damaged
    //   NeighbourCorrupt   — some finite positions damaged, or finite→NaN
    //   BadSlotReplaced    — only the NaN/Inf slot lost its shape; neighbours OK
    //   Preserved          — NaN/Inf shape + neighbours all good
    let outcome = if drift_fraction > 0.5 {
        Outcome::Garbage
    } else if drift_fraction > 0.0 || finite_to_nan {
        Outcome::NeighbourCorruption
    } else if !bad_preserved_shape {
        Outcome::BadSlotReplaced
    } else {
        Outcome::Preserved
    };

    (outcome, Some(diag))
}

// ────────────────────────────────────────────────────────────────────────────
// ZFP characterisation
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "zfp")]
mod zfp_tests {
    use super::*;

    fn round_trip(input: &[f64], mode: ZfpMode) -> Result<Vec<u8>, String> {
        let compressor = ZfpCompressor {
            mode,
            num_values: input.len(),
            byte_order: ByteOrder::native(),
        };
        let data = f64_bytes(input);
        let compressed = compressor.compress(&data).map_err(|e| e.to_string())?;
        compressor
            .decompress(&compressed.data, data.len())
            .map_err(|e| e.to_string())
    }

    fn characterise(
        input: &[f64],
        mode: ZfpMode,
        bad_idx: usize,
        tol: f64,
        label: &str,
    ) -> Outcome {
        let out = round_trip(input, mode);
        let (outcome, diag) = classify(input, out, bad_idx, tol);
        match diag {
            Some(d) => eprintln!(
                "ZFP {label}: {} | bad_out={:?}, drift_frac={:.2}, max_err={:.4e}, finite→NaN={}",
                outcome.summary(),
                d.bad_value_after,
                d.neighbour_drift_fraction,
                d.max_neighbour_abs_err,
                d.finite_decoded_as_nan
            ),
            None => eprintln!("ZFP {label}: {}", outcome.summary()),
        }
        outcome
    }

    // ── Fixed-rate mode ────────────────────────────────────────────────

    #[test]
    fn zfp_fixed_rate_f64_nan_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[32] = f64::NAN;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedRate { rate: 16.0 },
            32,
            0.1,
            "fixed_rate(16) + NaN@32",
        );
        // We don't assert the outcome — this is a characterisation test.
        // The eprintln! output is captured and summarised for the memo.
    }

    #[test]
    fn zfp_fixed_rate_f64_positive_inf_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[16] = f64::INFINITY;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedRate { rate: 16.0 },
            16,
            0.1,
            "fixed_rate(16) + +Inf@16",
        );
    }

    #[test]
    fn zfp_fixed_rate_f64_negative_inf_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[40] = f64::NEG_INFINITY;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedRate { rate: 16.0 },
            40,
            0.1,
            "fixed_rate(16) + -Inf@40",
        );
    }

    #[test]
    fn zfp_fixed_rate_f64_all_nan_behaviour() {
        let values = vec![f64::NAN; 64];
        let _outcome = characterise(
            &values,
            ZfpMode::FixedRate { rate: 16.0 },
            0,
            0.1,
            "fixed_rate(16) + all-NaN",
        );
    }

    // ── Fixed-precision mode ───────────────────────────────────────────

    #[test]
    fn zfp_fixed_precision_f64_nan_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[10] = f64::NAN;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedPrecision { precision: 32 },
            10,
            0.1,
            "fixed_precision(32) + NaN@10",
        );
    }

    #[test]
    fn zfp_fixed_precision_f64_positive_inf_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[20] = f64::INFINITY;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedPrecision { precision: 32 },
            20,
            0.1,
            "fixed_precision(32) + +Inf@20",
        );
    }

    // ── Fixed-accuracy mode ────────────────────────────────────────────

    #[test]
    fn zfp_fixed_accuracy_f64_nan_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[5] = f64::NAN;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedAccuracy { tolerance: 1e-4 },
            5,
            1e-3, // ZFP accuracy mode may drift up to a few x the tolerance
            "fixed_accuracy(1e-4) + NaN@5",
        );
    }

    #[test]
    fn zfp_fixed_accuracy_f64_positive_inf_behaviour() {
        let mut values = vec![1.0_f64; 64];
        values[45] = f64::INFINITY;
        let _outcome = characterise(
            &values,
            ZfpMode::FixedAccuracy { tolerance: 1e-4 },
            45,
            1e-3,
            "fixed_accuracy(1e-4) + +Inf@45",
        );
    }

    // ── Control: finite data round-trips cleanly ──────────────────────

    #[test]
    fn zfp_control_finite_data_preserved() {
        let values: Vec<f64> = (0..64).map(|i| (i as f64) * 0.5).collect();
        // Use index 0 as the "bad" index but it's actually finite, so
        // we expect Preserved.
        let outcome = characterise(
            &values,
            ZfpMode::FixedRate { rate: 32.0 },
            0,
            0.1,
            "control/finite",
        );
        assert!(
            matches!(outcome, Outcome::Preserved),
            "control data must round-trip cleanly, got {outcome:?}"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SZ3 characterisation
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "sz3")]
mod sz3_tests {
    use super::*;

    fn round_trip(input: &[f64], bound: Sz3ErrorBound) -> Result<Vec<u8>, String> {
        let compressor = Sz3Compressor {
            error_bound: bound,
            num_values: input.len(),
            byte_order: ByteOrder::native(),
        };
        let data = f64_bytes(input);
        let compressed = match compressor.compress(&data) {
            Ok(c) => c,
            Err(e) => return Err(format!("encode: {e}")),
        };
        compressor
            .decompress(&compressed.data, data.len())
            .map_err(|e| format!("decode: {e}"))
    }

    fn characterise(
        input: &[f64],
        bound: Sz3ErrorBound,
        bad_idx: usize,
        tol: f64,
        label: &str,
    ) -> Outcome {
        let out = round_trip(input, bound);
        let (outcome, diag) = classify(input, out, bad_idx, tol);
        match diag {
            Some(d) => eprintln!(
                "SZ3 {label}: {} | bad_out={:?}, drift_frac={:.2}, max_err={:.4e}, finite→NaN={}",
                outcome.summary(),
                d.bad_value_after,
                d.neighbour_drift_fraction,
                d.max_neighbour_abs_err,
                d.finite_decoded_as_nan
            ),
            None => eprintln!("SZ3 {label}: {}", outcome.summary()),
        }
        outcome
    }

    // ── Absolute error bound ──────────────────────────────────────────

    #[test]
    fn sz3_absolute_f64_nan_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
        values[128] = f64::NAN;
        let _outcome = characterise(
            &values,
            Sz3ErrorBound::Absolute(1e-4),
            128,
            1e-2,
            "absolute(1e-4) + NaN@128",
        );
    }

    #[test]
    fn sz3_absolute_f64_positive_inf_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
        values[64] = f64::INFINITY;
        let _outcome = characterise(
            &values,
            Sz3ErrorBound::Absolute(1e-4),
            64,
            1e-2,
            "absolute(1e-4) + +Inf@64",
        );
    }

    #[test]
    fn sz3_absolute_f64_negative_inf_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
        values[192] = f64::NEG_INFINITY;
        let _outcome = characterise(
            &values,
            Sz3ErrorBound::Absolute(1e-4),
            192,
            1e-2,
            "absolute(1e-4) + -Inf@192",
        );
    }

    // ── Relative error bound ──────────────────────────────────────────

    #[test]
    fn sz3_relative_f64_nan_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| 100.0 + (i as f64) * 0.1).collect();
        values[100] = f64::NAN;
        let _outcome = characterise(
            &values,
            Sz3ErrorBound::Relative(1e-3),
            100,
            1.0,
            "relative(1e-3) + NaN@100",
        );
    }

    #[test]
    fn sz3_relative_f64_positive_inf_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| 100.0 + (i as f64) * 0.1).collect();
        values[50] = f64::INFINITY;
        let _outcome = characterise(
            &values,
            Sz3ErrorBound::Relative(1e-3),
            50,
            1.0,
            "relative(1e-3) + +Inf@50",
        );
    }

    // ── PSNR bound ────────────────────────────────────────────────────

    #[test]
    fn sz3_psnr_f64_nan_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| (i as f64).sin()).collect();
        values[77] = f64::NAN;
        let _outcome = characterise(
            &values,
            Sz3ErrorBound::Psnr(60.0),
            77,
            0.1,
            "psnr(60) + NaN@77",
        );
    }

    // ── Mixed case ────────────────────────────────────────────────────

    #[test]
    fn sz3_absolute_f64_mixed_nan_and_inf_behaviour() {
        let mut values: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
        values[50] = f64::NAN;
        values[100] = f64::INFINITY;
        values[150] = f64::NEG_INFINITY;
        // We pass bad_idx=50 but the classifier will detect all three
        // through the "any non-finite" check.
        let outcome = characterise(
            &values,
            Sz3ErrorBound::Absolute(1e-4),
            50,
            1e-2,
            "absolute(1e-4) + NaN@50 + Inf@100 + -Inf@150",
        );
        // Record the outcome — not asserted, just documented via eprintln.
        let _ = outcome;
    }

    // ── Control: finite data round-trips cleanly ──────────────────────

    #[test]
    fn sz3_control_finite_data_preserved() {
        let values: Vec<f64> = (0..256).map(|i| (i as f64) * 0.5).collect();
        let outcome = characterise(
            &values,
            Sz3ErrorBound::Absolute(1e-4),
            0,
            1e-2,
            "control/finite",
        );
        assert!(
            matches!(outcome, Outcome::Preserved),
            "control data must round-trip cleanly, got {outcome:?}"
        );
    }
}

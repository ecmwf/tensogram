// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use tensogram_benchmarks::codec_matrix::run_codec_matrix;
use tensogram_benchmarks::datagen::generate_weather_field;
use tensogram_benchmarks::report::{
    BenchmarkResult, Fidelity, TimingStats, compute_fidelity, format_table,
};

fn make_stats(ms: f64) -> TimingStats {
    TimingStats {
        median_ms: ms,
        min_ms: ms * 0.9,
        max_ms: ms * 1.1,
    }
}

// ── datagen ──────────────────────────────────────────────────────────────────

#[test]
fn datagen_determinism() {
    let a = generate_weather_field(1000, 42);
    let b = generate_weather_field(1000, 42);
    assert_eq!(a, b, "same seed must produce identical output");
}

#[test]
fn datagen_different_seeds_differ() {
    let a = generate_weather_field(1000, 1);
    let b = generate_weather_field(1000, 2);
    assert_ne!(a, b, "different seeds must produce different output");
}

#[test]
fn datagen_exact_length() {
    for n in [0usize, 1, 100, 999, 1000, 1024] {
        let v = generate_weather_field(n, 42);
        assert_eq!(v.len(), n, "length mismatch for n={n}");
    }
}

#[test]
fn datagen_physical_range() {
    let v = generate_weather_field(1000, 42);
    for (i, &val) in v.iter().enumerate() {
        assert!(val.is_finite(), "non-finite value at index {i}: {val}");
        assert!(
            (240.0..=320.0).contains(&val),
            "value out of physical range at index {i}: {val}"
        );
    }
}

// ── report ───────────────────────────────────────────────────────────────────

#[test]
fn report_formatting() {
    let results = vec![
        BenchmarkResult {
            name: "none+none".to_string(),
            encode: make_stats(1.0),
            decode: make_stats(0.5),
            compressed_bytes: 8000,
            original_bytes: 8000,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        },
        BenchmarkResult {
            name: "sp(24)+szip".to_string(),
            encode: make_stats(20.0),
            decode: make_stats(15.0),
            compressed_bytes: 3000,
            original_bytes: 8000,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Lossy {
                linf: 0.01,
                l1: 0.005,
                l2: 0.003,
            },
        },
    ];
    let table = format_table(&results, "none+none", "Test title");

    assert!(table.contains("[REF]"));
    assert!(table.contains("none+none [REF]"));
    assert!(table.contains("Enc (ms)"));
    assert!(table.contains("Ratio (%)"));
    assert!(table.contains("Fidelity"));
    assert!(table.contains("Reference: none+none"));
    assert!(table.contains("Test title"));
    assert!(table.contains("37.50"));
    assert!(table.contains("exact"));
}

#[test]
fn report_ref_not_found() {
    let results = vec![BenchmarkResult {
        name: "only".to_string(),
        encode: make_stats(1.0),
        decode: make_stats(1.0),
        compressed_bytes: 100,
        original_bytes: 100,
        compressed_bytes_varied: false,
        fidelity: Fidelity::Exact,
    }];
    let table = format_table(&results, "missing_ref", "Test");
    assert!(table.contains("Enc (ms)"));
}

// ── fidelity ─────────────────────────────────────────────────────────────────

#[test]
fn fidelity_exact_roundtrip() {
    let data = [1.0f64, 2.0, 3.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let fidelity = compute_fidelity(&bytes, &bytes, false);
    assert!(matches!(fidelity, Fidelity::Exact));
}

#[test]
fn fidelity_lossy_reports_error() {
    let orig = [1.0f64, 2.0, 3.0];
    let decoded = [1.01f64, 2.0, 3.0];
    let orig_bytes: Vec<u8> = orig.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dec_bytes: Vec<u8> = decoded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let fidelity = compute_fidelity(&orig_bytes, &dec_bytes, true);
    match fidelity {
        Fidelity::Lossy { linf, l1, l2 } => {
            assert!(linf > 0.0);
            assert!(l1 > 0.0);
            assert!(l2 > 0.0);
            assert!((linf - 0.01).abs() < 1e-10);
        }
        _ => panic!("expected Lossy fidelity"),
    }
}

// ── codec_matrix ─────────────────────────────────────────────────────────────

#[test]
fn codec_matrix_smoke() {
    let run = run_codec_matrix(1000, 1, 1, 42).expect("codec matrix must not error");

    assert_eq!(run.total_cases, 24, "expected 24 total cases attempted");
    assert_eq!(
        run.results.len() + run.failures.len(),
        run.total_cases,
        "results + failures must equal total_cases"
    );

    assert!(
        run.failures.is_empty(),
        "unexpected failures: {:?}",
        run.failures
    );

    assert_eq!(
        run.results[0].name, "none+none",
        "first result must be reference"
    );

    for r in &run.results {
        assert!(
            r.original_bytes > 0,
            "original_bytes must be > 0 for '{}'",
            r.name
        );
    }
}

#[test]
fn codec_matrix_fidelity() {
    let run = run_codec_matrix(1000, 1, 1, 42).expect("must succeed");

    for r in &run.results {
        match &r.fidelity {
            Fidelity::Exact => {}
            Fidelity::Lossy { linf, l1, l2 } => {
                assert!(linf.is_finite(), "non-finite Linf for '{}'", r.name);
                assert!(l1.is_finite(), "non-finite L1 for '{}'", r.name);
                assert!(l2.is_finite(), "non-finite L2 for '{}'", r.name);
                assert!(*linf >= 0.0);
                assert!(*l1 >= 0.0);
                assert!(*l2 >= 0.0);
            }
            Fidelity::Unchecked => panic!("fidelity must be checked for '{}'", r.name),
        }
    }

    let lossless_names = [
        "none+none",
        "none+zstd(3)",
        "none+lz4",
        "none+blosc2(blosclz)",
        "none+szip(32)",
    ];
    for name in lossless_names {
        if let Some(r) = run.results.iter().find(|r| r.name == name) {
            assert!(
                matches!(r.fidelity, Fidelity::Exact),
                "expected exact fidelity for lossless codec '{name}', got {:?}",
                r.fidelity
            );
        }
    }
}

#[test]
fn codec_matrix_result_names_unique() {
    let run = run_codec_matrix(500, 1, 1, 0).expect("must not error");
    let mut names: Vec<&str> = run.results.iter().map(|r| r.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        run.results.len(),
        "duplicate result names detected"
    );
}

// ── edge cases ───────────────────────────────────────────────────────────────

#[test]
fn codec_matrix_zero_points_returns_err() {
    let result = run_codec_matrix(0, 1, 1, 42);
    assert!(result.is_err(), "num_points=0 must return Err");
}

#[test]
fn codec_matrix_zero_iterations_returns_err() {
    let result = run_codec_matrix(100, 0, 1, 42);
    assert!(result.is_err(), "iterations=0 must return Err");
}

#[test]
fn codec_matrix_zero_warmup_returns_err() {
    let result = run_codec_matrix(100, 1, 0, 42);
    assert!(result.is_err(), "warmup=0 must return Err");
}

#[test]
fn codec_matrix_non_aligned_points() {
    let run =
        run_codec_matrix(501, 1, 1, 42).expect("non-aligned num_points must succeed (rounded up)");
    assert_eq!(run.results.len(), 24, "expected 24 results");
    assert!(
        run.failures.is_empty(),
        "unexpected failures: {:?}",
        run.failures
    );
}

#[test]
fn codec_matrix_single_point() {
    let run = run_codec_matrix(1, 1, 1, 42).expect("num_points=1 must succeed (rounded to 4)");
    assert_eq!(run.results.len(), 24, "expected 24 results");
    assert!(run.failures.is_empty());
    assert_eq!(
        run.results[0].original_bytes, 32,
        "expected 32 bytes for 4 f64 values"
    );
}

#[test]
fn codec_matrix_padded_original_bytes() {
    let run = run_codec_matrix(501, 1, 1, 42).expect("must succeed");
    assert_eq!(
        run.results[0].original_bytes,
        504 * 8,
        "original_bytes must reflect padded count (504 x 8)"
    );
}

#[test]
fn codec_matrix_error_message_content() {
    let err = run_codec_matrix(0, 1, 1, 42).unwrap_err();
    assert!(
        err.to_string().contains("num_points"),
        "error message should mention 'num_points': got '{err}'"
    );

    let err = run_codec_matrix(100, 0, 1, 42).unwrap_err();
    assert!(
        err.to_string().contains("iterations"),
        "error message should mention 'iterations': got '{err}'"
    );
}

// ── lib.rs entry points ──────────────────────────────────────────────────────

#[test]
fn run_codec_matrix_prints_table() {
    tensogram_benchmarks::run_codec_matrix(100, 1, 1, 42).expect("run_codec_matrix must not error");
}

#[test]
fn benchmark_error_display() {
    use tensogram_benchmarks::BenchmarkError;
    let e = BenchmarkError::Validation("test error".to_string());
    assert!(format!("{e}").contains("test error"));
}

#[test]
fn benchmark_run_all_passed() {
    use tensogram_benchmarks::BenchmarkRun;
    let run = BenchmarkRun {
        results: vec![],
        failures: vec![],
        total_cases: 0,
    };
    assert!(run.all_passed());
}

#[test]
fn benchmark_run_has_failures() {
    use tensogram_benchmarks::{BenchmarkRun, CaseFailure};
    let run = BenchmarkRun {
        results: vec![],
        failures: vec![CaseFailure {
            name: "test".to_string(),
            error: "broke".to_string(),
        }],
        total_cases: 1,
    };
    assert!(!run.all_passed());
}

// ── compressed_bytes_varied ──────────────────────────────────────────────────

#[test]
fn codec_matrix_deterministic_sizes() {
    let run = run_codec_matrix(1000, 3, 1, 42).expect("must succeed");
    for r in &run.results {
        assert!(
            !r.compressed_bytes_varied,
            "compressed size varied for '{}' — codec may be non-deterministic",
            r.name
        );
    }
}

// ── grib_comparison (eccodes feature-gated) ──────────────────────────────────

#[cfg(feature = "eccodes")]
#[test]
fn grib_comparison_zero_points_returns_err() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison;
    let result = run_grib_comparison(0, 1, 1, 42);
    assert!(result.is_err(), "num_points=0 must return Err");
}

#[cfg(feature = "eccodes")]
#[test]
fn grib_comparison_zero_iterations_returns_err() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison;
    let result = run_grib_comparison(100, 0, 1, 42);
    assert!(result.is_err(), "iterations=0 must return Err");
}

#[cfg(feature = "eccodes")]
#[test]
fn grib_comparison_smoke() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison;

    let run = run_grib_comparison(1000, 1, 1, 42).expect("grib comparison must not error");

    assert_eq!(run.total_cases, 3, "expected 3 total cases");
    assert_eq!(
        run.results.len() + run.failures.len(),
        run.total_cases,
        "results + failures must equal total_cases"
    );
    assert!(
        run.failures.is_empty(),
        "unexpected failures: {:?}",
        run.failures
    );

    assert_eq!(
        run.results[0].name, "eccodes grid_ccsds",
        "first result must be reference"
    );

    for r in &run.results {
        assert!(
            r.compressed_bytes > 0,
            "compressed_bytes must be > 0 for '{}'",
            r.name
        );
    }
}

#[cfg(feature = "eccodes")]
#[test]
fn grib_comparison_fidelity() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison;

    let run = run_grib_comparison(1000, 1, 1, 42).expect("must succeed");
    for r in &run.results {
        match &r.fidelity {
            Fidelity::Lossy { linf, l1, l2 } => {
                assert!(linf.is_finite(), "non-finite Linf for '{}'", r.name);
                assert!(l1.is_finite(), "non-finite L1 for '{}'", r.name);
                assert!(l2.is_finite(), "non-finite L2 for '{}'", r.name);
            }
            Fidelity::Exact => {}
            Fidelity::Unchecked => panic!("fidelity must be checked for '{}'", r.name),
        }
    }
}

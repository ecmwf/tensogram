//! Smoke tests for the tensogram-benchmarks crate.
//!
//! These tests use small data sizes (1000 points, 1 iteration) so they
//! run in milliseconds on CI, verifying that each benchmark module compiles,
//! runs, and produces well-formed output without errors.

use tensogram_benchmarks::codec_matrix::run_codec_matrix_results;
use tensogram_benchmarks::datagen::generate_weather_field;
use tensogram_benchmarks::report::{format_table, BenchmarkResult};

// ── datagen ───────────────────────────────────────────────────────────────────

#[test]
fn test_datagen_determinism() {
    let a = generate_weather_field(1000, 42);
    let b = generate_weather_field(1000, 42);
    assert_eq!(a, b, "same seed must produce identical output");
}

#[test]
fn test_datagen_different_seeds_differ() {
    let a = generate_weather_field(1000, 1);
    let b = generate_weather_field(1000, 2);
    assert_ne!(a, b, "different seeds must produce different output");
}

#[test]
fn test_datagen_exact_length() {
    for n in [0usize, 1, 100, 999, 1000, 1024] {
        let v = generate_weather_field(n, 42);
        assert_eq!(v.len(), n, "length mismatch for n={n}");
    }
}

#[test]
fn test_datagen_physical_range() {
    let v = generate_weather_field(1000, 42);
    for (i, &val) in v.iter().enumerate() {
        assert!(val.is_finite(), "non-finite value at index {i}: {val}");
        assert!(
            (240.0..=320.0).contains(&val),
            "value out of physical range at index {i}: {val}"
        );
    }
}

// ── report ────────────────────────────────────────────────────────────────────

#[test]
fn test_report_formatting() {
    let results = vec![
        BenchmarkResult {
            name: "none+none".to_string(),
            encode_ms: 1.0,
            decode_ms: 0.5,
            compressed_bytes: 8000,
            original_bytes: 8000,
        },
        BenchmarkResult {
            name: "sp(24)+szip".to_string(),
            encode_ms: 20.0,
            decode_ms: 15.0,
            compressed_bytes: 3000,
            original_bytes: 8000,
        },
    ];
    let table = format_table(&results, "none+none", "Test title");

    assert!(
        table.contains("[REF]"),
        "reference row must be marked [REF]"
    );
    assert!(
        table.contains("none+none [REF]"),
        "reference name must have [REF] suffix"
    );
    assert!(table.contains("Encode (ms)"), "header missing");
    assert!(table.contains("Ratio (%)"), "ratio column missing");
    assert!(
        table.contains("Reference: none+none"),
        "reference line missing"
    );
    assert!(table.contains("Test title"), "title missing");
    // sp(24)+szip should show 37.5% ratio (3000/8000)
    assert!(table.contains("37.50"), "expected 37.50% ratio");
}

#[test]
fn test_report_ref_not_found() {
    // Table should still render even if reference_name doesn't match any result.
    let results = vec![BenchmarkResult {
        name: "only".to_string(),
        encode_ms: 1.0,
        decode_ms: 1.0,
        compressed_bytes: 100,
        original_bytes: 100,
    }];
    let table = format_table(&results, "missing_ref", "Test");
    // Must not panic and must contain the column headers.
    assert!(table.contains("Encode (ms)"));
    assert!(table.contains("N/A")); // vs-ref columns should be N/A
}

// ── codec_matrix ──────────────────────────────────────────────────────────────

#[test]
fn test_codec_matrix_smoke() {
    // Run with 1000 points and 1 iteration — completes in < 5 seconds.
    let results =
        run_codec_matrix_results(1000, 1, 42).expect("codec matrix must not return an error");

    // Should produce 24 results (one per combo).
    assert_eq!(results.len(), 24, "expected 24 benchmark results");

    // First result is the reference (none+none).
    assert_eq!(
        results[0].name, "none+none",
        "first result must be the reference"
    );

    // All results must have positive original_bytes.
    for r in &results {
        assert!(
            r.original_bytes > 0,
            "original_bytes must be > 0 for '{}'",
            r.name
        );
    }

    // None of the results should be ERROR-tagged (compression should succeed
    // for all 24 combos on this size of data).
    for r in &results {
        assert!(
            !r.name.contains("[ERROR]"),
            "unexpected error in result: '{}'",
            r.name
        );
    }

    // Compression ratios: sp(16) should give a ratio below 35%.
    let sp16 = results
        .iter()
        .find(|r| r.name == "sp(16)+none")
        .expect("sp(16)+none result missing");
    assert!(
        sp16.ratio_pct() < 35.0,
        "sp(16) ratio {} should be below 35%",
        sp16.ratio_pct()
    );
}

#[test]
fn test_codec_matrix_result_names_unique() {
    let results = run_codec_matrix_results(500, 1, 0).expect("must not error");
    let mut names: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        results.len(),
        "duplicate result names detected"
    );
}

// ── edge cases ────────────────────────────────────────────────────────────────

#[test]
fn test_codec_matrix_zero_points_returns_err() {
    let result = run_codec_matrix_results(0, 1, 42);
    assert!(result.is_err(), "num_points=0 must return Err");
}

#[test]
fn test_codec_matrix_zero_iterations_returns_err() {
    let result = run_codec_matrix_results(100, 0, 42);
    assert!(result.is_err(), "iterations=0 must return Err");
}

#[test]
fn test_codec_matrix_non_aligned_points() {
    // 501 is not a multiple of 4.  The codec matrix rounds up internally
    // so that szip cases never fail due to alignment.
    let results = run_codec_matrix_results(501, 1, 42)
        .expect("non-aligned num_points must succeed (rounded up)");
    assert_eq!(results.len(), 24, "expected 24 results");
    for r in &results {
        assert!(
            !r.name.contains("[ERROR]"),
            "unexpected error for non-aligned size in '{}'",
            r.name
        );
    }
}

#[test]
fn test_codec_matrix_single_point() {
    // num_points=1 rounds to 4. All 24 codecs should handle tiny data.
    let results =
        run_codec_matrix_results(1, 1, 42).expect("num_points=1 must succeed (rounded to 4)");
    assert_eq!(results.len(), 24, "expected 24 results");
    for r in &results {
        assert!(
            !r.name.contains("[ERROR]"),
            "unexpected error for tiny data in '{}'",
            r.name
        );
    }
    // original_bytes should be 4 * 8 = 32 (padded to 4 values).
    assert_eq!(
        results[0].original_bytes, 32,
        "expected 32 bytes for 4 f64 values"
    );
}

#[test]
fn test_codec_matrix_padded_original_bytes() {
    // 501 rounds to 504. original_bytes = 504 * 8 = 4032.
    let results = run_codec_matrix_results(501, 1, 42).expect("must succeed");
    assert_eq!(
        results[0].original_bytes,
        504 * 8,
        "original_bytes must reflect padded count (504 × 8)"
    );
}

#[test]
fn test_codec_matrix_error_message_content() {
    let err = run_codec_matrix_results(0, 1, 42).unwrap_err();
    assert!(
        err.to_string().contains("num_points"),
        "error message should mention 'num_points': got '{err}'"
    );

    let err = run_codec_matrix_results(100, 0, 42).unwrap_err();
    assert!(
        err.to_string().contains("iterations"),
        "error message should mention 'iterations': got '{err}'"
    );
}

// ── lib.rs entry points ──────────────────────────────────────────────────────

#[test]
fn test_run_codec_matrix_prints_table() {
    // Just verify the top-level entry point doesn't panic.
    tensogram_benchmarks::run_codec_matrix(100, 1, 42).expect("run_codec_matrix must not error");
}

#[test]
fn test_benchmark_error_display() {
    use tensogram_benchmarks::BenchmarkError;
    let e = BenchmarkError("test error".to_string());
    assert_eq!(format!("{e}"), "test error");
}

#[test]
fn test_benchmark_error_from_string() {
    use tensogram_benchmarks::BenchmarkError;
    let e: BenchmarkError = String::from("from string").into();
    assert_eq!(e.0, "from string");
}

// ── grib_comparison (eccodes feature-gated) ──────────────────────────────────

#[cfg(feature = "eccodes")]
#[test]
fn test_grib_comparison_zero_points_returns_err() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison_results;
    let result = run_grib_comparison_results(0, 1, 42);
    assert!(result.is_err(), "num_points=0 must return Err");
}

#[cfg(feature = "eccodes")]
#[test]
fn test_grib_comparison_zero_iterations_returns_err() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison_results;
    let result = run_grib_comparison_results(100, 0, 42);
    assert!(result.is_err(), "iterations=0 must return Err");
}

#[cfg(feature = "eccodes")]
#[test]
fn test_grib_comparison_smoke() {
    use tensogram_benchmarks::grib_comparison::run_grib_comparison_results;

    let results = run_grib_comparison_results(1000, 1, 42).expect("grib comparison must not error");

    // Expect exactly 3 results.
    assert_eq!(results.len(), 3, "expected 3 results");

    // First result must be the eccodes CCSDS reference.
    assert_eq!(
        results[0].name, "eccodes grid_ccsds",
        "first result must be eccodes grid_ccsds"
    );

    // No results should be ERROR-tagged.
    for r in &results {
        assert!(
            !r.name.contains("[ERROR]"),
            "unexpected error in grib result: '{}'",
            r.name
        );
    }

    // All should have positive compressed_bytes.
    for r in &results {
        assert!(
            r.compressed_bytes > 0,
            "compressed_bytes must be > 0 for '{}'",
            r.name
        );
    }
}

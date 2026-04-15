// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use crate::BenchmarkRun;

// ── Fidelity ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Fidelity {
    Exact,
    Lossy { linf: f64, l1: f64, l2: f64 },
    Unchecked,
}

impl std::fmt::Display for Fidelity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fidelity::Exact => write!(f, "exact"),
            Fidelity::Lossy { linf, l1, l2 } => {
                write!(f, "Linf={linf:.2e} L1={l1:.2e} L2={l2:.2e}")
            }
            Fidelity::Unchecked => write!(f, "\u{2014}"),
        }
    }
}

// ── Timing ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TimingStats {
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

pub fn compute_timing_stats(samples: &mut [u64]) -> TimingStats {
    if samples.is_empty() {
        return TimingStats {
            median_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
        };
    }
    samples.sort_unstable();
    // Safe: emptiness is guarded above, so both first and last element exist.
    let last_idx = samples.len() - 1;
    TimingStats {
        median_ms: ns_to_ms(median_of_sorted(samples)),
        min_ms: ns_to_ms(samples[0]),
        max_ms: ns_to_ms(samples[last_idx]),
    }
}

/// Median of a pre-sorted slice. Returns 0 for empty input.
fn median_of_sorted(sorted: &[u64]) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        ((sorted[mid - 1] as u128 + sorted[mid] as u128) / 2) as u64
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
fn median_ns(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    median_of_sorted(samples)
}

pub fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

// ── Fidelity computation ─────────────────────────────────────────────────────

/// Compare decoded output to original input bytes (little-endian f64).
///
/// `is_lossy` controls the expectation: lossy codecs report error metrics,
/// lossless codecs that mismatch are treated as lossy (bug indicator).
pub fn compute_fidelity(original_bytes: &[u8], decoded_bytes: &[u8], is_lossy: bool) -> Fidelity {
    let orig_len = original_bytes.len();
    let dec_len = decoded_bytes.len();

    if orig_len == 0
        || !orig_len.is_multiple_of(8)
        || dec_len < orig_len
        || !dec_len.is_multiple_of(8)
    {
        return Fidelity::Unchecked;
    }

    let n = orig_len / 8;

    let bytes_match = original_bytes[..n * 8] == decoded_bytes[..n * 8];

    if !is_lossy && bytes_match {
        return Fidelity::Exact;
    }

    let mut max_abs_error: f64 = 0.0;
    let mut sum_abs_error: f64 = 0.0;
    let mut sum_sq_error: f64 = 0.0;

    for (orig_chunk, dec_chunk) in original_bytes
        .chunks_exact(8)
        .zip(decoded_bytes.chunks_exact(8))
    {
        let orig = f64::from_le_bytes(orig_chunk.try_into().unwrap_or([0; 8]));
        let dec = f64::from_le_bytes(dec_chunk.try_into().unwrap_or([0; 8]));
        let err = (orig - dec).abs();
        max_abs_error = max_abs_error.max(err);
        sum_abs_error += err;
        sum_sq_error += err * err;
    }

    if bytes_match {
        Fidelity::Exact
    } else {
        Fidelity::Lossy {
            linf: max_abs_error,
            l1: sum_abs_error / n as f64,
            l2: (sum_sq_error / n as f64).sqrt(),
        }
    }
}

// ── Result ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub encode: TimingStats,
    pub decode: TimingStats,
    pub compressed_bytes: usize,
    pub original_bytes: usize,
    pub compressed_bytes_varied: bool,
    pub fidelity: Fidelity,
}

impl BenchmarkResult {
    pub fn ratio_pct(&self) -> f64 {
        if self.original_bytes == 0 {
            return 100.0;
        }
        100.0 * self.compressed_bytes as f64 / self.original_bytes as f64
    }

    pub fn size_kib(&self) -> f64 {
        self.compressed_bytes as f64 / 1024.0
    }

    /// Encode throughput in MB/s (uncompressed input size / median encode time).
    pub fn encode_throughput_mbs(&self) -> f64 {
        if self.encode.median_ms <= 0.0 {
            return 0.0;
        }
        let mb = self.original_bytes as f64 / (1024.0 * 1024.0);
        let secs = self.encode.median_ms / 1000.0;
        mb / secs
    }

    /// Decode throughput in MB/s (uncompressed output size / median decode time).
    pub fn decode_throughput_mbs(&self) -> f64 {
        if self.decode.median_ms <= 0.0 {
            return 0.0;
        }
        let mb = self.original_bytes as f64 / (1024.0 * 1024.0);
        let secs = self.decode.median_ms / 1000.0;
        mb / secs
    }
}

// ── Table formatting ─────────────────────────────────────────────────────────

pub fn format_table(results: &[BenchmarkResult], reference_name: &str, title: &str) -> String {
    let ref_result = results.iter().find(|r| r.name == reference_name);

    struct Row {
        name: String,
        encode_ms: String,
        decode_ms: String,
        enc_mbs: String,
        dec_mbs: String,
        ratio_pct: String,
        size_kib: String,
        fidelity: String,
    }

    let rows: Vec<Row> = results
        .iter()
        .map(|r| {
            let display_name = if r.name == reference_name {
                format!("{} [REF]", r.name)
            } else {
                r.name.clone()
            };

            let size_note = if r.compressed_bytes_varied { "~" } else { "" };

            Row {
                name: display_name,
                encode_ms: format!("{:.3}", r.encode.median_ms),
                decode_ms: format!("{:.3}", r.decode.median_ms),
                enc_mbs: format!("{:.1}", r.encode_throughput_mbs()),
                dec_mbs: format!("{:.1}", r.decode_throughput_mbs()),
                ratio_pct: format!("{}{:.2}", size_note, r.ratio_pct()),
                size_kib: format!("{}{:.1}", size_note, r.size_kib()),
                fidelity: r.fidelity.to_string(),
            }
        })
        .collect();

    let headers = [
        "Combo",
        "Enc (ms)",
        "Dec (ms)",
        "Enc MB/s",
        "Dec MB/s",
        "Ratio (%)",
        "Size (KiB)",
        "Fidelity",
    ];

    let widths: [usize; 8] = {
        let mut w = [0usize; 8];
        for (i, h) in headers.iter().enumerate() {
            w[i] = h.len();
        }
        for row in &rows {
            let cells = [
                row.name.as_str(),
                row.encode_ms.as_str(),
                row.decode_ms.as_str(),
                row.enc_mbs.as_str(),
                row.dec_mbs.as_str(),
                row.ratio_pct.as_str(),
                row.size_kib.as_str(),
                row.fidelity.as_str(),
            ];
            for (i, cell) in cells.iter().enumerate() {
                w[i] = w[i].max(cell.len());
            }
        }
        w
    };

    let mut out = String::new();

    out.push_str(title);
    out.push('\n');
    if let Some(rr) = ref_result {
        out.push_str(&format!(
            "Reference: {} ({:.1} MB/s enc, {:.1} MB/s dec)\n",
            reference_name,
            rr.encode_throughput_mbs(),
            rr.decode_throughput_mbs()
        ));
    } else {
        out.push_str(&format!("Reference: {reference_name}\n"));
    }
    out.push('\n');

    out.push_str(&format!(
        " {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} | {:>w6$} | {:>w7$}",
        headers[0],
        headers[1],
        headers[2],
        headers[3],
        headers[4],
        headers[5],
        headers[6],
        headers[7],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
        w3 = widths[3],
        w4 = widths[4],
        w5 = widths[5],
        w6 = widths[6],
        w7 = widths[7],
    ));
    out.push('\n');

    let sep: String = widths
        .iter()
        .enumerate()
        .map(|(i, &w)| "-".repeat(w + if i == 0 { 2 } else { 3 }))
        .collect::<Vec<_>>()
        .join("+");
    out.push_str(&sep);
    out.push('\n');

    for row in &rows {
        out.push_str(&format!(
            " {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} | {:>w6$} | {:>w7$}",
            row.name,
            row.encode_ms,
            row.decode_ms,
            row.enc_mbs,
            row.dec_mbs,
            row.ratio_pct,
            row.size_kib,
            row.fidelity,
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
            w3 = widths[3],
            w4 = widths[4],
            w5 = widths[5],
            w6 = widths[6],
            w7 = widths[7],
        ));
        out.push('\n');
    }

    out
}

/// Print full benchmark report: table, failures, and summary.
pub fn print_report(run: &BenchmarkRun, reference_name: &str, title: &str) {
    print!("{}", format_table(&run.results, reference_name, title));

    if !run.failures.is_empty() {
        eprintln!("\nFailed cases ({}):", run.failures.len());
        for f in &run.failures {
            eprintln!("  \u{2717} {f}");
        }
    }

    let vary_count = run
        .results
        .iter()
        .filter(|r| r.compressed_bytes_varied)
        .count();
    if vary_count > 0 {
        eprintln!(
            "\nNote: {vary_count} case(s) had variable compressed sizes across iterations (marked ~)."
        );
    }

    eprintln!(
        "\nSummary: {} attempted, {} succeeded, {} failed",
        run.total_cases,
        run.results.len(),
        run.failures.len()
    );
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(median: f64) -> TimingStats {
        TimingStats {
            median_ms: median,
            min_ms: median * 0.9,
            max_ms: median * 1.1,
        }
    }

    fn make_results() -> Vec<BenchmarkResult> {
        vec![
            BenchmarkResult {
                name: "none+none".to_string(),
                encode: make_stats(0.5),
                decode: make_stats(0.3),
                compressed_bytes: 8000,
                original_bytes: 8000,
                compressed_bytes_varied: false,
                fidelity: Fidelity::Exact,
            },
            BenchmarkResult {
                name: "sp(24)+szip".to_string(),
                encode: make_stats(10.0),
                decode: make_stats(5.0),
                compressed_bytes: 3000,
                original_bytes: 8000,
                compressed_bytes_varied: false,
                fidelity: Fidelity::Lossy {
                    linf: 0.01,
                    l1: 0.005,
                    l2: 0.003,
                },
            },
        ]
    }

    #[test]
    fn reference_marked() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        assert!(table.contains("[REF]"));
        assert!(table.contains("none+none [REF]"));
    }

    #[test]
    fn headers_present() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        assert!(table.contains("Enc (ms)"));
        assert!(table.contains("Dec (ms)"));
        assert!(table.contains("Enc MB/s"));
        assert!(table.contains("Ratio (%)"));
        assert!(table.contains("Fidelity"));
    }

    #[test]
    fn reference_line() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        assert!(table.contains("Reference: none+none"));
    }

    #[test]
    fn ratio_calculation() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode: make_stats(1.0),
            decode: make_stats(1.0),
            compressed_bytes: 3000,
            original_bytes: 8000,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        };
        let ratio = r.ratio_pct();
        assert!((ratio - 37.5).abs() < 0.001, "ratio {ratio} != 37.5");
    }

    #[test]
    fn zero_original_bytes() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode: make_stats(1.0),
            decode: make_stats(1.0),
            compressed_bytes: 0,
            original_bytes: 0,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        };
        assert_eq!(r.ratio_pct(), 100.0);
    }

    #[test]
    fn throughput_calculation() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode: make_stats(1000.0), // 1 second
            decode: make_stats(500.0),  // 0.5 seconds
            compressed_bytes: 0,
            original_bytes: 1024 * 1024, // 1 MiB
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        };
        assert!((r.encode_throughput_mbs() - 1.0).abs() < 0.001);
        assert!((r.decode_throughput_mbs() - 2.0).abs() < 0.001);
    }

    #[test]
    fn zero_time_throughput() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode: make_stats(0.0),
            decode: make_stats(0.0),
            compressed_bytes: 0,
            original_bytes: 1024,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        };
        assert_eq!(r.encode_throughput_mbs(), 0.0);
        assert_eq!(r.decode_throughput_mbs(), 0.0);
    }

    #[test]
    fn fidelity_display() {
        assert_eq!(format!("{}", Fidelity::Exact), "exact");
        assert_eq!(format!("{}", Fidelity::Unchecked), "\u{2014}");
        let lossy = Fidelity::Lossy {
            linf: 0.01,
            l1: 0.005,
            l2: 0.003,
        };
        let s = format!("{lossy}");
        assert!(s.starts_with("Linf="));
    }

    #[test]
    fn compute_fidelity_exact() {
        let data = [1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fidelity = compute_fidelity(&bytes, &bytes, false);
        assert!(matches!(fidelity, Fidelity::Exact));
    }

    #[test]
    fn compute_fidelity_lossy() {
        let orig = [1.0f64, 2.0, 3.0];
        let decoded = [1.01f64, 2.02, 3.03];
        let orig_bytes: Vec<u8> = orig.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec_bytes: Vec<u8> = decoded.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fidelity = compute_fidelity(&orig_bytes, &dec_bytes, true);
        match fidelity {
            Fidelity::Lossy { linf, l1, l2 } => {
                assert!(linf > 0.0);
                assert!(l1 > 0.0);
                assert!(l2 > 0.0);
                assert!((linf - 0.03).abs() < 0.001);
            }
            _ => panic!("expected Lossy fidelity"),
        }
    }

    #[test]
    fn compute_fidelity_empty() {
        let fidelity = compute_fidelity(&[], &[], false);
        assert!(matches!(fidelity, Fidelity::Unchecked));
    }

    #[test]
    fn compute_fidelity_misaligned_input() {
        let fidelity = compute_fidelity(&[0u8; 7], &[0u8; 7], false);
        assert!(matches!(fidelity, Fidelity::Unchecked));
    }

    #[test]
    fn compute_fidelity_decoded_too_short() {
        let orig = [1.0f64];
        let orig_bytes: Vec<u8> = orig.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fidelity = compute_fidelity(&orig_bytes, &[0u8; 4], false);
        assert!(matches!(fidelity, Fidelity::Unchecked));
    }

    #[test]
    fn compute_fidelity_lossy_returns_exact_when_identical() {
        let data = [1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fidelity = compute_fidelity(&bytes, &bytes, true);
        assert!(matches!(fidelity, Fidelity::Exact));
    }

    #[test]
    fn median_ns_odd() {
        let mut s = [3u64, 1, 4, 1, 5];
        assert_eq!(median_ns(&mut s), 3);
    }

    #[test]
    fn median_ns_even() {
        let mut s = [1u64, 2, 3, 4];
        assert_eq!(median_ns(&mut s), 2);
    }

    #[test]
    fn median_ns_single() {
        let mut s = [42u64];
        assert_eq!(median_ns(&mut s), 42);
    }

    #[test]
    fn median_ns_empty() {
        let mut s: [u64; 0] = [];
        assert_eq!(median_ns(&mut s), 0);
    }

    #[test]
    fn median_ns_large_values() {
        let mut s = [u64::MAX, u64::MAX];
        assert_eq!(median_ns(&mut s), u64::MAX);
    }

    #[test]
    fn median_ns_two_elements() {
        let mut s = [10u64, 20];
        assert_eq!(median_ns(&mut s), 15);
    }

    #[test]
    fn ns_to_ms_conversion() {
        assert!((ns_to_ms(1_000_000) - 1.0).abs() < f64::EPSILON);
        assert!((ns_to_ms(0) - 0.0).abs() < f64::EPSILON);
        assert!((ns_to_ms(500_000) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn size_kib() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode: make_stats(1.0),
            decode: make_stats(1.0),
            compressed_bytes: 2048,
            original_bytes: 4096,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        };
        assert!((r.size_kib() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ratio_above_100() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode: make_stats(1.0),
            decode: make_stats(1.0),
            compressed_bytes: 10000,
            original_bytes: 8000,
            compressed_bytes_varied: false,
            fidelity: Fidelity::Exact,
        };
        let ratio = r.ratio_pct();
        assert!((ratio - 125.0).abs() < 0.001, "expected 125%, got {ratio}");
    }

    #[test]
    fn format_table_empty_results() {
        let table = format_table(&[], "ref", "Empty");
        assert!(table.contains("Combo"));
        assert!(table.contains("Empty"));
    }

    #[test]
    fn varied_size_marker() {
        let results = vec![BenchmarkResult {
            name: "varied".to_string(),
            encode: make_stats(1.0),
            decode: make_stats(1.0),
            compressed_bytes: 1000,
            original_bytes: 2000,
            compressed_bytes_varied: true,
            fidelity: Fidelity::Exact,
        }];
        let table = format_table(&results, "varied", "Test");
        assert!(table.contains("~"), "varied sizes should be marked with ~");
    }

    #[test]
    fn timing_stats_from_samples() {
        let mut samples = [100_000u64, 200_000, 300_000, 400_000, 500_000];
        let stats = compute_timing_stats(&mut samples);
        assert!((stats.median_ms - 0.3).abs() < 0.001);
        assert!((stats.min_ms - 0.1).abs() < 0.001);
        assert!((stats.max_ms - 0.5).abs() < 0.001);
    }

    #[test]
    fn timing_stats_empty() {
        let stats = compute_timing_stats(&mut []);
        assert_eq!(stats.median_ms, 0.0);
        assert_eq!(stats.min_ms, 0.0);
        assert_eq!(stats.max_ms, 0.0);
    }

    #[test]
    fn print_report_with_failures() {
        let run = BenchmarkRun {
            results: make_results(),
            failures: vec![crate::CaseFailure {
                name: "broken+codec".to_string(),
                error: "unsupported".to_string(),
            }],
            total_cases: 3,
        };
        print_report(&run, "none+none", "Test");
    }
}

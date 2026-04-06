//! ASCII table reporter for benchmark results.
//!
//! Formats a list of [`BenchmarkResult`] values into a human-readable table
//! with reference-relative comparison columns. The reference row is marked
//! with `[REF]` and the "vs Ref" columns show how many times faster or slower
//! each variant is compared to the reference.

// ── Types ─────────────────────────────────────────────────────────────────────

/// Timing and size results for one benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Short display name, e.g. `"none+none"` or `"sp(24)+szip"`.
    pub name: String,
    /// Median encode time across all iterations, in milliseconds.
    pub encode_ms: f64,
    /// Median decode time across all iterations, in milliseconds.
    pub decode_ms: f64,
    /// Compressed payload size in bytes.
    pub compressed_bytes: usize,
    /// Original (uncompressed) payload size in bytes.
    pub original_bytes: usize,
}

impl BenchmarkResult {
    /// Compression ratio as a percentage: 100 % × compressed / original.
    ///
    /// Returns 100.0 when original_bytes is zero.
    pub fn ratio_pct(&self) -> f64 {
        if self.original_bytes == 0 {
            return 100.0;
        }
        100.0 * self.compressed_bytes as f64 / self.original_bytes as f64
    }

    /// Compressed size in kibibytes.
    pub fn size_kib(&self) -> f64 {
        self.compressed_bytes as f64 / 1024.0
    }
}

// ── Timing helpers ────────────────────────────────────────────────────────────

/// Compute the median of a mutable slice of durations (in nanoseconds).
///
/// Returns 0 if the slice is empty.
pub fn median_ns(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        // Average of the two middle values (avoids overflow via u128).
        ((samples[mid - 1] as u128 + samples[mid] as u128) / 2) as u64
    } else {
        samples[mid]
    }
}

/// Convert nanoseconds to milliseconds.
pub fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

// ── Table formatting ──────────────────────────────────────────────────────────

/// Format benchmark results as an ASCII table string.
///
/// `reference_name` must match the `.name` field of exactly one result; that
/// row is marked `[REF]` and used as the baseline for the "vs Ref" columns.
/// If no matching row is found the function still produces a valid table and
/// keeps the reference-relative columns, filling their cells with `N/A`.
pub fn format_table(results: &[BenchmarkResult], reference_name: &str, title: &str) -> String {
    // Find the reference row.
    let ref_result = results.iter().find(|r| r.name == reference_name);

    // Build display rows: (name, encode_ms, decode_ms, ratio, kib, vs_enc, vs_dec)
    struct Row {
        name: String,
        encode_ms: String,
        decode_ms: String,
        ratio_pct: String,
        size_kib: String,
        vs_enc: String,
        vs_dec: String,
    }

    let rows: Vec<Row> = results
        .iter()
        .map(|r| {
            let is_ref = r.name == reference_name;
            let display_name = if is_ref {
                format!("{} [REF]", r.name)
            } else {
                r.name.clone()
            };

            let vs_enc = match ref_result {
                Some(ref_r) if ref_r.encode_ms > 0.0 => {
                    format!("{:.2}x", r.encode_ms / ref_r.encode_ms)
                }
                Some(_) => "N/A".to_string(),
                None => "N/A".to_string(),
            };
            let vs_dec = match ref_result {
                Some(ref_r) if ref_r.decode_ms > 0.0 => {
                    format!("{:.2}x", r.decode_ms / ref_r.decode_ms)
                }
                Some(_) => "N/A".to_string(),
                None => "N/A".to_string(),
            };

            Row {
                name: display_name,
                encode_ms: format!("{:.3}", r.encode_ms),
                decode_ms: format!("{:.3}", r.decode_ms),
                ratio_pct: format!("{:.2}", r.ratio_pct()),
                size_kib: format!("{:.1}", r.size_kib()),
                vs_enc,
                vs_dec,
            }
        })
        .collect();

    // Column headers
    let headers = [
        "Combo",
        "Encode (ms)",
        "Decode (ms)",
        "Ratio (%)",
        "Size (KiB)",
        "vs Ref Enc",
        "vs Ref Dec",
    ];

    // Compute column widths from max of header and data.
    let widths: [usize; 7] = {
        let mut w = [0usize; 7];
        for (i, h) in headers.iter().enumerate() {
            w[i] = h.len();
        }
        for row in &rows {
            let cells = [
                row.name.as_str(),
                row.encode_ms.as_str(),
                row.decode_ms.as_str(),
                row.ratio_pct.as_str(),
                row.size_kib.as_str(),
                row.vs_enc.as_str(),
                row.vs_dec.as_str(),
            ];
            for (i, cell) in cells.iter().enumerate() {
                w[i] = w[i].max(cell.len());
            }
        }
        w
    };

    let mut out = String::new();

    // Title line
    out.push_str(title);
    out.push('\n');

    // Reference line
    out.push_str(&format!("Reference: {reference_name}\n"));
    out.push('\n');

    // Header row
    let header_line = format!(
        " {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} | {:>w6$}",
        headers[0],
        headers[1],
        headers[2],
        headers[3],
        headers[4],
        headers[5],
        headers[6],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
        w3 = widths[3],
        w4 = widths[4],
        w5 = widths[5],
        w6 = widths[6],
    );
    out.push_str(&header_line);
    out.push('\n');

    // Separator
    let sep: String = widths
        .iter()
        .enumerate()
        .map(|(i, &w)| "-".repeat(w + if i == 0 { 2 } else { 3 }))
        .collect::<Vec<_>>()
        .join("+");
    out.push_str(&sep);
    out.push('\n');

    // Data rows
    for row in &rows {
        let line = format!(
            " {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} | {:>w6$}",
            row.name,
            row.encode_ms,
            row.decode_ms,
            row.ratio_pct,
            row.size_kib,
            row.vs_enc,
            row.vs_dec,
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
            w3 = widths[3],
            w4 = widths[4],
            w5 = widths[5],
            w6 = widths[6],
        );
        out.push_str(&line);
        out.push('\n');
    }

    out
}

/// Print benchmark results to stdout.
pub fn print_table(results: &[BenchmarkResult], reference_name: &str, title: &str) {
    print!("{}", format_table(results, reference_name, title));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results() -> Vec<BenchmarkResult> {
        vec![
            BenchmarkResult {
                name: "none+none".to_string(),
                encode_ms: 0.5,
                decode_ms: 0.3,
                compressed_bytes: 8000,
                original_bytes: 8000,
            },
            BenchmarkResult {
                name: "sp(24)+szip".to_string(),
                encode_ms: 10.0,
                decode_ms: 5.0,
                compressed_bytes: 3000,
                original_bytes: 8000,
            },
        ]
    }

    #[test]
    fn test_reference_marked() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        assert!(
            table.contains("[REF]"),
            "reference row must be marked [REF]"
        );
        assert!(
            table.contains("none+none [REF]"),
            "reference name must appear with [REF] suffix"
        );
    }

    #[test]
    fn test_headers_present() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        assert!(table.contains("Encode (ms)"), "header Encode (ms) missing");
        assert!(table.contains("Decode (ms)"), "header Decode (ms) missing");
        assert!(table.contains("Ratio (%)"), "header Ratio (%) missing");
        assert!(table.contains("Size (KiB)"), "header Size (KiB) missing");
        assert!(table.contains("vs Ref Enc"), "header vs Ref Enc missing");
    }

    #[test]
    fn test_reference_line() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        assert!(
            table.contains("Reference: none+none"),
            "reference line missing"
        );
    }

    #[test]
    fn test_ratio_calculation() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode_ms: 1.0,
            decode_ms: 1.0,
            compressed_bytes: 3000,
            original_bytes: 8000,
        };
        let ratio = r.ratio_pct();
        // 3000/8000 = 37.5%
        assert!((ratio - 37.5).abs() < 0.001, "ratio {ratio} != 37.5");
    }

    #[test]
    fn test_zero_original_bytes() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode_ms: 1.0,
            decode_ms: 1.0,
            compressed_bytes: 0,
            original_bytes: 0,
        };
        // Must not divide by zero.
        let ratio = r.ratio_pct();
        assert_eq!(ratio, 100.0);
    }

    #[test]
    fn test_zero_reference_time() {
        let results = vec![
            BenchmarkResult {
                name: "ref".to_string(),
                encode_ms: 0.0, // zero reference time
                decode_ms: 0.0,
                compressed_bytes: 100,
                original_bytes: 100,
            },
            BenchmarkResult {
                name: "other".to_string(),
                encode_ms: 5.0,
                decode_ms: 3.0,
                compressed_bytes: 50,
                original_bytes: 100,
            },
        ];
        let table = format_table(&results, "ref", "Test");
        // Should contain N/A for vs-ref columns when reference time is zero.
        assert!(
            table.contains("N/A"),
            "N/A expected when reference time is zero"
        );
    }

    #[test]
    fn test_median_ns_odd() {
        let mut s = [3u64, 1, 4, 1, 5];
        assert_eq!(median_ns(&mut s), 3);
    }

    #[test]
    fn test_median_ns_even() {
        let mut s = [1u64, 2, 3, 4];
        assert_eq!(median_ns(&mut s), 2); // (2+3)/2 = 2
    }

    #[test]
    fn test_median_ns_single() {
        let mut s = [42u64];
        assert_eq!(median_ns(&mut s), 42);
    }

    #[test]
    fn test_median_ns_empty() {
        let mut s: [u64; 0] = [];
        assert_eq!(median_ns(&mut s), 0);
    }

    #[test]
    fn test_median_ns_large_values() {
        // Two u64::MAX values — the u128 addition must not overflow.
        let mut s = [u64::MAX, u64::MAX];
        let m = median_ns(&mut s);
        assert_eq!(m, u64::MAX);
    }

    #[test]
    fn test_median_ns_two_elements() {
        let mut s = [10u64, 20];
        // (10+20)/2 = 15
        assert_eq!(median_ns(&mut s), 15);
    }

    #[test]
    fn test_ns_to_ms() {
        assert!((ns_to_ms(1_000_000) - 1.0).abs() < f64::EPSILON);
        assert!((ns_to_ms(0) - 0.0).abs() < f64::EPSILON);
        assert!((ns_to_ms(500_000) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_size_kib() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode_ms: 1.0,
            decode_ms: 1.0,
            compressed_bytes: 2048,
            original_bytes: 4096,
        };
        assert!((r.size_kib() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ratio_above_100() {
        // Compression expansion: compressed > original (e.g. LZ4 on random data).
        let r = BenchmarkResult {
            name: "x".to_string(),
            encode_ms: 1.0,
            decode_ms: 1.0,
            compressed_bytes: 10000,
            original_bytes: 8000,
        };
        let ratio = r.ratio_pct();
        assert!((ratio - 125.0).abs() < 0.001, "expected 125%, got {ratio}");
    }

    #[test]
    fn test_format_table_empty_results() {
        let table = format_table(&[], "ref", "Empty");
        // Must not panic. Header and separator should still appear.
        assert!(table.contains("Combo"), "header missing for empty table");
        assert!(table.contains("Empty"), "title missing for empty table");
    }

    #[test]
    fn test_format_table_vs_ref_values() {
        let results = make_results();
        let table = format_table(&results, "none+none", "Test");
        // Reference row should show 1.00x for both vs-ref columns.
        assert!(table.contains("1.00x"), "reference row must show 1.00x");
        // sp(24)+szip: encode 10.0 / 0.5 = 20.00x
        assert!(table.contains("20.00x"), "expected 20.00x vs-ref encode");
    }
}

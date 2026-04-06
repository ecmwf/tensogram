// Tensogram benchmark library.
//
// The `run_*` functions are the public entry points called by the binaries.

pub mod codec_matrix;
pub mod datagen;
pub mod report;

/// Error type for benchmark operations.
#[derive(Debug)]
pub struct BenchmarkError(pub String);

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BenchmarkError {}

impl From<String> for BenchmarkError {
    fn from(s: String) -> Self {
        BenchmarkError(s)
    }
}

/// Run the codec matrix benchmark and print results to stdout.
///
/// Called by the `codec-matrix` binary.
pub fn run_codec_matrix(
    num_points: usize,
    iterations: usize,
    seed: u64,
) -> Result<(), BenchmarkError> {
    let results = codec_matrix::run_codec_matrix_results(num_points, iterations, seed)?;

    // Derive actual count from results: original_bytes / 8 bytes per f64.
    // This reflects the rounded-up value (num_points may have been padded to
    // the next multiple of 4 for szip alignment).
    let actual_count = results.first().map_or(num_points, |r| r.original_bytes / 8);
    let title = format!(
        "Tensogram Codec Matrix ({actual_count} float64 values, {iterations} iterations, median)"
    );
    report::print_table(&results, "none+none", &title);
    Ok(())
}

/// Run the GRIB comparison benchmark and print results to stdout.
///
/// Called by the `grib-comparison` binary. Requires the `eccodes` C library.
#[cfg(feature = "eccodes")]
pub fn run_grib_comparison(
    num_points: usize,
    iterations: usize,
    seed: u64,
) -> Result<(), BenchmarkError> {
    let results = grib_comparison::run_grib_comparison_results(num_points, iterations, seed)?;

    let title = format!(
        "GRIB vs Tensogram Comparison ({num_points} float64 values, 24 bit, {iterations} iterations)"
    );
    report::print_table(&results, "eccodes grid_ccsds", &title);
    Ok(())
}

// grib_comparison module declared only when eccodes feature is active.
#[cfg(feature = "eccodes")]
pub mod grib_comparison;

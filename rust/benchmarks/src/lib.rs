// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

pub mod codec_matrix;
pub mod constants;
pub mod datagen;
pub mod report;

#[derive(Debug)]
pub enum BenchmarkError {
    Validation(String),
    Pipeline(String),
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkError::Validation(msg) => write!(f, "validation: {msg}"),
            BenchmarkError::Pipeline(msg) => write!(f, "pipeline: {msg}"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

impl From<tensogram_encodings::pipeline::PipelineError> for BenchmarkError {
    fn from(e: tensogram_encodings::pipeline::PipelineError) -> Self {
        BenchmarkError::Pipeline(e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct CaseFailure {
    pub name: String,
    pub error: String,
}

impl std::fmt::Display for CaseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.error)
    }
}

#[derive(Debug)]
pub struct BenchmarkRun {
    pub results: Vec<report::BenchmarkResult>,
    pub failures: Vec<CaseFailure>,
    pub total_cases: usize,
}

impl BenchmarkRun {
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty()
    }
}

pub fn run_codec_matrix(
    num_points: usize,
    iterations: usize,
    warmup: usize,
    seed: u64,
) -> Result<BenchmarkRun, BenchmarkError> {
    let run = codec_matrix::run_codec_matrix(num_points, iterations, warmup, seed)?;

    let actual_count = run
        .results
        .first()
        .map_or(num_points, |r| r.original_bytes / 8);
    let title = format!(
        "Tensogram Codec Matrix ({actual_count} float64 values, \
         {iterations} iterations, {warmup} warmup, median)"
    );
    report::print_report(&run, "none+none", &title);
    Ok(run)
}

#[cfg(feature = "eccodes")]
pub fn run_grib_comparison(
    num_points: usize,
    iterations: usize,
    warmup: usize,
    seed: u64,
) -> Result<BenchmarkRun, BenchmarkError> {
    let run = grib_comparison::run_grib_comparison(num_points, iterations, warmup, seed)?;

    let title = format!(
        "GRIB vs Tensogram Comparison ({num_points} float64 values, 24 bit, \
         {iterations} iterations, {warmup} warmup)"
    );
    report::print_report(&run, "eccodes grid_ccsds", &title);
    Ok(run)
}

#[cfg(feature = "eccodes")]
pub mod grib_comparison;

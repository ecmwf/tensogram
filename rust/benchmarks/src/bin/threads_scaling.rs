// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Measure encode/decode throughput scaling across `threads` values
//! for a representative codec set.
//!
//! Example:
//!
//!   cargo run --release --bin threads-scaling -- \
//!       --num-points 16000000 --threads 0,1,2,4,8

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "threads-scaling",
    about = "Measure encode/decode scaling vs `threads` for the v0.13.0 pipeline"
)]
struct Args {
    /// Number of float64 values per case.  Rounded up to multiple of 4.
    #[arg(long, default_value = "16000000")]
    num_points: usize,

    /// Number of timed iterations (median reported).
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// Number of warm-up iterations (discarded).
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Random seed for deterministic data generation.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Comma-separated thread counts to sweep (first is the baseline).
    #[arg(long, default_value = "0,1,2,4,8")]
    threads: String,
}

fn main() {
    let args = Args::parse();

    let thread_counts: Result<Vec<u32>, _> = args
        .threads
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect();
    let thread_counts = match thread_counts {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid --threads list '{}': {e}", args.threads);
            std::process::exit(2);
        }
    };

    match tensogram_benchmarks::threads_scaling::run_threads_scaling(
        args.num_points,
        args.iterations,
        args.warmup,
        args.seed,
        &thread_counts,
    ) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

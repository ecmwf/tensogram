use clap::Parser;

/// Compare ecCodes CCSDS packing against Tensogram simple_packing+szip.
///
/// Benchmarks encoding and decoding of 10M float64 values (packed to 24 bit)
/// using ecCodes `grid_ccsds` as the reference and tensogram as the candidate.
/// Requires the `eccodes` feature and the ecCodes C library to be installed.
#[derive(Parser, Debug)]
#[command(
    name = "grib-comparison",
    about = "Compare ecCodes CCSDS packing vs tensogram simple_packing+szip"
)]
struct Args {
    /// Number of float64 values to encode per benchmark run.
    #[arg(long, default_value = "10000000")]
    num_points: usize,

    /// Number of timed iterations (median is reported).
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// Random seed for deterministic data generation.
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    match tensogram_benchmarks::run_grib_comparison(args.num_points, args.iterations, args.seed) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

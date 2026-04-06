use clap::Parser;

/// Benchmark all encoder × compressor × bit-width combinations.
///
/// Runs the full tensogram encoding pipeline for every valid combination,
/// measuring encode/decode throughput and compressed size against the
/// `none+none` baseline.
#[derive(Parser, Debug)]
#[command(
    name = "codec-matrix",
    about = "Benchmark all encoder × compressor × bit-width combinations"
)]
struct Args {
    /// Number of float64 values to encode per benchmark run.
    /// Rounded up to the next multiple of 4 (at most 3 extra values) for szip alignment.
    #[arg(long, default_value = "16000000")]
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
    match tensogram_benchmarks::run_codec_matrix(args.num_points, args.iterations, args.seed) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

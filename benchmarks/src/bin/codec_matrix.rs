use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "codec-matrix",
    about = "Benchmark all encoder \u{d7} compressor \u{d7} bit-width combinations"
)]
struct Args {
    /// Number of float64 values to encode per benchmark run.
    /// Rounded up to the next multiple of 4 for szip alignment.
    #[arg(long, default_value = "16000000")]
    num_points: usize,

    /// Number of timed iterations (median reported).
    #[arg(long, default_value = "10")]
    iterations: usize,

    /// Number of warm-up iterations (discarded).
    #[arg(long, default_value = "3")]
    warmup: usize,

    /// Random seed for deterministic data generation.
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    match tensogram_benchmarks::run_codec_matrix(
        args.num_points,
        args.iterations,
        args.warmup,
        args.seed,
    ) {
        Ok(run) => {
            if !run.all_passed() {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

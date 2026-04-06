use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

mod commands;
mod filter;
mod output;

#[derive(Parser)]
#[command(name = "tensogram", about = "Tensogram message format CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show file-level summary: message count, size, version
    Info { files: Vec<PathBuf> },
    /// List metadata from messages in tabular format
    Ls {
        /// Where-clause filter (e.g., mars.param=2t/10u)
        #[arg(short = 'w')]
        where_clause: Option<String>,
        /// Comma-separated keys to display
        #[arg(short = 'p')]
        keys: Option<String>,
        /// JSON output
        #[arg(short = 'j')]
        json: bool,
        files: Vec<PathBuf>,
    },
    /// Full dump of message contents
    Dump {
        #[arg(short = 'w')]
        where_clause: Option<String>,
        #[arg(short = 'p')]
        keys: Option<String>,
        #[arg(short = 'j')]
        json: bool,
        files: Vec<PathBuf>,
    },
    /// Extract specific key values (strict: errors on missing keys)
    Get {
        #[arg(short = 'w')]
        where_clause: Option<String>,
        /// Comma-separated keys to extract (required)
        #[arg(short = 'p')]
        keys: String,
        files: Vec<PathBuf>,
    },
    /// Modify metadata key/value pairs
    Set {
        /// Key=value pairs to set (comma-separated)
        #[arg(short = 's')]
        set_values: String,
        #[arg(short = 'w')]
        where_clause: Option<String>,
        input: PathBuf,
        output: PathBuf,
    },
    /// Copy/split messages from input to output
    Copy {
        #[arg(short = 'w')]
        where_clause: Option<String>,
        input: PathBuf,
        /// Output path (supports `[keyName]` placeholders for splitting)
        output: String,
    },
    /// Merge messages from multiple files into a single message
    Merge {
        /// Input files
        inputs: Vec<PathBuf>,
        /// Output file
        #[arg(short = 'o', long)]
        output: PathBuf,
    },
    /// Split multi-object messages into separate single-object files
    Split {
        /// Input file
        input: PathBuf,
        /// Output template (use `\[index\]` for numbering, e.g. "split_\[index\].tgm")
        #[arg(short = 'o', long)]
        output: String,
    },
    /// Reshuffle frames: move footer frames to header position
    Reshuffle {
        /// Input file
        input: PathBuf,
        /// Output file
        #[arg(short = 'o', long)]
        output: PathBuf,
    },
    /// Convert GRIB messages to Tensogram format (requires ecCodes)
    #[cfg(feature = "grib")]
    ConvertGrib {
        /// Input GRIB file(s)
        #[arg(required = true)]
        inputs: Vec<String>,
        /// Output file (omit for stdout)
        #[arg(short = 'o', long)]
        output: Option<String>,
        /// Each GRIB message becomes a separate Tensogram message
        #[arg(long)]
        split: bool,
        /// Preserve all GRIB namespace keys under a "grib" sub-object
        #[arg(long)]
        all_keys: bool,
    },
}

fn main() {
    // Activate tracing output via TENSOGRAM_LOG env var.
    // Examples: TENSOGRAM_LOG=debug, TENSOGRAM_LOG=tensogram_core=trace
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("TENSOGRAM_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("off")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Info { files } => commands::info::run(&files),
        Commands::Ls {
            where_clause,
            keys,
            json,
            files,
        } => commands::ls::run(&files, where_clause.as_deref(), keys.as_deref(), json),
        Commands::Dump {
            where_clause,
            keys,
            json,
            files,
        } => commands::dump::run(&files, where_clause.as_deref(), keys.as_deref(), json),
        Commands::Get {
            where_clause,
            keys,
            files,
        } => commands::get::run(&files, where_clause.as_deref(), &keys),
        Commands::Set {
            set_values,
            where_clause,
            input,
            output,
        } => commands::set::run(&input, &output, &set_values, where_clause.as_deref()),
        Commands::Copy {
            where_clause,
            input,
            output,
        } => commands::copy::run(&input, &output, where_clause.as_deref()),
        Commands::Merge { inputs, output } => commands::merge::run(&inputs, &output),
        Commands::Split { input, output } => commands::split::run(&input, &output),
        Commands::Reshuffle { input, output } => commands::reshuffle::run(&input, &output),
        #[cfg(feature = "grib")]
        Commands::ConvertGrib {
            inputs,
            output,
            split,
            all_keys,
        } => commands::convert_grib::run(&inputs, output.as_deref(), split, all_keys),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

#[cfg(any(feature = "grib", feature = "netcdf"))]
use encoding_args::PipelineArgs;

mod commands;
mod encoding_args;
mod filter;
mod output;

#[derive(Parser)]
#[command(name = "tensogram", about = "Tensogram message format CLI", version)]
struct Cli {
    /// Thread budget for the coding pipeline.
    ///
    /// 0 (default) runs sequentially (and may be overridden by the
    /// `TENSOGRAM_THREADS` env var).  N >= 1 spawns a scoped rayon
    /// pool of size N that is spent axis-B-first (intra-codec
    /// parallelism for blosc2/zstd/simple_packing/shuffle), falling
    /// back to axis A (across objects) when no codec stage can use
    /// the budget.  See docs/src/guide/multi-threaded-pipeline.md.
    ///
    /// Read-only metadata commands (info/ls/get/dump) ignore this
    /// flag — they do no decoding work.  Decode-heavy commands
    /// (copy/merge/split/reshuffle/convert-grib/convert-netcdf/validate)
    /// honour it.
    #[arg(long, global = true, default_value_t = 0, env = "TENSOGRAM_THREADS")]
    threads: u32,

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
        /// Merge strategy for conflicting metadata keys:
        /// first (default) — first value wins,
        /// last — last value wins,
        /// error — fail on conflict
        #[arg(short = 's', long, default_value = "first")]
        strategy: String,
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
        #[command(flatten)]
        pipeline: PipelineArgs,
    },
    /// Validate .tgm files for correctness and integrity
    Validate {
        /// Paths to .tgm files
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Quick mode: structure only (level 1)
        #[arg(long, group = "vlevel")]
        quick: bool,
        /// Checksum only: hash verification (level 3)
        #[arg(long, group = "vlevel")]
        checksum: bool,
        /// Full mode: all levels including fidelity (levels 1-4)
        #[arg(long, group = "vlevel")]
        full: bool,
        /// Check RFC 8949 canonical CBOR key ordering (combinable with any level)
        #[arg(long)]
        canonical: bool,
        /// Machine-parseable JSON output
        #[arg(long)]
        json: bool,
    },
    /// Convert NetCDF files to Tensogram format (requires libnetcdf)
    #[cfg(feature = "netcdf")]
    ConvertNetcdf {
        /// Input NetCDF file(s)
        #[arg(required = true)]
        inputs: Vec<String>,
        /// Output file (omit for stdout)
        #[arg(short = 'o', long)]
        output: Option<String>,
        /// How to group variables into messages: file (default), variable, record
        #[arg(long, default_value = "file")]
        split_by: String,
        /// Extract CF convention attributes into base[i]["cf"]
        #[arg(long)]
        cf: bool,
        #[command(flatten)]
        pipeline: PipelineArgs,
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
    let threads = cli.threads;

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
        Commands::Merge {
            inputs,
            output,
            strategy,
        } => commands::merge::run(&inputs, &output, &strategy, threads),
        Commands::Split { input, output } => commands::split::run(&input, &output, threads),
        Commands::Reshuffle { input, output } => commands::reshuffle::run(&input, &output, threads),
        Commands::Validate {
            files,
            quick,
            checksum,
            full,
            canonical,
            json,
        } => {
            let max_level = if quick {
                tensogram_core::ValidationLevel::Structure
            } else if full {
                tensogram_core::ValidationLevel::Fidelity
            } else {
                tensogram_core::ValidationLevel::Integrity
            };
            let options = tensogram_core::ValidateOptions {
                max_level,
                check_canonical: canonical,
                checksum_only: checksum,
            };
            // Note: `threads` is intentionally not forwarded to validate.
            // Levels 1-2 are structure/metadata only (no decoding).
            // Level 3-4 decode payloads via the library's own path; a
            // future enhancement could thread `threads` into
            // `ValidateOptions`, but for now validate runs sequentially.
            commands::validate::run(&files, &options, json)
        }
        #[cfg(feature = "grib")]
        Commands::ConvertGrib {
            inputs,
            output,
            split,
            all_keys,
            pipeline,
        } => commands::convert_grib::run(
            &inputs,
            output.as_deref(),
            split,
            all_keys,
            &pipeline,
            threads,
        ),
        #[cfg(feature = "netcdf")]
        Commands::ConvertNetcdf {
            inputs,
            output,
            split_by,
            cf,
            pipeline,
        } => commands::convert_netcdf::run(
            &inputs,
            output.as_deref(),
            &split_by,
            cf,
            &pipeline,
            threads,
        ),
    };

    if let Err(e) = result {
        // ValidationFailed is a clean exit — the validate command already
        // printed its own FAILED output, so just set exit code 1.
        if e.downcast_ref::<commands::validate::ValidationFailed>()
            .is_some()
        {
            process::exit(1);
        }
        // Other errors: show the full error chain.
        eprintln!("error: {e}");
        let mut source = e.source();
        while let Some(cause) = source {
            eprintln!("  caused by: {cause}");
            source = cause.source();
        }
        process::exit(1);
    }
}

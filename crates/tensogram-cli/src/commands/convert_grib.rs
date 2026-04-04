use std::io::Write;
use std::path::Path;

use tensogram_grib::{convert_grib_file, ConvertOptions, Grouping};

/// Convert GRIB messages to Tensogram format.
pub fn run(
    inputs: &[impl AsRef<Path>],
    output: Option<&str>,
    split: bool,
    all_keys: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err("no input files specified".into());
    }

    let grouping = if split {
        Grouping::OneToOne
    } else {
        Grouping::MergeAll
    };

    let options = ConvertOptions {
        grouping,
        preserve_all_keys: all_keys,
        ..Default::default()
    };

    let mut all_messages = Vec::new();

    for input in inputs {
        let messages = convert_grib_file(input.as_ref(), &options)?;
        all_messages.extend(messages);
    }

    match output {
        Some(out_path) => {
            let mut out = std::fs::File::create(out_path)?;
            for msg in &all_messages {
                out.write_all(msg)?;
            }
            println!(
                "Converted {} message(s) to {}",
                all_messages.len(),
                out_path
            );
        }
        None => {
            let mut stdout = std::io::stdout().lock();
            for msg in &all_messages {
                stdout.write_all(msg)?;
            }
        }
    }

    Ok(())
}

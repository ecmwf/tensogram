use std::io::Write;
use std::path::{Path, PathBuf};

use tensogram_core::{decode, encode, DecodeOptions, EncodeOptions, TensogramFile};

/// Split a multi-object message into separate single-object messages.
///
/// Each data object becomes its own message, inheriting the global metadata.
/// Output files are named using the template with `[index]` placeholder,
/// or sequentially numbered in the output directory.
pub fn run(input: &Path, output_template: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = TensogramFile::open(input)?;
    let count = file.message_count()?;

    let mut total_written = 0;

    for i in 0..count {
        let msg = file.read_message(i)?;
        let (meta, objects) = decode(&msg, &DecodeOptions::default())?;

        if objects.len() <= 1 {
            // Single-object message: write as-is
            let out_name = expand_index(output_template, total_written);
            let mut out = std::fs::File::create(&out_name)?;
            out.write_all(&msg)?;
            total_written += 1;
            continue;
        }

        for (idx, (desc, data)) in objects.iter().enumerate() {
            let out_name = expand_index(output_template, total_written);

            // Extract the payload entry specific to this object so that
            // per-object metadata (e.g. mars keys) is preserved in the split.
            let mut split_meta = meta.clone();
            split_meta.payload = if idx < meta.payload.len() {
                vec![meta.payload[idx].clone()]
            } else {
                vec![]
            };

            let encoded = encode(
                &split_meta,
                &[(desc, data.as_slice())],
                &EncodeOptions::default(),
            )?;

            let mut out = std::fs::File::create(&out_name)?;
            out.write_all(&encoded)?;
            total_written += 1;
        }
    }

    println!("Split into {total_written} message(s)");

    Ok(())
}

fn expand_index(template: &str, index: usize) -> String {
    if template.contains("[index]") {
        template.replace("[index]", &format!("{index:04}"))
    } else {
        // Append index before extension
        let path = PathBuf::from(template);
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "split".to_string());
        let ext = path
            .extension()
            .map(|s| format!(".{}", s.to_string_lossy()))
            .unwrap_or_default();
        let parent = path
            .parent()
            .map(|p| {
                let s = p.to_string_lossy().to_string();
                if s.is_empty() {
                    String::new()
                } else {
                    format!("{s}/")
                }
            })
            .unwrap_or_default();
        format!("{parent}{stem}_{index:04}{ext}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_index_with_placeholder() {
        assert_eq!(expand_index("out/msg_[index].tgm", 3), "out/msg_0003.tgm");
    }

    #[test]
    fn expand_index_without_placeholder() {
        assert_eq!(expand_index("output.tgm", 0), "output_0000.tgm");
        assert_eq!(expand_index("dir/output.tgm", 5), "dir/output_0005.tgm");
    }
}

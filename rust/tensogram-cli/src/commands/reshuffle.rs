// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::io::Write;
use std::path::Path;

use tensogram::{DecodeOptions, EncodeOptions, RESERVED_KEY, TensogramFile, decode, encode};

/// Reshuffle frames inside messages: move footer frames to header position.
///
/// Converts streaming-mode messages (footer-based index/hash) into
/// random-access-mode messages (header-based index/hash).
/// This is a decode → re-encode operation.
pub fn run(
    input: &Path,
    output: &Path,
    threads: u32,
    reject_nan: bool,
    reject_inf: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = TensogramFile::open(input)?;
    let count = file.message_count()?;

    let mut out = std::fs::File::create(output)?;

    for i in 0..count {
        let msg = file.read_message(i)?;
        // Use wire byte order to preserve the original byte layout when
        // re-encoding — native_byte_order=true would byteswap the payload
        // but leave the descriptor's byte_order unchanged, creating a mismatch.
        let wire_opts = DecodeOptions {
            native_byte_order: false,
            threads,
            ..Default::default()
        };
        let (mut meta, objects) = decode(&msg, &wire_opts)?;

        // Clear reserved fields — the encoder will regenerate them.
        meta.reserved.clear();
        for entry in &mut meta.base {
            entry.remove(RESERVED_KEY);
        }

        let refs: Vec<_> = objects.iter().map(|(d, b)| (d, b.as_slice())).collect();

        // Re-encode produces a header-mode message (non-streaming)
        let encoded = encode(
            &meta,
            &refs,
            &EncodeOptions {
                threads,
                reject_nan,
                reject_inf,
                ..Default::default()
            },
        )?;
        out.write_all(&encoded)?;
    }

    println!(
        "Reshuffled {count} message(s) from {} to {}",
        input.display(),
        output.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensogram::{
        ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
        decode,
    };

    fn make_test_file(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("reshuffle_input.tgm");
        let mut f = tensogram::TensogramFile::create(&path).unwrap();
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".into(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".into(),
            filter: "none".into(),
            compression: "none".into(),
            params: Default::default(),
            hash: None,
        };
        let data = vec![0u8; 16];
        let meta = GlobalMetadata {
            version: 2,
            ..Default::default()
        };
        f.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
        f.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
        path
    }

    #[test]
    fn reshuffle_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_test_file(dir.path());
        let output = dir.path().join("reshuffled.tgm");
        run(&input, &output, 0, false, false).unwrap();

        // Verify output is valid and has same content
        let f = tensogram::TensogramFile::open(&output).unwrap();
        assert_eq!(f.message_count().unwrap(), 2);
        let msg = f.read_message(0).unwrap();
        let (meta, objs) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert_eq!(meta.version, 2);
        assert_eq!(objs.len(), 1);
    }
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 18 — Remote-Scan Tracing
//!
//! Subscribes to the `tensogram::remote_scan` tracing target and runs
//! a full-file iteration with the bidirectional walker enabled, so every
//! mode/round/footer-eager event surfaces on stderr.  The subscriber
//! `EnvFilter` is configured in code (no `RUST_LOG` env-var required);
//! point your environment at it if you want a different filter.
//!
//! Run with:
//!   cargo run -p tensogram-rust-examples --features remote --bin 18_remote_scan_trace

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    RemoteScanOptions, TensogramFile,
};
use tracing_subscriber::EnvFilter;

const N_MESSAGES: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("tensogram::remote_scan=debug,tensogram::remote=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();

    let mut file_bytes = Vec::new();
    for i in 0..N_MESSAGES {
        let meta = GlobalMetadata::default();
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: std::collections::BTreeMap::new(),
            masks: None,
        };
        let payload: Vec<u8> = (0..16).map(|b| b as u8 + i as u8).collect();
        let msg =
            tensogram::encode::encode(&meta, &[(&desc, &payload)], &EncodeOptions::default())?;
        file_bytes.extend_from_slice(&msg);
    }
    println!(
        "Encoded {N_MESSAGES} messages, {} bytes total",
        file_bytes.len()
    );

    let tgm = Arc::new(file_bytes);
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let url = format!("http://127.0.0.1:{port}/forecast.tgm");
    let server_data = tgm.clone();
    std::thread::spawn(move || serve_http(listener, &server_data));

    println!("Serving at {url}");
    println!("\n── Forward-only walker (default) ──");
    let storage = std::collections::BTreeMap::new();
    let file = TensogramFile::open_remote(&url, &storage, None)?;
    println!("Opened: {} messages", file.message_count()?);
    for i in 0..N_MESSAGES {
        let _ = file.decode_object(i, 0, &DecodeOptions::default())?;
    }
    drop(file);

    println!("\n── Bidirectional walker (opt-in) ──");
    let bidi_opts = Some(RemoteScanOptions {
        bidirectional: true,
    });
    let file = TensogramFile::open_remote(&url, &storage, bidi_opts)?;
    println!("Opened: {} messages", file.message_count()?);
    for i in 0..N_MESSAGES {
        let _ = file.decode_object(i, 0, &DecodeOptions::default())?;
    }

    println!("\nTraced events emitted to stderr.  Re-run with");
    println!("`RUST_LOG=tensogram::remote_scan=trace` for finer granularity.");
    Ok(())
}

fn serve_http(listener: TcpListener, data: &[u8]) {
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = vec![0u8; 4096];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let request = String::from_utf8_lossy(&buf[..n]);
        let is_head = request.starts_with("HEAD ");
        let range = parse_range(&request, data.len());
        match range {
            Some((start, end)) => {
                let chunk = &data[start..end];
                let header = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                    end - 1,
                    data.len(),
                    chunk.len(),
                );
                let _ = stream.write_all(header.as_bytes());
                if !is_head {
                    let _ = stream.write_all(chunk);
                }
            }
            None => {
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                    data.len(),
                );
                let _ = stream.write_all(header.as_bytes());
                if !is_head {
                    let _ = stream.write_all(data);
                }
            }
        }
    }
}

fn parse_range(request: &str, file_len: usize) -> Option<(usize, usize)> {
    let line = request
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))?;
    let spec = line.split("bytes=").nth(1)?.trim();
    if let Some(suffix) = spec.strip_prefix('-') {
        let n: usize = suffix.parse().ok()?;
        Some((file_len.saturating_sub(n), file_len))
    } else {
        let mut parts = spec.split('-');
        let start: usize = parts.next()?.parse().ok()?;
        let end_str = parts.next()?;
        let end = if end_str.is_empty() {
            file_len
        } else {
            end_str.parse::<usize>().ok()? + 1
        };
        Some((start, end.min(file_len)))
    }
}

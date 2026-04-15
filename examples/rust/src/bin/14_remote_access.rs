// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 14 — Remote Access
//!
//! Opens a tensogram file over HTTP using the remote backend.
//! A minimal HTTP server with Range-request support runs in a
//! background thread so the example is self-contained.
//!
//! Run with:  cargo run -p tensogram-rust-examples --features remote --bin 14_remote_access

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use tensogram_core::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    TensogramFile,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let meta = GlobalMetadata {
        version: 2,
        base: Vec::new(),
        ..Default::default()
    };
    let desc = DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![72, 144],
        strides: vec![144, 1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: std::collections::BTreeMap::new(),
        hash: None,
    };
    let data = vec![0u8; 72 * 144 * 4];
    let tgm_bytes =
        tensogram_core::encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    println!("Encoded {} bytes", tgm_bytes.len());

    let tgm = Arc::new(tgm_bytes);
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let url = format!("http://127.0.0.1:{port}/forecast.tgm");

    let server_data = tgm.clone();
    std::thread::spawn(move || serve_http(listener, &server_data));

    println!("Serving at {url}");

    assert!(tensogram_core::remote::is_remote_url(&url));
    println!("is_remote_url = true");

    let file = TensogramFile::open_source(&url)?;
    println!(
        "\nOpened: source={} is_remote={} messages={}",
        file.source(),
        file.is_remote(),
        file.message_count()?,
    );

    let meta = file.decode_metadata(0)?;
    println!("Metadata: version={}", meta.version);

    let (_, descriptors) = file.decode_descriptors(0)?;
    println!("Descriptors: {} objects", descriptors.len());
    for (i, d) in descriptors.iter().enumerate() {
        println!("  [{i}] shape={:?} dtype={:?}", d.shape, d.dtype);
    }

    let (_, desc, decoded) = file.decode_object(0, 0, &DecodeOptions::default())?;
    println!(
        "\nObject 0: shape={:?} dtype={:?} bytes={}",
        desc.shape,
        desc.dtype,
        decoded.len()
    );
    assert_eq!(decoded, data);
    println!("  matches original data");

    println!("\nRemote access example complete.");
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
                    "HTTP/1.1 206 Partial Content\r\n\
                     Content-Range: bytes {start}-{}/{}\r\n\
                     Content-Length: {}\r\n\
                     Accept-Ranges: bytes\r\n\
                     Connection: close\r\n\r\n",
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
                    "HTTP/1.1 200 OK\r\n\
                     Content-Length: {}\r\n\
                     Accept-Ranges: bytes\r\n\
                     Connection: close\r\n\r\n",
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
    let range_line = request
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))?;
    let spec = range_line.split("bytes=").nth(1)?.trim();
    if let Some(suffix) = spec.strip_prefix('-') {
        let n: usize = suffix.trim().parse().ok()?;
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

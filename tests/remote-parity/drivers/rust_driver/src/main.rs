// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Remote-parity harness driver (Rust).
//!
//! Usage: `remote-parity-rust-driver --url <URL> --op <OP> [--bidirectional]`
//!
//! Operations:
//!
//! - `open` — open the remote file and exit.
//! - `message-count` — full forward scan.
//! - `read-first` — read message 0.
//! - `read-last` — read the last message (index = count - 1).
//! - `read-metadata` — read message N-1's global metadata.  Forces
//!   layout-populate on the last message; on bidirectional + footer-
//!   indexed files this exercises the eager-footer fast path.
//! - `dump-layout` — full scan + emit `[{"offset": .., "length": ..}, ...]`
//!   JSON to stdout.  The orchestrator parses this to compare scan
//!   results across walker modes.
//!
//! The `--bidirectional` flag opens the file with
//! `RemoteScanOptions { bidirectional: true }`; without it the
//! forward-only walker is used (the default).
//!
//! The driver does not emit logs itself.  The mock server captures
//! every HTTP request, tagged by the `run_id` embedded in the URL
//! path; the orchestrator collects that captured request log from
//! the in-process server after the driver exits.

use std::collections::BTreeMap;
use std::process::ExitCode;

use tensogram::{RemoteScanOptions, TensogramFile};

#[derive(Debug)]
struct Args {
    url: String,
    op: Op,
    bidirectional: bool,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Open,
    MessageCount,
    ReadFirst,
    ReadLast,
    ReadMetadata,
    DumpLayout,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "open" => Ok(Op::Open),
            "message-count" => Ok(Op::MessageCount),
            "read-first" => Ok(Op::ReadFirst),
            "read-last" => Ok(Op::ReadLast),
            "read-metadata" => Ok(Op::ReadMetadata),
            "dump-layout" => Ok(Op::DumpLayout),
            other => Err(format!("unknown --op '{other}'")),
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut url: Option<String> = None;
    let mut op: Option<Op> = None;
    let mut bidirectional = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--url" => {
                url = Some(it.next().ok_or("--url requires a value")?);
            }
            "--op" => {
                op = Some(Op::parse(&it.next().ok_or("--op requires a value")?)?);
            }
            "--bidirectional" => {
                bidirectional = true;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Args {
        url: url.ok_or("missing --url")?,
        op: op.ok_or("missing --op")?,
        bidirectional,
    })
}

fn print_usage() {
    eprintln!(
        "usage: remote-parity-rust-driver --url <URL> --op <open|message-count|read-first|read-last|read-metadata|dump-layout> [--bidirectional]"
    );
}

fn run(args: Args) -> Result<(), String> {
    let storage: BTreeMap<String, String> = BTreeMap::new();
    let scan_opts = Some(RemoteScanOptions {
        bidirectional: args.bidirectional,
    });
    let file = TensogramFile::open_remote(&args.url, &storage, scan_opts)
        .map_err(|e| format!("open_remote failed: {e}"))?;

    match args.op {
        Op::Open => Ok(()),
        Op::MessageCount => {
            let n = file
                .message_count()
                .map_err(|e| format!("message_count failed: {e}"))?;
            println!("{n}");
            Ok(())
        }
        Op::ReadFirst => {
            let _ = file
                .read_message(0)
                .map_err(|e| format!("read_message(0) failed: {e}"))?;
            Ok(())
        }
        Op::ReadLast => {
            let n = file
                .message_count()
                .map_err(|e| format!("message_count failed: {e}"))?;
            if n == 0 {
                return Err("file contains 0 messages".to_string());
            }
            let _ = file
                .read_message(n - 1)
                .map_err(|e| format!("read_message({}) failed: {e}", n - 1))?;
            Ok(())
        }
        Op::ReadMetadata => {
            let n = file
                .message_count()
                .map_err(|e| format!("message_count failed: {e}"))?;
            if n == 0 {
                return Err("file contains 0 messages".to_string());
            }
            let _ = file
                .decode_metadata(n - 1)
                .map_err(|e| format!("decode_metadata({}) failed: {e}", n - 1))?;
            Ok(())
        }
        Op::DumpLayout => {
            let layouts = file
                .message_layouts()
                .map_err(|e| format!("message_layouts failed: {e}"))?;
            let mut out = String::from("[");
            for (i, l) in layouts.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"offset\":{},\"length\":{}}}",
                    l.offset, l.length
                ));
            }
            out.push(']');
            println!("{out}");
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("error: {msg}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

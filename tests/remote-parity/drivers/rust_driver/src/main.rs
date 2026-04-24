// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Remote-parity harness driver (Rust).
//!
//! Usage: `remote-parity-rust-driver --url <URL> --op <OP>`
//!
//! Operations:
//!
//! - `open` — open the remote file and exit.
//! - `message-count` — full forward scan.
//! - `read-first` — read message 0.
//! - `read-last` — read the last message (index = count - 1).
//!
//! The driver does not emit logs itself. The mock server captures every
//! HTTP request on its side, tagged by the `run_id` embedded in the URL
//! path. The orchestrator fetches those logs via `GET /_log/<run_id>`
//! after the driver exits.

use std::collections::BTreeMap;
use std::process::ExitCode;

use tensogram::TensogramFile;

#[derive(Debug)]
struct Args {
    url: String,
    op: Op,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Open,
    MessageCount,
    ReadFirst,
    ReadLast,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "open" => Ok(Op::Open),
            "message-count" => Ok(Op::MessageCount),
            "read-first" => Ok(Op::ReadFirst),
            "read-last" => Ok(Op::ReadLast),
            other => Err(format!("unknown --op '{other}'")),
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut url: Option<String> = None;
    let mut op: Option<Op> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--url" => {
                url = Some(it.next().ok_or("--url requires a value")?);
            }
            "--op" => {
                op = Some(Op::parse(&it.next().ok_or("--op requires a value")?)?);
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
    })
}

fn print_usage() {
    eprintln!(
        "usage: remote-parity-rust-driver --url <URL> --op <open|message-count|read-first|read-last>"
    );
}

fn run(args: Args) -> Result<(), String> {
    let storage: BTreeMap<String, String> = BTreeMap::new();
    let file = TensogramFile::open_remote(&args.url, &storage)
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

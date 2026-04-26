// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Deterministic single-pass remote-scan metrics emitter (NDJSON sidecar).
//!
//! Runs every (fixture × tier × scenario × walker) cell exactly once,
//! captures `total_requests`, `range_get_requests`, `head_requests`,
//! and `response_body_bytes` from the in-process counting mock server,
//! and emits one NDJSON record per cell to
//! `target/remote-scan-bench/rust.ndjson` (or the path given via
//! `--out`).  No warmup, no repetition, no Criterion: the wall-clock
//! sample is deliberately a single timing of one open + operation
//! cycle, since cross-language wall-clock comparison is the point of
//! the sidecar contract.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use tensogram_benchmarks::mock_http::MockServer;
use tensogram_benchmarks::remote_scan_bench::{FixtureSpec, Scenario, load_fixtures, run_cell};

#[derive(Debug, Serialize)]
struct CellRecord {
    language: &'static str,
    mode: &'static str,
    fixture_kind: &'static str,
    fixture_name: String,
    tier: usize,
    scenario: &'static str,
    walker: &'static str,
    total_requests: usize,
    range_get_requests: usize,
    head_requests: usize,
    response_body_bytes: usize,
    wall_ms: f64,
    semantics: &'static str,
}

fn parse_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut out: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                    eprintln!("--out requires a path argument");
                    std::process::exit(2);
                })));
            }
            "--help" | "-h" => {
                println!("usage: remote-scan-metrics [--out <path>]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    out.unwrap_or_else(default_out_path)
}

fn default_out_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("target/remote-scan-bench/rust.ndjson"))
        .unwrap_or_else(|| PathBuf::from("rust.ndjson"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = parse_args();
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out_file = fs::File::create(&out_path)?;

    let fixtures = load_fixtures()?;
    let server = MockServer::start(
        fixtures
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect(),
    )?;

    let walkers = [("forward-only", false), ("bidirectional", true)];
    let mut total_cells = 0usize;
    for spec in FixtureSpec::matrix() {
        for scenario in Scenario::ALL {
            for (walker_label, walker_flag) in walkers {
                let started = Instant::now();
                let counters = run_cell(
                    &server,
                    &spec.fixture_name,
                    spec.tier,
                    scenario,
                    walker_flag,
                )?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

                let record = CellRecord {
                    language: "rust",
                    mode: "sync",
                    fixture_kind: spec.kind.as_str(),
                    fixture_name: spec.fixture_name.clone(),
                    tier: spec.tier,
                    scenario: scenario.as_str(),
                    walker: walker_label,
                    total_requests: counters.total_requests(),
                    range_get_requests: counters.range_get_requests(),
                    head_requests: counters.head_requests(),
                    response_body_bytes: counters.response_body_bytes(),
                    wall_ms: elapsed_ms,
                    semantics: "cold_open_plus_operation_plus_close",
                };
                serde_json::to_writer(&mut out_file, &record)?;
                writeln!(out_file)?;
                total_cells += 1;
            }
        }
    }

    out_file.sync_all()?;
    eprintln!("wrote {total_cells} cells to {}", out_path.display());
    Ok(())
}

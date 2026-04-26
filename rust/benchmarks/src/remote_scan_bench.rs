// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Shared scenario runner for the remote-scan walker bench.
//!
//! Defines the (fixture × tier × scenario × walker) matrix and the
//! single `run_cell` entry point used by both the metrics binary
//! (`bin/remote_scan_metrics.rs`) and the Criterion harness
//! (`benches/remote_scan.rs`).  Each cell measures a fresh
//! `open_remote → operation → close` cycle so the metric semantics
//! match the Python and TypeScript harnesses byte-for-byte.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tensogram::{RemoteScanOptions, TensogramFile};

use crate::mock_http::{Counters, MockServer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureKind {
    HeaderIndexed,
    FooterIndexed,
    StreamingTail,
}

impl FixtureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FixtureKind::HeaderIndexed => "header-indexed",
            FixtureKind::FooterIndexed => "footer-indexed",
            FixtureKind::StreamingTail => "streaming-tail",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    MessageCount,
    ReadFirst,
    ReadLast,
    ReadMiddle,
    Iter,
}

impl Scenario {
    pub const ALL: [Scenario; 5] = [
        Scenario::MessageCount,
        Scenario::ReadFirst,
        Scenario::ReadLast,
        Scenario::ReadMiddle,
        Scenario::Iter,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Scenario::MessageCount => "message_count",
            Scenario::ReadFirst => "read_message(0)",
            Scenario::ReadLast => "read_message(N-1)",
            Scenario::ReadMiddle => "read_message(N/2)",
            Scenario::Iter => "iter",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FixtureSpec {
    pub kind: FixtureKind,
    pub tier: usize,
    pub fixture_name: String,
}

impl FixtureSpec {
    pub fn matrix() -> Vec<FixtureSpec> {
        let mut specs = Vec::new();
        for tier in [1usize, 10, 100, 1000] {
            specs.push(FixtureSpec {
                kind: FixtureKind::HeaderIndexed,
                tier,
                fixture_name: header_indexed_fixture(tier).to_string(),
            });
            specs.push(FixtureSpec {
                kind: FixtureKind::FooterIndexed,
                tier,
                fixture_name: footer_indexed_fixture(tier).to_string(),
            });
        }
        specs.push(FixtureSpec {
            kind: FixtureKind::StreamingTail,
            tier: 10,
            fixture_name: "streaming-tail.tgm".to_string(),
        });
        specs
    }
}

fn header_indexed_fixture(tier: usize) -> &'static str {
    match tier {
        1 => "single-msg.tgm",
        10 => "ten-msg.tgm",
        100 => "hundred-msg.tgm",
        1000 => "thousand-msg.tgm",
        _ => panic!("unsupported header-indexed tier {tier}"),
    }
}

fn footer_indexed_fixture(tier: usize) -> &'static str {
    match tier {
        1 => "single-msg-footer.tgm",
        10 => "ten-msg-footer.tgm",
        100 => "hundred-msg-footer.tgm",
        1000 => "thousand-msg-footer.tgm",
        _ => panic!("unsupported footer-indexed tier {tier}"),
    }
}

pub fn fixtures_dir() -> PathBuf {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("tests/remote-parity/fixtures"))
        .expect("fixtures dir under repo root")
}

pub fn load_fixtures() -> std::io::Result<BTreeMap<String, Arc<Vec<u8>>>> {
    let dir = fixtures_dir();
    let mut out = BTreeMap::new();
    for spec in FixtureSpec::matrix() {
        let path = dir.join(&spec.fixture_name);
        let data = std::fs::read(&path)?;
        out.insert(spec.fixture_name.clone(), Arc::new(data));
    }
    Ok(out)
}

pub fn run_cell(
    server: &MockServer,
    fixture_name: &str,
    n: usize,
    scenario: Scenario,
    bidirectional: bool,
) -> Result<Counters, String> {
    server.reset();
    let url = server.url_for(fixture_name);
    let scan_opts = Some(RemoteScanOptions { bidirectional });
    let storage: BTreeMap<String, String> = BTreeMap::new();

    let file = TensogramFile::open_remote(&url, &storage, scan_opts)
        .map_err(|e| format!("open_remote {url}: {e}"))?;

    let last_idx = if n == 0 { 0 } else { n - 1 };
    match scenario {
        Scenario::MessageCount => {
            let _ = file
                .message_count()
                .map_err(|e| format!("message_count: {e}"))?;
        }
        Scenario::ReadFirst => {
            let _ = file
                .read_message(0)
                .map_err(|e| format!("read_message(0): {e}"))?;
        }
        Scenario::ReadLast => {
            let _ = file
                .read_message(last_idx)
                .map_err(|e| format!("read_message({last_idx}): {e}"))?;
        }
        Scenario::ReadMiddle => {
            let _ = file
                .read_message(n / 2)
                .map_err(|e| format!("read_message({}): {e}", n / 2))?;
        }
        Scenario::Iter => {
            let count = file
                .message_count()
                .map_err(|e| format!("message_count: {e}"))?;
            for i in 0..count {
                let _ = file
                    .read_message(i)
                    .map_err(|e| format!("read_message({i}): {e}"))?;
            }
        }
    }

    drop(file);
    Ok(server.snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_matrix_covers_all_kinds_and_tiers() {
        let specs = FixtureSpec::matrix();
        assert_eq!(specs.len(), 4 + 4 + 1);
        let header_count = specs
            .iter()
            .filter(|s| s.kind == FixtureKind::HeaderIndexed)
            .count();
        let footer_count = specs
            .iter()
            .filter(|s| s.kind == FixtureKind::FooterIndexed)
            .count();
        let streaming_count = specs
            .iter()
            .filter(|s| s.kind == FixtureKind::StreamingTail)
            .count();
        assert_eq!(header_count, 4);
        assert_eq!(footer_count, 4);
        assert_eq!(streaming_count, 1);
    }

    #[test]
    fn fixtures_dir_resolves_under_repo() {
        let dir = fixtures_dir();
        assert!(
            dir.join("single-msg.tgm").exists(),
            "fixtures dir missing single-msg.tgm: {dir:?}"
        );
    }

    #[test]
    fn run_cell_produces_request_counts() {
        let fixtures = load_fixtures().expect("load fixtures");
        let server = MockServer::start(fixtures).expect("start server");
        let counters =
            run_cell(&server, "ten-msg.tgm", 10, Scenario::Iter, false).expect("run_cell");
        assert!(counters.total_requests() > 0);
        assert!(counters.range_get_requests() > 0);
    }
}

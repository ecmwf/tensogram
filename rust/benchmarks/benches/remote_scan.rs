// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Criterion wall-clock for the remote-scan walker headline cells.
//!
//! Wall-clock only.  HTTP request and byte counts are collected by the
//! sibling `remote-scan-metrics` binary (deterministic single-pass);
//! Criterion's variable iteration count would smear those counters
//! across many runs.  The cells benched here are the headline ones the
//! decision artifact references — N=100 iter, N=1000 read-last,
//! N=10 streaming-tail — both walkers each.

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};

use tensogram_benchmarks::mock_http::MockServer;
use tensogram_benchmarks::remote_scan_bench::{Scenario, load_fixtures, run_cell};

fn bench_remote_scan(c: &mut Criterion) {
    let fixtures = match load_fixtures() {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "remote_scan bench: cannot load fixtures ({e}); regenerate with \
                 `python tests/remote-parity/tools/gen_fixtures.py`."
            );
            return;
        }
    };
    let server = MockServer::start(
        fixtures
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect(),
    )
    .expect("MockServer start");

    let mut group = c.benchmark_group("remote_scan_wallclock");
    group.sample_size(20);

    let cells: &[(&str, &str, usize, Scenario)] = &[
        ("hundred-msg.tgm", "header-indexed", 100, Scenario::Iter),
        (
            "hundred-msg-footer.tgm",
            "footer-indexed",
            100,
            Scenario::Iter,
        ),
        (
            "thousand-msg.tgm",
            "header-indexed",
            1000,
            Scenario::ReadLast,
        ),
        ("streaming-tail.tgm", "streaming-tail", 10, Scenario::Iter),
    ];

    for (fixture, kind, n, scenario) in cells {
        for (walker_label, bidirectional) in [("forward", false), ("bidir", true)] {
            let id = format!("{kind}/N={n}/{}/{walker_label}", scenario.as_str());
            group.bench_function(&id, |b| {
                b.iter(|| {
                    run_cell(&server, fixture, *n, *scenario, bidirectional).expect("run_cell");
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_remote_scan);
criterion_main!(benches);

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for `tensogram doctor`.

use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tensogram"))
}

/// Run `tensogram doctor` with the given args and assert it exits successfully.
/// Returns captured stdout as a `String`.
fn run_doctor(args: &[&str]) -> String {
    let output = cli()
        .args(args)
        .output()
        .expect("failed to spawn tensogram");
    assert!(
        output.status.success(),
        "tensogram {args:?} exited non-zero: {:?}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout).expect("doctor stdout is not valid UTF-8")
}

#[test]
fn doctor_exits_zero_and_prints_healthy() {
    let stdout = run_doctor(&["doctor"]);
    assert!(
        stdout.contains("Status: HEALTHY"),
        "expected 'Status: HEALTHY' in output:\n{stdout}"
    );
}

#[test]
fn doctor_human_output_contains_section_headers() {
    let stdout = run_doctor(&["doctor"]);
    for header in [
        "tensogram doctor",
        "Build",
        "Compiled-in features",
        "Self-test",
    ] {
        assert!(
            stdout.contains(header),
            "human output missing section header '{header}':\n{stdout}"
        );
    }
}

#[test]
fn doctor_human_output_renders_linkage_strings() {
    // The default build has at least one FFI codec (zstd is in default
    // features) and at least one pure-Rust crate (lz4_flex).  We assert
    // both linkage strings appear so that any future Linkage rename
    // surfaces here rather than silently regressing user-facing output.
    let stdout = run_doctor(&["doctor"]);
    assert!(
        stdout.contains("FFI"),
        "expected 'FFI' linkage string:\n{stdout}"
    );
    assert!(
        stdout.contains("pure-Rust"),
        "expected 'pure-Rust' linkage string:\n{stdout}"
    );
}

#[test]
fn doctor_human_output_shows_status_line_last() {
    // Sanity check: `Status: HEALTHY` appears on the final non-empty line —
    // means failures can't be hidden by accidentally appending more output.
    let stdout = run_doctor(&["doctor"]);
    let last_non_empty = stdout
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .expect("doctor output should have at least one line");
    assert!(
        last_non_empty.starts_with("Status: "),
        "last line should be Status, got: {last_non_empty:?}"
    );
}

#[test]
fn doctor_json_flag_produces_valid_json_with_expected_keys() {
    let stdout = run_doctor(&["doctor", "--json"]);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("output is not valid JSON");

    let obj = parsed.as_object().expect("top-level JSON is not an object");
    for key in ["build", "features", "self_test"] {
        assert!(obj.contains_key(key), "JSON missing top-level key '{key}'");
    }

    let build = obj["build"].as_object().expect("'build' is not an object");
    for key in ["version", "wire_version", "target", "profile"] {
        assert!(build.contains_key(key), "build JSON missing key '{key}'");
    }

    let features = obj["features"]
        .as_array()
        .expect("'features' is not an array");
    assert!(!features.is_empty(), "'features' array is empty");

    let self_test = obj["self_test"]
        .as_array()
        .expect("'self_test' is not an array");
    assert!(!self_test.is_empty(), "'self_test' array is empty");
}

#[test]
fn doctor_json_features_use_flat_shape() {
    // Locks in the JSON shape promised in `docs/src/cli/doctor.md`:
    // each `On` feature has flat top-level keys (no nested `state.state`).
    let stdout = run_doctor(&["doctor", "--json"]);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("output is not valid JSON");
    let features = parsed["features"]
        .as_array()
        .expect("'features' is not an array");
    let on_row = features
        .iter()
        .find(|f| f.get("state").and_then(|s| s.as_str()) == Some("on"))
        .expect("at least one feature should be On in default build");
    let obj = on_row.as_object().unwrap();
    for key in ["name", "kind", "state", "backend", "linkage"] {
        assert!(obj.contains_key(key), "On feature missing flat key '{key}'");
    }
    // `state` must be a string, not nested.
    assert!(
        obj["state"].is_string(),
        "expected `state` as string, got: {:?}",
        obj["state"]
    );
}

#[test]
fn doctor_json_self_test_uses_flat_shape() {
    let stdout = run_doctor(&["doctor", "--json"]);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("output is not valid JSON");
    let self_test = parsed["self_test"]
        .as_array()
        .expect("'self_test' is not an array");
    for row in self_test {
        let obj = row.as_object().expect("row is a JSON object");
        assert!(obj.contains_key("label"), "row missing 'label': {obj:?}");
        // `outcome` is the discriminator string, not a nested object.
        let outcome = obj["outcome"]
            .as_str()
            .unwrap_or_else(|| panic!("row 'outcome' must be a string: {obj:?}"));
        assert!(
            ["ok", "failed", "skipped"].contains(&outcome),
            "unexpected outcome value: {outcome:?}"
        );
    }
}

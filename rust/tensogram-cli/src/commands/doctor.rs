// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! `tensogram doctor` — environment diagnostics subcommand.
//!
//! Collects build metadata, compiled-in feature states, and runs a
//! self-test of the encode/decode pipeline plus the GRIB and NetCDF
//! converters (when those features are compiled in).

use tensogram::doctor::{DoctorReport, FeatureState, SelfTestOutcome, SelfTestResult};

// ── Layout constants ─────────────────────────────────────────────────────────

/// Column width for the leftmost label in the `Build` and `Compiled-in
/// features` sections.
const FEATURE_NAME_WIDTH: usize = 22;

/// Column width for the leftmost label in the `Self-test` section.
const SELF_TEST_LABEL_WIDTH: usize = 42;

// ── Self-test row labels ─────────────────────────────────────────────────────

const GRIB_LABEL: &str = "convert        grib    (sanity.grib2)";
const NETCDF_CLASSIC_LABEL: &str = "convert        netcdf3 (sanity-classic.nc)";
const NETCDF_HDF5_LABEL: &str = "convert        netcdf4 (sanity-hdf5.nc)";

// ── Sanity fixtures (embedded at compile time) ────────────────────────────────

#[cfg(feature = "grib")]
const GRIB_FIXTURE: &[u8] = include_bytes!("../../../../share/tensogram/doctor/sanity.grib2");

#[cfg(feature = "netcdf")]
const NETCDF_CLASSIC_FIXTURE: &[u8] =
    include_bytes!("../../../../share/tensogram/doctor/sanity-classic.nc");

#[cfg(feature = "netcdf")]
const NETCDF_HDF5_FIXTURE: &[u8] =
    include_bytes!("../../../../share/tensogram/doctor/sanity-hdf5.nc");

// ── DoctorFailed sentinel ─────────────────────────────────────────────────────

/// Sentinel error returned when one or more self-tests fail.
///
/// `main.rs` catches this type and exits with code 1 without printing
/// a redundant error message (the human/JSON output already shows the
/// failures).
#[derive(Debug)]
pub struct DoctorFailed;

impl std::fmt::Display for DoctorFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "doctor self-test failed")
    }
}

impl std::error::Error for DoctorFailed {}

// ── TempFileGuard ─────────────────────────────────────────────────────────────

/// RAII guard that writes bytes to a temporary file and deletes it on drop.
#[cfg(any(feature = "grib", feature = "netcdf", test))]
struct TempFileGuard {
    path: std::path::PathBuf,
}

#[cfg(any(feature = "grib", feature = "netcdf", test))]
impl TempFileGuard {
    /// Write `bytes` to a new temp file with the given `suffix` and return
    /// the guard.  The file is deleted when the guard is dropped.
    fn new(bytes: &[u8], suffix: &str) -> std::io::Result<Self> {
        let pid = std::process::id();
        let name = format!("tensogram-doctor-{pid}-{suffix}");
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, bytes)?;
        Ok(Self { path })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[cfg(any(feature = "grib", feature = "netcdf", test))]
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        // Ignore NotFound — already deleted or never created.
        let _ = std::fs::remove_file(&self.path);
    }
}

// ── Converter backend versions ────────────────────────────────────────────────

/// Decode eccodes' packed-integer version into a `"major.minor.patch"`
/// string.  Returns `None` for non-positive inputs (libeccodes documents
/// the value as non-negative; defensive against a malformed return).
///
/// Extracted from [`grib_backend_version`] so the decoding logic is unit
/// testable without the FFI call.
#[cfg(feature = "grib")]
fn decode_eccodes_version(packed: std::ffi::c_long) -> Option<String> {
    if packed <= 0 {
        return None;
    }
    let major = packed / 10_000;
    let minor = (packed % 10_000) / 100;
    let patch = packed % 100;
    Some(format!("{major}.{minor}.{patch}"))
}

/// Probe libeccodes for its runtime API version, decoded into a
/// `"major.minor.patch"` string.  Returns `None` for the version field
/// of the [`BackendVersion`] when libeccodes returns 0 or a negative
/// (i.e. unrecognised) packed value.
#[cfg(feature = "grib")]
fn grib_backend_version() -> tensogram::doctor::BackendVersion {
    unsafe extern "C" {
        /// Returns the eccodes API version as a packed integer:
        /// `(major * 10_000) + (minor * 100) + patch`.  Documented in
        /// `<eccodes.h>` to return a non-negative value.
        fn codes_get_api_version() -> std::ffi::c_long;
    }

    // SAFETY: `codes_get_api_version()` takes no arguments, performs no
    // I/O, and returns a plain `long` from a thread-safe getter.
    let v = unsafe { codes_get_api_version() };
    tensogram::doctor::BackendVersion::ffi("libeccodes", decode_eccodes_version(v))
}

/// Probe libnetcdf for its runtime version string.
///
/// libnetcdf's `nc_inq_libvers()` returns a pointer to a string of the
/// form `"4.10.0 of Apr  3 2024 ..."` — we keep only the first
/// whitespace-separated token so the JSON output stays compact.
#[cfg(feature = "netcdf")]
fn netcdf_backend_version() -> tensogram::doctor::BackendVersion {
    unsafe extern "C" {
        /// Returns a pointer to libnetcdf's static version string, e.g.
        /// `"4.10.0 of Apr  3 2024 ..."`.  Documented to return a
        /// non-null pointer to immutable string storage.
        fn nc_inq_libvers() -> *const std::ffi::c_char;
    }

    // SAFETY: `nc_inq_libvers()` returns a pointer to static, immutable
    // string storage owned by libnetcdf; reading it via CStr is sound.
    let raw = unsafe { tensogram::doctor::cstr_ptr_to_owned(nc_inq_libvers()) };

    let version = raw
        .as_deref()
        .and_then(|s| s.split_whitespace().next())
        .map(str::to_owned);
    tensogram::doctor::BackendVersion::ffi("libnetcdf", version)
}

// ── Converter self-tests ──────────────────────────────────────────────────────

#[cfg(feature = "grib")]
fn self_test_grib() -> SelfTestResult {
    use tensogram::decode::DecodeOptions;
    use tensogram_grib::{ConvertOptions, Grouping, convert_grib_file};

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let guard = TempFileGuard::new(GRIB_FIXTURE, "sanity.grib2")?;
        let opts = ConvertOptions {
            grouping: Grouping::OneToOne,
            ..Default::default()
        };
        let messages = convert_grib_file(guard.path(), &opts)?;
        if messages.len() != 1 {
            return Err(format!("expected 1 message, got {}", messages.len()).into());
        }
        let (_meta, objects) = tensogram::decode::decode(&messages[0], &DecodeOptions::default())?;
        if objects.len() != 1 {
            return Err(format!("expected 1 object, got {}", objects.len()).into());
        }
        let (desc, _payload) = &objects[0];
        if desc.shape != vec![4, 4] {
            return Err(format!("expected shape [4,4], got {:?}", desc.shape).into());
        }
        if desc.dtype != tensogram::Dtype::Float64 {
            return Err(format!("expected dtype f64, got {:?}", desc.dtype).into());
        }
        Ok(())
    })();
    match result {
        Ok(()) => SelfTestResult::ok(GRIB_LABEL),
        Err(e) => SelfTestResult::failed(GRIB_LABEL, e.to_string()),
    }
}

#[cfg(feature = "netcdf")]
fn self_test_netcdf_classic() -> SelfTestResult {
    self_test_netcdf_fixture(
        NETCDF_CLASSIC_FIXTURE,
        "sanity-classic.nc",
        NETCDF_CLASSIC_LABEL,
    )
}

#[cfg(feature = "netcdf")]
fn self_test_netcdf_hdf5() -> SelfTestResult {
    self_test_netcdf_fixture(NETCDF_HDF5_FIXTURE, "sanity-hdf5.nc", NETCDF_HDF5_LABEL)
}

#[cfg(feature = "netcdf")]
fn self_test_netcdf_fixture(bytes: &[u8], suffix: &str, label: &'static str) -> SelfTestResult {
    use tensogram::decode::DecodeOptions;
    use tensogram_netcdf::{ConvertOptions, convert_netcdf_file};

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let guard = TempFileGuard::new(bytes, suffix)?;
        let opts = ConvertOptions::default();
        let messages = convert_netcdf_file(guard.path(), &opts)?;
        if messages.is_empty() {
            return Err("no messages produced".into());
        }
        // Find the temperature variable object (shape [2,2], dtype f32).
        let mut found = false;
        for msg_bytes in &messages {
            let (_meta, objects) = tensogram::decode::decode(msg_bytes, &DecodeOptions::default())?;
            for (desc, _payload) in &objects {
                if desc.shape == vec![2, 2] && desc.dtype == tensogram::Dtype::Float32 {
                    found = true;
                }
            }
        }
        if !found {
            return Err("temperature variable with shape [2,2] f32 not found".into());
        }
        Ok(())
    })();
    match result {
        Ok(()) => SelfTestResult::ok(label),
        Err(e) => SelfTestResult::failed(label, e.to_string()),
    }
}

// ── Human renderer ────────────────────────────────────────────────────────────

/// Render a [`DoctorReport`] (plus the CLI-only converter rows) to stdout
/// in the human-readable format documented in `docs/src/cli/doctor.md`.
fn print_human(report: &DoctorReport, converter_rows: &[SelfTestResult]) {
    println!("tensogram doctor");
    println!("================");
    println!();

    // Build section
    println!("Build");
    let nw = FEATURE_NAME_WIDTH;
    println!("  {:<nw$}{}", "tensogram", report.build.version);
    println!("  {:<nw$}{}", "wire-format", report.build.wire_version);
    println!("  {:<nw$}{}", "target", report.build.target);
    println!("  {:<nw$}{}", "profile", report.build.profile);
    println!();

    // Features section
    println!("Compiled-in features");
    for feat in &report.features {
        match &feat.state {
            FeatureState::On {
                backend,
                linkage,
                version,
            } => {
                let linkage_str = linkage_label(linkage);
                let ver_str = version.as_deref().unwrap_or("unknown");
                println!(
                    "  {:<nw$}on  ({backend} {linkage_str} {ver_str})",
                    feat.name,
                );
            }
            FeatureState::Off => {
                println!("  {:<nw$}off", feat.name);
            }
        }
    }
    println!();

    // Self-test section
    println!("Self-test");
    let lw = SELF_TEST_LABEL_WIDTH;
    let any_failed = report
        .self_test
        .iter()
        .chain(converter_rows.iter())
        .any(|r| r.is_failed());
    for row in report.self_test.iter().chain(converter_rows.iter()) {
        // Match-and-print directly to avoid an intermediate `String` allocation.
        match &row.outcome {
            SelfTestOutcome::Ok => println!("  {:<lw$}ok", row.label),
            SelfTestOutcome::Failed { error } => {
                println!("  {:<lw$}FAILED  ({error})", row.label)
            }
            SelfTestOutcome::Skipped { reason } => {
                println!("  {:<lw$}skipped ({reason})", row.label)
            }
        }
    }
    println!();

    println!(
        "Status: {}",
        if any_failed { "UNHEALTHY" } else { "HEALTHY" }
    );
}

/// Convert a [`Linkage`] into its user-facing label.
///
/// Kept as a free function (not a `Display` impl on `Linkage`) so the
/// renderer can decide how to display the value without committing every
/// downstream consumer of the library to the same vocabulary.
fn linkage_label(linkage: &tensogram::doctor::Linkage) -> &'static str {
    match linkage {
        tensogram::doctor::Linkage::Ffi => "FFI",
        tensogram::doctor::Linkage::PureRust => "pure-Rust",
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Run the doctor subcommand.
///
/// Collects the full [`DoctorReport`], appends converter self-test rows,
/// then renders human-readable or JSON output.  Returns [`DoctorFailed`]
/// when any self-test row has outcome `Failed`.
pub fn run(json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = tensogram::doctor::run_diagnostics();

    // Inject a forced failure when the hidden test env var is set.
    // This is only compiled in test builds to keep the production binary clean.
    #[cfg(test)]
    if std::env::var("RUN_DOCTOR_FORCE_FAILURE").as_deref() == Ok("1") {
        report.self_test.push(SelfTestResult::failed(
            "forced failure (test injection)",
            "RUN_DOCTOR_FORCE_FAILURE=1",
        ));
    }

    // Append converter feature rows — grib and netcdf are CLI-only features.
    use tensogram::doctor::{FeatureKind, FeatureStatus};

    report.features.extend([
        #[cfg(feature = "grib")]
        FeatureStatus::on("grib", FeatureKind::Converter, grib_backend_version()),
        #[cfg(not(feature = "grib"))]
        FeatureStatus::off("grib", FeatureKind::Converter),
        #[cfg(feature = "netcdf")]
        FeatureStatus::on("netcdf", FeatureKind::Converter, netcdf_backend_version()),
        #[cfg(not(feature = "netcdf"))]
        FeatureStatus::off("netcdf", FeatureKind::Converter),
    ]);

    // Converter self-tests (the CLI layer owns the fixtures + their feature gates).
    let converter_rows: Vec<SelfTestResult> = [
        #[cfg(feature = "grib")]
        self_test_grib(),
        #[cfg(not(feature = "grib"))]
        SelfTestResult::skipped(GRIB_LABEL, "feature 'grib' not built in"),
        #[cfg(feature = "netcdf")]
        self_test_netcdf_classic(),
        #[cfg(not(feature = "netcdf"))]
        SelfTestResult::skipped(NETCDF_CLASSIC_LABEL, "feature 'netcdf' not built in"),
        #[cfg(feature = "netcdf")]
        self_test_netcdf_hdf5(),
        #[cfg(not(feature = "netcdf"))]
        SelfTestResult::skipped(NETCDF_HDF5_LABEL, "feature 'netcdf' not built in"),
    ]
    .into_iter()
    .collect();

    let any_failed = report.self_test.iter().any(|r| r.is_failed())
        || converter_rows.iter().any(|r| r.is_failed());

    if json {
        // For JSON, fold converter rows into a single combined report so the
        // output is one object instead of two.
        report.self_test.extend(converter_rows);
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report, &converter_rows);
    }

    if any_failed {
        return Err(Box::new(DoctorFailed));
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialise all tests that call `run()` so the env-var injection in
    /// `exit_code_one_when_self_test_fails` cannot bleed into other tests.
    static RUN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire `RUN_LOCK`, recovering the inner guard if a previous test
    /// panicked while holding it.  Without this, a single panicking test
    /// would cascade-fail every other test in this module via mutex
    /// poisoning, which is a worse diagnostic than running concurrently.
    fn lock_or_recover() -> std::sync::MutexGuard<'static, ()> {
        match RUN_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn diagnostics_pass_on_healthy_build() {
        // The actual rendered output is exercised by the
        // `tests/doctor_cli.rs` integration tests; here we just confirm
        // the underlying report has no failures so any future failure
        // surfaces as a clear unit-test name rather than `run()` returning
        // `DoctorFailed`.
        let _guard = lock_or_recover();
        let report = tensogram::doctor::run_diagnostics();
        let any_failed = report.self_test.iter().any(|r| r.is_failed());
        assert!(!any_failed, "self-test should pass on a healthy build");
    }

    #[test]
    fn json_output_parses_as_json() {
        let _guard = lock_or_recover();
        let report = tensogram::doctor::run_diagnostics();
        let json = serde_json::to_string_pretty(&report).expect("serialisation failed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON parse failed");
        let obj = parsed
            .as_object()
            .expect("doctor report did not serialise as a JSON object");
        for key in ["build", "features", "self_test"] {
            assert!(obj.contains_key(key), "JSON missing top-level key '{key}'");
        }
    }

    #[test]
    fn exit_code_zero_on_healthy_build() {
        let _guard = lock_or_recover();
        let result = run(false);
        assert!(result.is_ok(), "run() failed: {result:?}");
    }

    #[test]
    fn exit_code_zero_on_healthy_build_with_json_output() {
        // Drives the JSON branch of `run()` — distinct from the human
        // renderer.  Captures stdout into the test buffer; no
        // assertion on payload content (the schema is locked by the
        // unit tests in `tensogram::doctor` and the integration tests
        // in `tests/doctor_cli.rs`).
        let _guard = lock_or_recover();
        let result = run(true);
        assert!(result.is_ok(), "run(json=true) failed: {result:?}");
    }

    #[test]
    fn doctor_failed_display_says_self_test_failed() {
        // The sentinel is caught by `main.rs` before any rendering, so
        // its `Display` impl normally goes uncalled.  Locking it in here
        // means a future rename of the message surfaces as a test
        // failure rather than silently breaking error-chain output for
        // any embedder that does dispatch on `e.to_string()`.
        let err = DoctorFailed;
        assert_eq!(err.to_string(), "doctor self-test failed");
    }

    #[test]
    fn exit_code_one_when_self_test_fails() {
        let _guard = lock_or_recover();
        // SAFETY: `set_var` is unsafe in Rust 2024 because it can race
        // with concurrent `getenv` from other threads (e.g. libc, glibc
        // dynamic loader).  We hold `RUN_LOCK`, so no other test in this
        // module is reading or writing this env var concurrently, and
        // `RUN_DOCTOR_FORCE_FAILURE` is private to this module — the
        // dynamic loader and other libraries do not consult it.
        unsafe {
            std::env::set_var("RUN_DOCTOR_FORCE_FAILURE", "1");
        }
        let result = run(false);
        // SAFETY: same justification as the matching `set_var` above.
        unsafe {
            std::env::remove_var("RUN_DOCTOR_FORCE_FAILURE");
        }
        assert!(
            result.is_err(),
            "run() should return Err when a self-test fails"
        );
        let err = result.unwrap_err();
        assert!(
            err.downcast_ref::<DoctorFailed>().is_some(),
            "error should be DoctorFailed"
        );
    }

    #[test]
    fn temp_file_guard_writes_then_deletes_on_drop() {
        let bytes = b"hello doctor";
        let path = {
            let guard = TempFileGuard::new(bytes, "test-guard.txt").unwrap();
            let p = guard.path().to_path_buf();
            assert!(p.exists(), "temp file should exist while guard is alive");
            assert_eq!(
                std::fs::read(&p).unwrap(),
                bytes,
                "temp file content should match bytes written"
            );
            p
        };
        assert!(
            !path.exists(),
            "temp file should be deleted after guard drops"
        );
    }

    #[test]
    fn temp_file_guard_drops_cleanly_when_file_is_already_gone() {
        // Drop should not panic if the file disappeared between creation and drop.
        let guard = TempFileGuard::new(b"x", "vanish-test.txt").unwrap();
        std::fs::remove_file(guard.path()).unwrap();
        // Implicit drop here exercises the NotFound path in `Drop`.
    }

    #[test]
    fn temp_file_guard_propagates_io_error_for_invalid_suffix() {
        // A suffix containing a path separator escapes the temp-dir base.
        // On most systems this resolves to a non-existent parent dir and
        // `fs::write` fails with NotFound — the guard must surface that
        // as an `Err` rather than panicking.
        let result = TempFileGuard::new(b"x", "no-such-dir/leaf.txt");
        assert!(
            result.is_err(),
            "writing through a missing intermediate dir should fail"
        );
    }

    #[cfg(feature = "grib")]
    #[test]
    fn decode_eccodes_version_rejects_zero_and_negative() {
        // libeccodes documents `codes_get_api_version()` as non-negative;
        // we still defensively map 0 and negative inputs to None so a
        // malformed return doesn't show as "0.0.0".
        assert_eq!(decode_eccodes_version(0), None);
        assert_eq!(decode_eccodes_version(-1), None);
        assert_eq!(decode_eccodes_version(std::ffi::c_long::MIN), None);
    }

    #[cfg(feature = "grib")]
    #[test]
    fn decode_eccodes_version_decodes_packed_integer() {
        // 2.46.0 is packed as 2 * 10_000 + 46 * 100 + 0 = 24600.
        assert_eq!(decode_eccodes_version(24_600).as_deref(), Some("2.46.0"));
        // Edge: leading-zero patch.
        assert_eq!(decode_eccodes_version(20_000).as_deref(), Some("2.0.0"));
        // Edge: single-digit major, two-digit minor, two-digit patch.
        assert_eq!(decode_eccodes_version(10_203).as_deref(), Some("1.2.3"));
    }

    #[test]
    #[cfg(feature = "grib")]
    fn embedded_grib_fixture_starts_with_grib_magic() {
        assert_eq!(&GRIB_FIXTURE[..4], b"GRIB", "GRIB fixture has wrong magic");
    }

    #[test]
    #[cfg(feature = "netcdf")]
    fn embedded_netcdf_classic_fixture_starts_with_cdf_magic() {
        assert_eq!(
            &NETCDF_CLASSIC_FIXTURE[..4],
            b"CDF\x01",
            "NetCDF-3 fixture has wrong magic"
        );
    }

    #[test]
    #[cfg(feature = "netcdf")]
    fn embedded_netcdf_hdf5_fixture_starts_with_hdf5_magic() {
        assert_eq!(
            &NETCDF_HDF5_FIXTURE[..4],
            b"\x89HDF",
            "NetCDF-4 fixture has wrong magic"
        );
    }
}

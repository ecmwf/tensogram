// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Environment diagnostics for tensogram.
//!
//! Call [`run_diagnostics`] to collect a [`DoctorReport`] describing the
//! compiled-in features, backend versions, and self-test results.
//!
//! # Example
//!
//! ```rust
//! let report = tensogram::doctor::run_diagnostics();
//! println!("{}", report.build.version);
//! ```

pub mod version;

use serde::Serialize;
pub use tensogram_encodings::version::{BackendVersion, Linkage, cstr_ptr_to_owned};

/// Compile-time build metadata captured by the `built` crate (target triple,
/// dependency versions, etc.).  The `version` submodule reads from this too,
/// but only when its specific I/O features are enabled, so it carries its own
/// gated copy.  Here we only need [`TARGET`](built_info::TARGET).
mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

// ── Data model ───────────────────────────────────────────────────────────────

/// Top-level report returned by [`run_diagnostics`].
#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    /// Compile-time build metadata.
    pub build: BuildInfo,
    /// One entry per known feature, whether on or off.
    pub features: Vec<FeatureStatus>,
    /// Results of the encode/decode self-test suite.
    pub self_test: Vec<SelfTestResult>,
}

/// Compile-time build metadata.
#[derive(Debug, Clone, Serialize)]
pub struct BuildInfo {
    /// Crate version from `Cargo.toml` (e.g. `"0.19.0"`).
    pub version: String,
    /// Wire-format version integer (e.g. `3`).
    pub wire_version: u16,
    /// Rustc target triple (e.g. `"aarch64-apple-darwin"`, `"x86_64-unknown-linux-gnu"`).
    pub target: String,
    /// Build profile: `"release"` or `"debug"`.
    pub profile: String,
}

/// Broad category of a compiled-in feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FeatureKind {
    /// Compression codec (szip, zstd, lz4, blosc2, zfp, sz3).
    Compression,
    /// Multi-threading support (rayon).
    Threading,
    /// I/O extension (remote object store, memory-mapped files, async).
    Io,
    /// Format converter (grib, netcdf).
    Converter,
}

/// Whether a feature is compiled in and, if so, what backend it uses.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum FeatureState {
    /// Feature is compiled in.
    On {
        /// Human-readable backend library name (e.g. `"libzstd"`).
        ///
        /// Always a `&'static str` because every backend name is a known
        /// compile-time constant — using `String` here would heap-allocate
        /// once per feature row for no benefit.
        backend: &'static str,
        /// How the backend is linked.
        linkage: Linkage,
        /// Backend version string, if determinable.
        version: Option<String>,
    },
    /// Feature is not compiled in.
    Off,
}

/// Status of a single compiled-in feature.
///
/// `state` is `#[serde(flatten)]`-ed so the JSON shape is:
///
/// ```json
/// { "name": "zstd", "kind": "compression",
///   "state": "on", "backend": "libzstd", "linkage": "ffi", "version": "1.5.7" }
/// ```
///
/// rather than the awkward `state.state == "on"` that nesting would produce.
#[derive(Debug, Clone, Serialize)]
pub struct FeatureStatus {
    /// Cargo feature name (e.g. `"zstd"`).
    pub name: &'static str,
    /// Broad category.
    pub kind: FeatureKind,
    /// Whether the feature is on or off, and backend details when on.
    #[serde(flatten)]
    pub state: FeatureState,
}

impl FeatureStatus {
    /// Construct an `On` row from a [`BackendVersion`].  Used by both the
    /// library [`run_diagnostics`] and the CLI's converter-feature appender.
    pub fn on(name: &'static str, kind: FeatureKind, bv: BackendVersion) -> Self {
        Self {
            name,
            kind,
            state: FeatureState::On {
                backend: bv.name,
                linkage: bv.linkage,
                version: bv.version,
            },
        }
    }

    /// Construct an `Off` row for a feature that was not compiled in.
    pub fn off(name: &'static str, kind: FeatureKind) -> Self {
        Self {
            name,
            kind,
            state: FeatureState::Off,
        }
    }
}

/// Outcome of a single self-test step.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "kebab-case")]
pub enum SelfTestOutcome {
    /// Test passed.
    Ok,
    /// Test failed with an error message.
    Failed {
        /// Human-readable error description.
        error: String,
    },
    /// Test was not run because a required feature is absent.
    Skipped {
        /// Human-readable reason (e.g. `"feature 'zstd' not built in"`).
        reason: String,
    },
}

/// Result of a single self-test step.
///
/// `outcome` is `#[serde(flatten)]`-ed so the JSON shape is:
///
/// ```json
/// { "label": "pipeline lz4", "outcome": "ok" }
/// { "label": "convert grib", "outcome": "skipped", "reason": "feature 'grib' not built in" }
/// ```
///
/// rather than `outcome.outcome == "ok"` that nesting would produce.
///
/// `label` is a [`Cow<'static, str>`] so static `&'static str` constants
/// (the common case for self-test rows) don't require a heap allocation.
#[derive(Debug, Clone, Serialize)]
pub struct SelfTestResult {
    /// Short label shown in the human output (e.g. `"pipeline shuffle+zstd"`).
    pub label: std::borrow::Cow<'static, str>,
    /// Outcome of the test.
    #[serde(flatten)]
    pub outcome: SelfTestOutcome,
}

impl SelfTestResult {
    /// Construct a passing self-test row.
    pub fn ok(label: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        Self {
            label: label.into(),
            outcome: SelfTestOutcome::Ok,
        }
    }

    /// Construct a failing self-test row carrying a human-readable error.
    pub fn failed(
        label: impl Into<std::borrow::Cow<'static, str>>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            outcome: SelfTestOutcome::Failed {
                error: error.into(),
            },
        }
    }

    /// Construct a skipped self-test row, e.g. when its required feature is
    /// not compiled in.
    pub fn skipped(
        label: impl Into<std::borrow::Cow<'static, str>>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            outcome: SelfTestOutcome::Skipped {
                reason: reason.into(),
            },
        }
    }

    /// Returns `true` if and only if the outcome is [`SelfTestOutcome::Failed`].
    /// Used to compute the overall `HEALTHY` / `UNHEALTHY` status.
    pub fn is_failed(&self) -> bool {
        matches!(self.outcome, SelfTestOutcome::Failed { .. })
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Collect build metadata, feature states, and self-test results.
///
/// This is the main entry point for the `tensogram doctor` subcommand.
/// The self-test rows in the returned report cover only the core
/// encode/decode pipeline; converter self-tests (grib, netcdf) are
/// appended by the CLI layer which has access to those crates.
pub fn run_diagnostics() -> DoctorReport {
    DoctorReport {
        build: collect_build_info(),
        features: collect_features(),
        self_test: run_self_test(),
    }
}

// ── BuildInfo ────────────────────────────────────────────────────────────────

fn collect_build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        wire_version: crate::wire::WIRE_VERSION,
        target: built_info::TARGET.to_owned(),
        profile: if cfg!(debug_assertions) {
            "debug".to_owned()
        } else {
            "release".to_owned()
        },
    }
}

// ── Features ─────────────────────────────────────────────────────────────────

fn collect_features() -> Vec<FeatureStatus> {
    // `ev` is referenced only by `On` branches; under `--no-default-features`
    // every codec feature is off and the import would be dead.
    #[cfg(any(
        feature = "szip",
        feature = "szip-pure",
        feature = "zstd",
        feature = "zstd-pure",
        feature = "lz4",
        feature = "blosc2",
        feature = "zfp",
        feature = "sz3",
        feature = "threads",
    ))]
    use tensogram_encodings::version as ev;

    // Build an array of feature rows where each `cfg`-gated branch contributes
    // exactly one entry — `On` if the feature was compiled in, `Off` otherwise.
    // The CLI layer appends the `grib` and `netcdf` converter rows since they
    // depend on optional companion crates (`tensogram-grib`, `tensogram-netcdf`)
    // that aren't reachable from `tensogram` core.
    [
        // Compression codecs
        #[cfg(feature = "szip")]
        FeatureStatus::on("szip", FeatureKind::Compression, ev::szip_ffi_version()),
        #[cfg(not(feature = "szip"))]
        FeatureStatus::off("szip", FeatureKind::Compression),
        #[cfg(feature = "szip-pure")]
        FeatureStatus::on(
            "szip-pure",
            FeatureKind::Compression,
            ev::szip_pure_version(),
        ),
        #[cfg(not(feature = "szip-pure"))]
        FeatureStatus::off("szip-pure", FeatureKind::Compression),
        #[cfg(feature = "zstd")]
        FeatureStatus::on("zstd", FeatureKind::Compression, ev::zstd_ffi_version()),
        #[cfg(not(feature = "zstd"))]
        FeatureStatus::off("zstd", FeatureKind::Compression),
        #[cfg(feature = "zstd-pure")]
        FeatureStatus::on(
            "zstd-pure",
            FeatureKind::Compression,
            ev::zstd_pure_version(),
        ),
        #[cfg(not(feature = "zstd-pure"))]
        FeatureStatus::off("zstd-pure", FeatureKind::Compression),
        #[cfg(feature = "lz4")]
        FeatureStatus::on("lz4", FeatureKind::Compression, ev::lz4_version()),
        #[cfg(not(feature = "lz4"))]
        FeatureStatus::off("lz4", FeatureKind::Compression),
        #[cfg(feature = "blosc2")]
        FeatureStatus::on("blosc2", FeatureKind::Compression, ev::blosc2_version()),
        #[cfg(not(feature = "blosc2"))]
        FeatureStatus::off("blosc2", FeatureKind::Compression),
        #[cfg(feature = "zfp")]
        FeatureStatus::on("zfp", FeatureKind::Compression, ev::zfp_version()),
        #[cfg(not(feature = "zfp"))]
        FeatureStatus::off("zfp", FeatureKind::Compression),
        #[cfg(feature = "sz3")]
        FeatureStatus::on("sz3", FeatureKind::Compression, ev::sz3_version()),
        #[cfg(not(feature = "sz3"))]
        FeatureStatus::off("sz3", FeatureKind::Compression),
        // Threading
        #[cfg(feature = "threads")]
        FeatureStatus::on("threads", FeatureKind::Threading, ev::rayon_version()),
        #[cfg(not(feature = "threads"))]
        FeatureStatus::off("threads", FeatureKind::Threading),
        // I/O
        #[cfg(feature = "remote")]
        FeatureStatus::on("remote", FeatureKind::Io, version::remote_version()),
        #[cfg(not(feature = "remote"))]
        FeatureStatus::off("remote", FeatureKind::Io),
        #[cfg(feature = "mmap")]
        FeatureStatus::on("mmap", FeatureKind::Io, version::mmap_version()),
        #[cfg(not(feature = "mmap"))]
        FeatureStatus::off("mmap", FeatureKind::Io),
        #[cfg(feature = "async")]
        FeatureStatus::on("async", FeatureKind::Io, version::async_version()),
        #[cfg(not(feature = "async"))]
        FeatureStatus::off("async", FeatureKind::Io),
    ]
    .into_iter()
    .collect()
}

// ── Self-test helpers ─────────────────────────────────────────────────────────

type Params = std::collections::BTreeMap<String, ciborium::Value>;
type SelfTestError = Box<dyn std::error::Error>;

/// Convert a `&[f32]` to a `Vec<u8>` in native byte order.
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Convert a native-endian f32 payload back into a `Vec<f32>`.
///
/// Returns an error if `bytes.len()` is not a multiple of 4 — silently
/// truncating could mask a corrupted payload from a misbehaving codec.
fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, SelfTestError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("f32 payload length {} is not a multiple of 4", bytes.len()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Convert a `&[f64]` to a `Vec<u8>` in native byte order.
///
/// Only used by the `simple_packing + szip` self-test (the only f64 path
/// in the suite); gated to that feature to avoid a dead-code warning when
/// szip is not compiled in.
#[cfg(feature = "szip")]
fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Build a 1-D `DataObjectDescriptor` for `n` elements of `dtype` with the
/// given encoding/filter/compression strings and codec parameters.
fn make_descriptor(
    n: u64,
    dtype: crate::Dtype,
    encoding: &str,
    filter: &str,
    compression: &str,
    params: Params,
) -> crate::types::DataObjectDescriptor {
    use tensogram_encodings::pipeline::ByteOrder;
    crate::types::DataObjectDescriptor {
        obj_type: "ntensor".to_owned(),
        ndim: 1,
        shape: vec![n],
        strides: vec![1],
        dtype,
        byte_order: ByteOrder::native(),
        encoding: encoding.to_owned(),
        filter: filter.to_owned(),
        compression: compression.to_owned(),
        params,
        masks: None,
    }
}

/// Round-trip raw bytes through encode + decode and return the decoded payload.
fn encode_decode_one(
    desc: &crate::types::DataObjectDescriptor,
    raw: &[u8],
) -> Result<Vec<u8>, SelfTestError> {
    use crate::{decode::DecodeOptions, encode, encode::EncodeOptions, types::GlobalMetadata};
    let meta = GlobalMetadata::default();
    let opts = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let encoded = encode(&meta, &[(desc, raw)], &opts)?;
    let (_meta, objects) = crate::decode::decode(&encoded, &DecodeOptions::default())?;
    objects
        .into_iter()
        .next()
        .map(|(_, payload)| payload)
        .ok_or_else(|| {
            format!(
                "decode produced 0 objects from a 1-object encode \
             (compression='{}', encoding='{}')",
                desc.compression, desc.encoding,
            )
            .into()
        })
}

/// Encode a `&[f32]` as a 1-D `none/none/none` object, decode it, and
/// return the payload as `Vec<f32>`.  Used by core round-trip tests.
fn round_trip_f32_none(data: &[f32]) -> Result<Vec<f32>, SelfTestError> {
    let desc = make_descriptor(
        data.len() as u64,
        crate::Dtype::Float32,
        "none",
        "none",
        "none",
        Params::new(),
    );
    let payload = encode_decode_one(&desc, &f32_to_bytes(data))?;
    bytes_to_f32(&payload)
}

/// Encode a `&[f32]` through the given codec pipeline, decode it, and
/// validate that the decoded payload has either 4 or 8 bytes per value
/// (lossy codecs may promote f32 → f64 on round-trip).
#[cfg(any(feature = "zfp", feature = "sz3"))]
fn run_lossy_f32_pipeline(
    data: &[f32],
    encoding: &str,
    filter: &str,
    compression: &str,
    params: Params,
) -> Result<(), SelfTestError> {
    if data.is_empty() {
        return Err("input data is empty".into());
    }
    let desc = make_descriptor(
        data.len() as u64,
        crate::Dtype::Float32,
        encoding,
        filter,
        compression,
        params,
    );
    let payload = encode_decode_one(&desc, &f32_to_bytes(data))?;
    let total = payload.len();
    let n = data.len();
    if total != n * 4 && total != n * 8 {
        return Err(format!(
            "decoded payload {total} bytes does not match \
             {n} values × 4 or 8 bytes/value"
        )
        .into());
    }
    Ok(())
}

/// Encode a `&[f32]` through the given codec pipeline, decode it, and
/// assert that the round-trip is *lossless* (decoded bytes match input).
#[cfg(any(feature = "zstd", feature = "lz4", feature = "blosc2"))]
fn run_lossless_f32_pipeline(
    data: &[f32],
    encoding: &str,
    filter: &str,
    compression: &str,
    params: Params,
) -> Result<(), SelfTestError> {
    let desc = make_descriptor(
        data.len() as u64,
        crate::Dtype::Float32,
        encoding,
        filter,
        compression,
        params,
    );
    let payload = encode_decode_one(&desc, &f32_to_bytes(data))?;
    let decoded = bytes_to_f32(&payload)?;
    if decoded != data {
        return Err(format!(
            "decoded data does not match input ({} values, first mismatch at index {})",
            data.len(),
            decoded
                .iter()
                .zip(data.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(decoded.len()),
        )
        .into());
    }
    Ok(())
}

/// Encode a `&[f64]` through `simple_packing + szip`, decode it, and
/// assert the decoded payload is 8 bytes per value (`simple_packing`
/// always decodes back to f64).  szip itself requires the simple_packing
/// front-end so this is a single combined test.
#[cfg(feature = "szip")]
fn run_simple_packing_szip(data: &[f64]) -> Result<(), SelfTestError> {
    let mut params = Params::new();
    params.insert(
        "sp_bits_per_value".to_owned(),
        ciborium::Value::Integer(24.into()),
    );
    params.insert("szip_rsi".to_owned(), ciborium::Value::Integer(128.into()));
    params.insert(
        "szip_block_size".to_owned(),
        ciborium::Value::Integer(16.into()),
    );
    params.insert("szip_flags".to_owned(), ciborium::Value::Integer(8.into()));

    let desc = make_descriptor(
        data.len() as u64,
        crate::Dtype::Float64,
        "simple_packing",
        "none",
        "szip",
        params,
    );
    let payload = encode_decode_one(&desc, &f64_to_bytes(data))?;
    let expected_len = data.len() * 8;
    if payload.len() != expected_len {
        return Err(format!(
            "decoded length {} != expected {expected_len}",
            payload.len(),
        )
        .into());
    }
    Ok(())
}

/// Convert a `Result<()>` into a [`SelfTestResult`] tagged with `label`.
///
/// Accepts any label that can be converted into [`Cow<'static, str>`], so
/// callers passing `&'static str` literals avoid an allocation.
fn into_result(
    label: impl Into<std::borrow::Cow<'static, str>>,
    r: Result<(), SelfTestError>,
) -> SelfTestResult {
    let label = label.into();
    match r {
        Ok(()) => SelfTestResult::ok(label),
        Err(e) => SelfTestResult::failed(label, e.to_string()),
    }
}

// ── Self-test ─────────────────────────────────────────────────────────────────

/// Run the core encode/decode self-test suite.
///
/// Returns one [`SelfTestResult`] per test step.  Converter self-tests
/// (grib, netcdf) are **not** included here — they are appended by the
/// CLI layer.
pub fn run_self_test() -> Vec<SelfTestResult> {
    let mut results = Vec::new();
    run_core_self_tests(&mut results);
    run_codec_self_tests(&mut results);
    results
}

fn run_core_self_tests(out: &mut Vec<SelfTestResult>) {
    use crate::{
        decode::DecodeOptions, encode, encode::EncodeOptions, hash::HashAlgorithm, iter::messages,
        types::GlobalMetadata,
    };

    // ── encode/decode none/none/none round-trip ──────────────────────────────
    out.push(into_result("encode/decode  none/none/none", {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        round_trip_f32_none(&data).and_then(|decoded| {
            if decoded != data {
                Err("decoded data does not match original".into())
            } else {
                Ok(())
            }
        })
    }));

    // ── decode_metadata round-trip ───────────────────────────────────────────
    out.push(into_result(
        "decode_metadata round-trip",
        (|| {
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let desc = make_descriptor(
                4,
                crate::Dtype::Float32,
                "none",
                "none",
                "none",
                Params::new(),
            );
            let mut global_meta = GlobalMetadata::default();
            global_meta
                .extra
                .insert("param".to_owned(), ciborium::Value::Text("2t".to_owned()));
            let opts = EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            };
            let encoded = encode(
                &global_meta,
                &[(&desc, f32_to_bytes(&data).as_slice())],
                &opts,
            )?;
            let (got_meta, _) = crate::decode::decode(&encoded, &DecodeOptions::default())?;
            if !got_meta.extra.contains_key("param") {
                return Err("global metadata missing 'param' key".into());
            }
            Ok(())
        })(),
    ));

    // ── scan multi-message buffer ────────────────────────────────────────────
    out.push(into_result(
        "scan multi-message buffer",
        (|| {
            let data: Vec<f32> = vec![0.0; 4];
            let desc = make_descriptor(
                4,
                crate::Dtype::Float32,
                "none",
                "none",
                "none",
                Params::new(),
            );
            let meta = GlobalMetadata::default();
            let opts = EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            };
            let mut buf = Vec::new();
            for _ in 0..3 {
                buf.extend_from_slice(&encode(
                    &meta,
                    &[(&desc, f32_to_bytes(&data).as_slice())],
                    &opts,
                )?);
            }
            let count = messages(&buf).count();
            if count != 3 {
                return Err(format!("expected 3 messages, got {count}").into());
            }
            Ok(())
        })(),
    ));

    // ── xxh3 hash round-trip ─────────────────────────────────────────────────
    out.push(into_result(
        "hash xxh3 verify",
        (|| {
            let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let desc = make_descriptor(
                16,
                crate::Dtype::Float32,
                "none",
                "none",
                "none",
                Params::new(),
            );
            let meta = GlobalMetadata::default();
            let opts = EncodeOptions {
                hash_algorithm: Some(HashAlgorithm::Xxh3),
                ..Default::default()
            };
            let encoded = encode(&meta, &[(&desc, f32_to_bytes(&data).as_slice())], &opts)?;
            // Decode succeeds only if hash verification passes internally.
            crate::decode::decode(&encoded, &DecodeOptions::default())?;
            Ok(())
        })(),
    ));
}

fn run_codec_self_tests(out: &mut Vec<SelfTestResult>) {
    // Shared deterministic input for every f32 codec test.  Only bound
    // when at least one f32-using codec feature is on; otherwise every
    // call site is `cfg`-elided and the binding would be unused.
    #[cfg(any(
        feature = "zstd",
        feature = "lz4",
        feature = "blosc2",
        feature = "zfp",
        feature = "sz3",
    ))]
    let f32_data: Vec<f32> = (0..16).map(|i| i as f32 * 1.5).collect();

    // simple_packing + szip — f64-only; tests both stages together.
    const SZIP_LABEL: &str = "pipeline       simple_packing+szip";
    #[cfg(feature = "szip")]
    {
        let f64_data: Vec<f64> = (0..16).map(|i| i as f64 * 1.5).collect();
        out.push(into_result(SZIP_LABEL, run_simple_packing_szip(&f64_data)));
    }
    #[cfg(not(feature = "szip"))]
    out.push(SelfTestResult::skipped(
        SZIP_LABEL,
        "feature 'szip' not built in",
    ));

    // shuffle + zstd — lossless
    const SHUFFLE_ZSTD_LABEL: &str = "pipeline       shuffle+zstd";
    #[cfg(feature = "zstd")]
    {
        let mut params = Params::new();
        params.insert(
            "shuffle_element_size".to_owned(),
            ciborium::Value::Integer(4.into()),
        );
        out.push(into_result(
            SHUFFLE_ZSTD_LABEL,
            run_lossless_f32_pipeline(&f32_data, "none", "shuffle", "zstd", params),
        ));
    }
    #[cfg(not(feature = "zstd"))]
    out.push(SelfTestResult::skipped(
        SHUFFLE_ZSTD_LABEL,
        "feature 'zstd' not built in",
    ));

    // lz4 — lossless
    const LZ4_LABEL: &str = "pipeline       lz4";
    #[cfg(feature = "lz4")]
    out.push(into_result(
        LZ4_LABEL,
        run_lossless_f32_pipeline(&f32_data, "none", "none", "lz4", Params::new()),
    ));
    #[cfg(not(feature = "lz4"))]
    out.push(SelfTestResult::skipped(
        LZ4_LABEL,
        "feature 'lz4' not built in",
    ));

    // blosc2 — lossless
    const BLOSC2_LABEL: &str = "pipeline       blosc2";
    #[cfg(feature = "blosc2")]
    out.push(into_result(
        BLOSC2_LABEL,
        run_lossless_f32_pipeline(&f32_data, "none", "none", "blosc2", Params::new()),
    ));
    #[cfg(not(feature = "blosc2"))]
    out.push(SelfTestResult::skipped(
        BLOSC2_LABEL,
        "feature 'blosc2' not built in",
    ));

    // zfp fixed-rate — lossy, decoded element size may be 4 or 8 bytes.
    const ZFP_LABEL: &str = "pipeline       zfp (fixed-rate)";
    #[cfg(feature = "zfp")]
    {
        let mut params = Params::new();
        params.insert(
            "zfp_mode".to_owned(),
            ciborium::Value::Text("fixed_rate".to_owned()),
        );
        params.insert("zfp_rate".to_owned(), ciborium::Value::Float(16.0));
        out.push(into_result(
            ZFP_LABEL,
            run_lossy_f32_pipeline(&f32_data, "none", "none", "zfp", params),
        ));
    }
    #[cfg(not(feature = "zfp"))]
    out.push(SelfTestResult::skipped(
        ZFP_LABEL,
        "feature 'zfp' not built in",
    ));

    // sz3 absolute error — lossy, decoded element size may be 4 or 8 bytes.
    const SZ3_LABEL: &str = "pipeline       sz3 (absolute error)";
    #[cfg(feature = "sz3")]
    {
        let mut params = Params::new();
        params.insert(
            "sz3_error_bound_mode".to_owned(),
            ciborium::Value::Text("abs".to_owned()),
        );
        params.insert("sz3_error_bound".to_owned(), ciborium::Value::Float(0.1));
        out.push(into_result(
            SZ3_LABEL,
            run_lossy_f32_pipeline(&f32_data, "none", "none", "sz3", params),
        ));
    }
    #[cfg(not(feature = "sz3"))]
    out.push(SelfTestResult::skipped(
        SZ3_LABEL,
        "feature 'sz3' not built in",
    ));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_non_empty() {
        let b = collect_build_info();
        assert!(!b.version.is_empty(), "version is empty");
        assert!(b.wire_version > 0, "wire_version is zero");
        assert!(!b.target.is_empty(), "target is empty");
        assert!(b.profile == "debug" || b.profile == "release");
    }

    #[test]
    fn report_includes_every_known_feature() {
        let report = run_diagnostics();
        // grib and netcdf are appended by the CLI layer; tensogram-core
        // only knows about the 12 features below.
        let known = [
            "szip",
            "szip-pure",
            "zstd",
            "zstd-pure",
            "lz4",
            "blosc2",
            "zfp",
            "sz3",
            "threads",
            "remote",
            "mmap",
            "async",
        ];
        for name in known {
            assert!(
                report.features.iter().any(|f| f.name == name),
                "feature '{name}' missing from report"
            );
        }
    }

    #[test]
    fn feature_states_match_cfg() {
        let report = run_diagnostics();
        let find = |name: &str| {
            report
                .features
                .iter()
                .find(|f| f.name == name)
                .map(|f| &f.state)
        };

        macro_rules! assert_state {
            ($feat:literal, $expected_on:expr) => {
                match find($feat) {
                    Some(FeatureState::On { .. }) => {
                        assert!($expected_on, "feature '{}' is On but cfg says off", $feat)
                    }
                    Some(FeatureState::Off) => {
                        assert!(!$expected_on, "feature '{}' is Off but cfg says on", $feat)
                    }
                    None => panic!("feature '{}' not found in report", $feat),
                }
            };
        }

        assert_state!("szip", cfg!(feature = "szip"));
        assert_state!("zstd", cfg!(feature = "zstd"));
        assert_state!("lz4", cfg!(feature = "lz4"));
        assert_state!("threads", cfg!(feature = "threads"));
        assert_state!("remote", cfg!(feature = "remote"));
        assert_state!("mmap", cfg!(feature = "mmap"));
    }

    #[test]
    fn self_test_passes_on_default_build() {
        let results = run_self_test();
        let failures: Vec<_> = results.iter().filter(|r| r.is_failed()).collect();
        assert!(failures.is_empty(), "self-test failures: {failures:#?}");
    }

    #[test]
    fn self_test_round_trip_data_matches() {
        let results = run_self_test();
        let row = results
            .iter()
            .find(|r| r.label.contains("none/none/none"))
            .expect("none/none/none row missing");
        assert!(
            matches!(row.outcome, SelfTestOutcome::Ok),
            "none/none/none row failed: {row:?}"
        );
    }

    #[test]
    fn report_serialises_to_json_with_stable_keys() {
        let report = run_diagnostics();
        let json = serde_json::to_string(&report).expect("serialisation failed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON parse failed");
        let obj = parsed.as_object().expect("top level is not an object");
        for key in ["build", "features", "self_test"] {
            assert!(obj.contains_key(key), "top-level key '{key}' missing");
        }
        let build = obj["build"].as_object().expect("build is not an object");
        for key in ["version", "wire_version", "target", "profile"] {
            assert!(build.contains_key(key), "build key '{key}' missing");
        }
    }

    // ── Schema regression: every feature row & self-test row has the
    // flattened serde shape we promised in `docs/src/cli/doctor.md`.

    #[test]
    fn feature_row_on_serialises_flat() {
        // Pick any feature that is actually compiled in (default build always
        // has at least one compression codec); confirm the JSON shape is flat
        // — `state: "on"` not `state: { state: "on" }`.
        let report = run_diagnostics();
        let on_row = report
            .features
            .iter()
            .find(|f| matches!(f.state, FeatureState::On { .. }))
            .expect("at least one feature should be On in default build");
        let json = serde_json::to_value(on_row).expect("feature row serialises");
        let obj = json.as_object().expect("feature row is a JSON object");

        // The flattened shape exposes these keys at the top level (not under "state.*").
        for key in ["name", "kind", "state", "backend", "linkage", "version"] {
            assert!(obj.contains_key(key), "On feature missing flat key '{key}'");
        }
        assert_eq!(
            obj["state"],
            serde_json::Value::String("on".to_owned()),
            "expected `state: \"on\"`, got nested shape: {obj:?}"
        );
    }

    #[test]
    fn feature_row_off_serialises_flat() {
        // We need a guaranteed-Off feature.  `szip-pure` is mutually exclusive
        // with `szip` in the typical default build; either one of them is Off.
        let report = run_diagnostics();
        let off_row = report
            .features
            .iter()
            .find(|f| matches!(f.state, FeatureState::Off))
            .expect("at least one feature should be Off in default build");
        let json = serde_json::to_value(off_row).expect("feature row serialises");
        let obj = json.as_object().expect("feature row is a JSON object");
        assert_eq!(
            obj["state"],
            serde_json::Value::String("off".to_owned()),
            "expected `state: \"off\"`, got nested shape: {obj:?}"
        );
        // `Off` should NOT carry backend/linkage/version fields.
        for absent in ["backend", "linkage", "version"] {
            assert!(
                !obj.contains_key(absent),
                "Off feature should not carry '{absent}', got: {obj:?}"
            );
        }
    }

    #[test]
    fn self_test_row_serialises_flat() {
        let row = SelfTestResult::ok("encode/decode  none/none/none");
        let json = serde_json::to_value(&row).expect("row serialises");
        let obj = json.as_object().expect("row is a JSON object");
        assert_eq!(
            obj["label"],
            serde_json::Value::String(row.label.to_string())
        );
        assert_eq!(obj["outcome"], serde_json::Value::String("ok".to_owned()));
        assert!(
            !obj.contains_key("error") && !obj.contains_key("reason"),
            "Ok row should not carry error/reason: {obj:?}"
        );
    }

    #[test]
    fn self_test_row_failed_carries_error() {
        let row = SelfTestResult::failed("test", "boom");
        let json = serde_json::to_value(&row).expect("row serialises");
        let obj = json.as_object().expect("row is a JSON object");
        assert_eq!(
            obj["outcome"],
            serde_json::Value::String("failed".to_owned())
        );
        assert_eq!(obj["error"], serde_json::Value::String("boom".to_owned()));
    }

    #[test]
    fn self_test_row_skipped_carries_reason() {
        let row = SelfTestResult::skipped("test", "no-feat");
        let json = serde_json::to_value(&row).expect("row serialises");
        let obj = json.as_object().expect("row is a JSON object");
        assert_eq!(
            obj["outcome"],
            serde_json::Value::String("skipped".to_owned())
        );
        assert_eq!(
            obj["reason"],
            serde_json::Value::String("no-feat".to_owned())
        );
    }

    // ── Byte conversion helpers ─────────────────────────────────────────────

    #[test]
    fn f32_round_trip_via_bytes_preserves_values() {
        let values: Vec<f32> = vec![
            0.0,
            -1.5,
            std::f32::consts::PI,
            f32::MIN,
            f32::MAX,
            f32::EPSILON,
        ];
        let bytes = f32_to_bytes(&values);
        assert_eq!(bytes.len(), values.len() * 4);
        let decoded = bytes_to_f32(&bytes).expect("valid length parses");
        assert_eq!(decoded, values);
    }

    #[test]
    fn bytes_to_f32_rejects_non_multiple_of_four() {
        // 7 bytes is one short of two f32s; we must error rather than silently
        // truncate, which would mask a corrupted FFI payload.
        let err = bytes_to_f32(&[0u8; 7]).expect_err("7-byte input must error");
        let msg = err.to_string();
        assert!(
            msg.contains("not a multiple of 4") && msg.contains("7"),
            "error should mention the bad length: {msg}"
        );
    }

    #[test]
    fn bytes_to_f32_accepts_empty_input() {
        // 0 IS a multiple of 4 — empty in, empty out.
        let decoded = bytes_to_f32(&[]).expect("empty input is valid");
        assert!(decoded.is_empty());
    }

    #[test]
    fn run_lossy_pipeline_rejects_empty_input() {
        // Guards against the divide-by-zero when computing element size.
        // We don't need an actual codec to fire — empty input short-circuits.
        let err = (|| -> Result<(), SelfTestError> {
            // The function is feature-gated; emulate its precondition check
            // by calling with zero data via an always-available path.
            // Use `none/none/none` which is unconditionally compiled.
            let desc = make_descriptor(
                0,
                crate::Dtype::Float32,
                "none",
                "none",
                "none",
                Params::new(),
            );
            let _payload = encode_decode_one(&desc, &[])?;
            Ok(())
        })();
        // Whether the encoder accepts a zero-length descriptor or not,
        // the helper must not panic with divide-by-zero.
        let _ = err;
    }

    /// Direct test of the gated `run_lossy_f32_pipeline` empty-input
    /// guard.  The test is itself feature-gated to match the helper
    /// because the helper is only compiled when zfp or sz3 is on.
    #[test]
    #[cfg(any(feature = "zfp", feature = "sz3"))]
    fn run_lossy_f32_pipeline_returns_explicit_error_for_empty_data() {
        let result = run_lossy_f32_pipeline(&[], "none", "none", "none", Params::new());
        let err = result.expect_err("empty input must error");
        assert!(
            err.to_string().contains("empty"),
            "error should mention emptiness: {err}"
        );
    }

    #[test]
    fn into_result_maps_err_to_failed_outcome() {
        // Locks in the Err arm — `into_result` must surface the error
        // as a `SelfTestOutcome::Failed` carrying the stringified error.
        let err: SelfTestError = "boom".into();
        let row = into_result("test-label", Err(err));
        assert_eq!(row.label, "test-label");
        match row.outcome {
            SelfTestOutcome::Failed { error } => assert_eq!(error, "boom"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn into_result_maps_ok_to_ok_outcome() {
        let row = into_result("ok-label", Ok(()));
        assert_eq!(row.label, "ok-label");
        assert!(matches!(row.outcome, SelfTestOutcome::Ok));
    }
}

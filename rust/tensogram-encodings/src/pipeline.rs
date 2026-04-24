// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::borrow::Cow;

#[cfg(feature = "blosc2")]
use crate::compression::Blosc2Compressor;
#[cfg(feature = "lz4")]
use crate::compression::Lz4Compressor;
#[cfg(feature = "sz3")]
use crate::compression::Sz3Compressor;
#[cfg(feature = "szip")]
use crate::compression::SzipCompressor;
#[cfg(feature = "szip-pure")]
use crate::compression::SzipPureCompressor;
#[cfg(feature = "zfp")]
use crate::compression::ZfpCompressor;
#[cfg(feature = "zstd")]
use crate::compression::ZstdCompressor;
#[cfg(feature = "zstd-pure")]
use crate::compression::ZstdPureCompressor;
use crate::compression::{CompressResult, CompressionError, Compressor};
use crate::shuffle;
use crate::simple_packing::{self, PackingError, SimplePackingParams};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ByteOrder {
    Big,
    Little,
}

impl ByteOrder {
    /// Returns the byte order of the platform this code was compiled for.
    #[inline]
    pub fn native() -> Self {
        #[cfg(target_endian = "little")]
        {
            ByteOrder::Little
        }
        #[cfg(target_endian = "big")]
        {
            ByteOrder::Big
        }
    }
}

/// Reverse bytes within each `unit_size`-byte chunk of `data`, converting
/// between big-endian and little-endian representations in place.
///
/// No-op when `unit_size <= 1` (single-byte types have no byte order).
/// Returns an error if `data.len()` is not a multiple of `unit_size`.
/// For complex types, pass the scalar component size (e.g. 4 for complex64)
/// so that each float32 component is swapped independently.
pub fn byteswap(data: &mut [u8], unit_size: usize) -> Result<(), PipelineError> {
    if unit_size <= 1 {
        return Ok(());
    }
    if !data.len().is_multiple_of(unit_size) {
        return Err(PipelineError::Range(format!(
            "byteswap: data length {} is not a multiple of unit_size {}",
            data.len(),
            unit_size,
        )));
    }
    for chunk in data.chunks_exact_mut(unit_size) {
        chunk.reverse();
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("encoding error: {0}")]
    Encoding(#[from] PackingError),
    #[error("compression error: {0}")]
    Compression(#[from] CompressionError),
    #[error("shuffle error: {0}")]
    Shuffle(String),
    #[error("range error: {0}")]
    Range(String),
    #[error("unknown encoding: {0}")]
    UnknownEncoding(String),
    #[error("unknown filter: {0}")]
    UnknownFilter(String),
    #[error("unknown compression: {0}")]
    UnknownCompression(String),
}

#[derive(Debug, Clone)]
pub enum EncodingType {
    None,
    SimplePacking(SimplePackingParams),
}

impl std::fmt::Display for EncodingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodingType::None => write!(f, "none"),
            EncodingType::SimplePacking(_) => write!(f, "simple_packing"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FilterType {
    None,
    Shuffle { element_size: usize },
}

#[cfg(feature = "blosc2")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Blosc2Codec {
    Blosclz,
    Lz4,
    Lz4hc,
    Zlib,
    Zstd,
}

#[cfg(feature = "zfp")]
#[derive(Debug, Clone)]
pub enum ZfpMode {
    FixedRate { rate: f64 },
    FixedPrecision { precision: u32 },
    FixedAccuracy { tolerance: f64 },
}

#[cfg(feature = "sz3")]
#[derive(Debug, Clone)]
pub enum Sz3ErrorBound {
    Absolute(f64),
    Relative(f64),
    Psnr(f64),
}

#[derive(Debug, Clone)]
pub enum CompressionType {
    None,
    #[cfg(any(feature = "szip", feature = "szip-pure"))]
    Szip {
        rsi: u32,
        block_size: u32,
        flags: u32,
        bits_per_sample: u32,
    },
    #[cfg(any(feature = "zstd", feature = "zstd-pure"))]
    Zstd {
        level: i32,
    },
    #[cfg(feature = "lz4")]
    Lz4,
    #[cfg(feature = "blosc2")]
    Blosc2 {
        codec: Blosc2Codec,
        clevel: i32,
        typesize: usize,
    },
    #[cfg(feature = "zfp")]
    Zfp {
        mode: ZfpMode,
    },
    #[cfg(feature = "sz3")]
    Sz3 {
        error_bound: Sz3ErrorBound,
    },
    /// Bit-level run-length encoding.  Bitmask-only — the pipeline
    /// build-layer rejects this codec on any other dtype.
    /// See [`crate::compression::RleCompressor`].
    Rle,
    /// Roaring bitmap.  Bitmask-only.
    /// See [`crate::compression::RoaringCompressor`].
    Roaring,
}

/// Selects which backend to use when both FFI and pure-Rust implementations
/// are compiled in for the same codec (szip or zstd).
///
/// When only one feature is enabled the backend field is ignored — the
/// available implementation is always used.
///
/// The default is resolved once from the `TENSOGRAM_COMPRESSION_BACKEND`
/// environment variable (values: `ffi` or `pure`).
/// On `wasm32` the default is always `Pure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionBackend {
    /// Use the C FFI implementation (libaec for szip, libzstd for zstd).
    /// Falls back to pure-Rust if the FFI feature is not compiled in.
    Ffi,
    /// Use the pure-Rust implementation (tensogram-szip, ruzstd).
    /// Falls back to FFI if the pure feature is not compiled in.
    Pure,
}

impl Default for CompressionBackend {
    fn default() -> Self {
        default_compression_backend()
    }
}

/// Resolve the default compression backend from environment variables.
///
/// - `TENSOGRAM_COMPRESSION_BACKEND=pure|ffi` overrides the default for
///   both szip and zstd.
/// - On `wasm32` the default is always `Pure` (FFI backends cannot exist).
/// - On native the default is `Ffi` (faster, battle-tested).
pub fn default_compression_backend() -> CompressionBackend {
    static DEFAULT: OnceLock<CompressionBackend> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        if cfg!(target_arch = "wasm32") {
            return CompressionBackend::Pure;
        }
        // Check env — accept "pure" or "ffi" (case-insensitive)
        if let Ok(val) = std::env::var("TENSOGRAM_COMPRESSION_BACKEND") {
            return parse_backend(&val);
        }
        // Native default: prefer FFI
        CompressionBackend::Ffi
    })
}

fn parse_backend(val: &str) -> CompressionBackend {
    match val.trim().to_ascii_lowercase().as_str() {
        "pure" | "rust" => CompressionBackend::Pure,
        _ => CompressionBackend::Ffi,
    }
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub encoding: EncodingType,
    pub filter: FilterType,
    pub compression: CompressionType,
    pub num_values: usize,
    pub byte_order: ByteOrder,
    pub dtype_byte_width: usize,
    /// The size of the fundamental scalar component for byte-order swapping.
    /// Equal to `dtype_byte_width` for simple types.  For complex types the
    /// swap must operate on each float component independently, so this is
    /// half of `dtype_byte_width` (e.g. 4 for complex64, 8 for complex128).
    pub swap_unit_size: usize,
    /// Which backend to use for szip / zstd when both are compiled in.
    pub compression_backend: CompressionBackend,
    /// Intra-codec thread budget for this object.
    ///
    /// - `0` — sequential (default; preserves pre-threads behaviour).
    /// - `N ≥ 1` — codec may use up to `N` threads internally.  Only a
    ///   subset of codecs honour this (currently: blosc2, zstd FFI,
    ///   simple_packing, shuffle); others ignore it.  Output bytes are
    ///   byte-identical regardless of `N`.
    ///
    /// Callers should expect this to be set by the `tensogram`
    /// layer after consulting
    /// [`EncodeOptions.threads`](../../../tensogram/encode/struct.EncodeOptions.html#structfield.threads)
    /// and the small-message threshold; direct pipeline callers may
    /// leave it at `0`.
    pub intra_codec_threads: u32,
    /// Opt-in: compute xxh3-64 over the final encoded bytes inline with
    /// encoding, exposing the digest via [`PipelineResult::hash`].
    ///
    /// - `false` (default) — the pipeline does no hashing; callers that
    ///   need a hash must walk the output buffer themselves with
    ///   `xxhash_rust::xxh3::xxh3_64`.
    /// - `true` — the pipeline drives an `Xxh3Default` hasher in lockstep
    ///   with the codec output, eliminating a second pass over the
    ///   encoded buffer.  The digest is bit-identical to
    ///   `xxhash_rust::xxh3::xxh3_64(&encoded_bytes)` by construction.
    ///
    /// Hashing always runs in the calling thread *after* any internal
    /// codec parallelism has joined; it never participates in the
    /// intra-codec thread budget.  For transparent codecs the hash is
    /// byte-identical across `intra_codec_threads` values; for opaque
    /// codecs the hash tracks the codec's output bytes (which may reorder
    /// by worker-completion order).
    pub compute_hash: bool,
}

pub struct PipelineResult {
    pub encoded_bytes: Vec<u8>,
    /// Block offsets produced by compressors that support random access (szip, blosc2).
    pub block_offsets: Option<Vec<u64>>,
    /// xxh3-64 digest over `encoded_bytes`, produced inline with encoding
    /// when [`PipelineConfig::compute_hash`] was `true`.  `None` otherwise.
    ///
    /// The digest is bit-identical to `xxhash_rust::xxh3::xxh3_64(&encoded_bytes)`
    /// by construction (both use seed 0 and the default xxh3 secret).
    pub hash: Option<u64>,
}

/// Build an szip compressor, dispatching between FFI and pure-Rust at runtime.
#[cfg(any(feature = "szip", feature = "szip-pure"))]
fn build_szip_compressor(
    #[allow(unused_variables)] backend: CompressionBackend,
    rsi: u32,
    block_size: u32,
    flags: u32,
    bits_per_sample: u32,
) -> Box<dyn Compressor> {
    // When both features are compiled in, dispatch at runtime.
    #[cfg(all(feature = "szip", feature = "szip-pure"))]
    if matches!(backend, CompressionBackend::Pure) {
        return Box::new(SzipPureCompressor {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        });
    }

    // FFI path — used when szip feature is enabled and either:
    // (a) both features compiled but backend is Ffi, or
    // (b) only szip is compiled.
    #[cfg(feature = "szip")]
    {
        Box::new(SzipCompressor {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        })
    }

    // Pure-only path — used when only szip-pure is compiled (no FFI).
    #[cfg(all(feature = "szip-pure", not(feature = "szip")))]
    {
        Box::new(SzipPureCompressor {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        })
    }
}

/// Build a zstd compressor, dispatching between FFI and pure-Rust at runtime.
///
/// `nb_workers` is forwarded to the FFI path (libzstd `NbWorkers`
/// parameter) and ignored by the pure-Rust path (ruzstd is
/// single-threaded).
#[cfg(any(feature = "zstd", feature = "zstd-pure"))]
fn build_zstd_compressor(
    #[allow(unused_variables)] backend: CompressionBackend,
    level: i32,
    #[allow(unused_variables)] nb_workers: u32,
) -> Box<dyn Compressor> {
    #[cfg(all(feature = "zstd", feature = "zstd-pure"))]
    if matches!(backend, CompressionBackend::Pure) {
        return Box::new(ZstdPureCompressor { level });
    }

    #[cfg(feature = "zstd")]
    {
        Box::new(ZstdCompressor { level, nb_workers })
    }

    #[cfg(all(feature = "zstd-pure", not(feature = "zstd")))]
    {
        Box::new(ZstdPureCompressor { level })
    }
}

/// Build a boxed compressor from a CompressionType variant.
///
/// For szip and zstd, the `compression_backend` field in `PipelineConfig`
/// selects between FFI and pure-Rust implementations at **runtime**.  When
/// only one feature is compiled in, the backend field is ignored.
fn build_compressor(
    compression: &CompressionType,
    #[allow(unused_variables)] config: &PipelineConfig,
) -> Result<Option<Box<dyn Compressor>>, CompressionError> {
    match compression {
        CompressionType::None => Ok(None),
        #[cfg(any(feature = "szip", feature = "szip-pure"))]
        CompressionType::Szip {
            rsi,
            block_size,
            flags,
            bits_per_sample,
        } => {
            let mut szip_flags = *flags;
            // simple_packing output is MSB-first; tell the szip codec so its
            // predictor sees bytes in the correct significance order.
            // AEC_DATA_MSB = 4 in both libaec-sys and tensogram-szip.
            if matches!(config.encoding, EncodingType::SimplePacking(_)) {
                szip_flags |= 4; // AEC_DATA_MSB
            }

            // Runtime backend selection.  The helper builds the right
            // Box<dyn Compressor> based on what features are compiled in
            // and what the caller requested.
            let compressor: Box<dyn Compressor> = build_szip_compressor(
                config.compression_backend,
                *rsi,
                *block_size,
                szip_flags,
                *bits_per_sample,
            );
            Ok(Some(compressor))
        }
        #[cfg(any(feature = "zstd", feature = "zstd-pure"))]
        CompressionType::Zstd { level } => {
            let compressor: Box<dyn Compressor> = build_zstd_compressor(
                config.compression_backend,
                *level,
                config.intra_codec_threads,
            );
            Ok(Some(compressor))
        }
        #[cfg(feature = "lz4")]
        CompressionType::Lz4 => Ok(Some(Box::new(Lz4Compressor))),
        #[cfg(feature = "blosc2")]
        CompressionType::Blosc2 {
            codec,
            clevel,
            typesize,
        } => Ok(Some(Box::new(Blosc2Compressor {
            codec: *codec,
            clevel: *clevel,
            typesize: *typesize,
            nthreads: config.intra_codec_threads,
        }))),
        #[cfg(feature = "zfp")]
        CompressionType::Zfp { mode } => Ok(Some(Box::new(ZfpCompressor {
            mode: mode.clone(),
            num_values: config.num_values,
            byte_order: config.byte_order,
        }))),
        #[cfg(feature = "sz3")]
        CompressionType::Sz3 { error_bound } => Ok(Some(Box::new(Sz3Compressor {
            error_bound: error_bound.clone(),
            num_values: config.num_values,
            byte_order: config.byte_order,
        }))),
        CompressionType::Rle => Ok(Some(Box::new(crate::compression::RleCompressor))),
        CompressionType::Roaring => Ok(Some(Box::new(crate::compression::RoaringCompressor))),
    }
}

/// Copy `src` into a freshly allocated `Vec<u8>` while updating `hasher` in
/// lockstep — one pass over the data.
///
/// When `hasher` is `None` this is a plain `src.to_vec()`.  When `hasher` is
/// `Some`, the function walks `src` in 64 KiB chunks: each chunk is first
/// fed to the hasher (bringing the bytes into L1/L2) and then appended to
/// the destination `Vec` (still cache-hot).  This avoids the second DRAM
/// read that `src.to_vec()` followed by `xxh3_64(&dst)` would incur on
/// buffers larger than L3.
///
/// The chunk size is a power-of-two that comfortably fits inside a typical
/// L2 cache while amortising the per-chunk call overhead of
/// `Xxh3Default::update`.
#[inline]
fn copy_and_hash(src: &[u8], hasher: Option<&mut xxhash_rust::xxh3::Xxh3Default>) -> Vec<u8> {
    match hasher {
        None => src.to_vec(),
        Some(h) => {
            const CHUNK: usize = 64 * 1024;
            let mut dst = Vec::with_capacity(src.len());
            let mut offset = 0;
            while offset < src.len() {
                let end = (offset + CHUNK).min(src.len());
                let chunk = &src[offset..end];
                h.update(chunk);
                dst.extend_from_slice(chunk);
                offset = end;
            }
            dst
        }
    }
}

/// Feed `bytes` to an optional hasher.  Used at each codec exit point in
/// `encode_pipeline` — the codec just wrote those bytes, so they are
/// maximally cache-hot; the hasher reads them from L2/L3 rather than DRAM.
#[inline]
fn update_hasher(bytes: &[u8], hasher: Option<&mut xxhash_rust::xxh3::Xxh3Default>) {
    if let Some(h) = hasher {
        h.update(bytes);
    }
}

/// Full forward pipeline: encode → filter → compress.
///
/// When `config.compute_hash` is `true`, the xxh3-64 digest of the final
/// encoded bytes is produced inline with the codec output (no second pass
/// over the buffer) and returned via `PipelineResult.hash`.  The digest is
/// bit-identical to what `xxhash_rust::xxh3::xxh3_64(&encoded_bytes)` would
/// return, by construction — both use seed 0 and the default secret.
///
/// Hashing runs entirely in the calling thread *after* any intra-codec
/// parallelism has joined.  The hasher is never shared across threads.
#[tracing::instrument(skip(data, config), fields(data_len = data.len(), encoding = %config.encoding))]
pub fn encode_pipeline(
    data: &[u8],
    config: &PipelineConfig,
) -> Result<PipelineResult, PipelineError> {
    let mut hasher = config
        .compute_hash
        .then(xxhash_rust::xxh3::Xxh3Default::new);

    // Step 1: Encoding — Cow avoids cloning when encoding is None
    let encoded: Cow<'_, [u8]> = match &config.encoding {
        EncodingType::None => Cow::Borrowed(data),
        EncodingType::SimplePacking(params) => {
            let values = bytes_to_f64(data, config.byte_order)?;
            Cow::Owned(simple_packing::encode_with_threads(
                &values,
                params,
                config.intra_codec_threads,
            )?)
        }
    };

    // Step 2: Filter
    let filtered: Cow<'_, [u8]> = match &config.filter {
        FilterType::None => encoded,
        FilterType::Shuffle { element_size } => Cow::Owned(
            shuffle::shuffle_with_threads(&encoded, *element_size, config.intra_codec_threads)
                .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
        ),
    };

    // Step 3: Compression
    let compressor = build_compressor(&config.compression, config)?;
    let (encoded_bytes, block_offsets) = match compressor {
        None => {
            // No compression: if `filtered` is still borrowed from `data`
            // (passthrough pipeline) we fuse the copy with hashing to
            // avoid a second walk over the source.  Otherwise it is
            // already owned and we just hash the Vec in place.
            let owned = match filtered {
                Cow::Borrowed(src) => copy_and_hash(src, hasher.as_mut()),
                Cow::Owned(buf) => {
                    update_hasher(&buf, hasher.as_mut());
                    buf
                }
            };
            (owned, None)
        }
        Some(compressor) => {
            let CompressResult {
                data: compressed,
                block_offsets,
            } = compressor.compress(&filtered)?;
            update_hasher(&compressed, hasher.as_mut());
            (compressed, block_offsets)
        }
    };

    Ok(PipelineResult {
        encoded_bytes,
        block_offsets,
        hash: hasher.map(|h| h.digest()),
    })
}

/// Encode from f64 values directly, avoiding the bytes→f64 conversion overhead
/// that `encode_pipeline` pays when the caller already has typed values.
///
/// Hash handling is identical to [`encode_pipeline`] — see that function's
/// documentation for the `compute_hash` contract.
#[tracing::instrument(skip(values, config), fields(num_values = values.len(), encoding = %config.encoding))]
pub fn encode_pipeline_f64(
    values: &[f64],
    config: &PipelineConfig,
) -> Result<PipelineResult, PipelineError> {
    let mut hasher = config
        .compute_hash
        .then(xxhash_rust::xxh3::Xxh3Default::new);

    let encoded: Cow<'_, [u8]> = match &config.encoding {
        EncodingType::None => Cow::Owned(f64_to_bytes(values, config.byte_order)?),
        EncodingType::SimplePacking(params) => Cow::Owned(simple_packing::encode_with_threads(
            values,
            params,
            config.intra_codec_threads,
        )?),
    };

    let filtered: Cow<'_, [u8]> = match &config.filter {
        FilterType::None => encoded,
        FilterType::Shuffle { element_size } => Cow::Owned(
            shuffle::shuffle_with_threads(&encoded, *element_size, config.intra_codec_threads)
                .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
        ),
    };

    let compressor = build_compressor(&config.compression, config)?;
    let (encoded_bytes, block_offsets) = match compressor {
        None => {
            // `encoded` is always owned in this function (both branches of
            // the match above construct a Cow::Owned), so `into_owned` is
            // a zero-cost unwrap.
            let owned = filtered.into_owned();
            update_hasher(&owned, hasher.as_mut());
            (owned, None)
        }
        Some(compressor) => {
            let CompressResult {
                data: compressed,
                block_offsets,
            } = compressor.compress(&filtered)?;
            update_hasher(&compressed, hasher.as_mut());
            (compressed, block_offsets)
        }
    };

    Ok(PipelineResult {
        encoded_bytes,
        block_offsets,
        hash: hasher.map(|h| h.digest()),
    })
}

/// Full reverse pipeline: decompress → unshuffle → decode → native byteswap.
///
/// When `native_byte_order` is true (the default at the API level), the
/// output bytes are converted to the caller's native byte order so that a
/// simple `reinterpret_cast` or `from_ne_bytes` produces correct values.
/// When false, bytes are returned in the message's declared wire byte order.
#[tracing::instrument(skip(encoded, config), fields(encoded_len = encoded.len()))]
pub fn decode_pipeline(
    encoded: &[u8],
    config: &PipelineConfig,
    native_byte_order: bool,
) -> Result<Vec<u8>, PipelineError> {
    // Step 1: Decompress — Cow avoids cloning when no compression
    let decompressed: Cow<'_, [u8]> = match build_compressor(&config.compression, config)? {
        None => Cow::Borrowed(encoded),
        Some(compressor) => {
            let expected_size = estimate_decompressed_size(config);
            Cow::Owned(compressor.decompress(encoded, expected_size)?)
        }
    };

    // Step 2: Unshuffle
    let unfiltered: Cow<'_, [u8]> = match &config.filter {
        FilterType::None => decompressed,
        FilterType::Shuffle { element_size } => Cow::Owned(
            shuffle::unshuffle_with_threads(
                &decompressed,
                *element_size,
                config.intra_codec_threads,
            )
            .map_err(|e| PipelineError::Shuffle(e.to_string()))?,
        ),
    };

    // Determine the target byte order for the output.  When the caller
    // requests native byte order, simple_packing can write directly in
    // native (avoiding a redundant write + swap).
    let target_byte_order = if native_byte_order {
        ByteOrder::native()
    } else {
        config.byte_order
    };

    // Step 3: Decode encoding
    let mut decoded = match &config.encoding {
        EncodingType::None => unfiltered.into_owned(),
        EncodingType::SimplePacking(params) => {
            // simple_packing decodes to Vec<f64> in-register values (no byte
            // order) then serialises directly to the target byte order.
            let values = simple_packing::decode_with_threads(
                &unfiltered,
                config.num_values,
                params,
                config.intra_codec_threads,
            )?;
            f64_to_bytes(&values, target_byte_order)?
        }
    };

    // Step 4: Native-endian byteswap for encoding=none.
    // (simple_packing already wrote in target_byte_order above.)
    if native_byte_order
        && matches!(config.encoding, EncodingType::None)
        && config.byte_order != ByteOrder::native()
    {
        byteswap(&mut decoded, config.swap_unit_size)?;
    }

    Ok(decoded)
}

/// Decode a partial sample range from a compressed+encoded pipeline.
///
/// Supports compressors with random access (szip, blosc2, zfp fixed-rate).
/// Shuffle filter is not supported with range decode.
///
/// `sample_offset` and `sample_count` are in logical element units.
/// `block_offsets` are block boundary offsets from encoding (compressor-specific).
///
/// When `native_byte_order` is true, the output bytes are in the caller's
/// native byte order.
pub fn decode_range_pipeline(
    encoded: &[u8],
    config: &PipelineConfig,
    block_offsets: &[u64],
    sample_offset: u64,
    sample_count: u64,
    native_byte_order: bool,
) -> Result<Vec<u8>, PipelineError> {
    if matches!(config.filter, FilterType::Shuffle { .. }) {
        return Err(PipelineError::Shuffle(
            "partial range decode is not supported with shuffle filter".to_string(),
        ));
    }

    // Phase 1: Compute byte range needed from the (possibly compressed) stream
    let (byte_start, byte_size, bit_offset_in_chunk) = match &config.encoding {
        EncodingType::SimplePacking(params) => {
            // Promote to u128 for the bit-position arithmetic so a hostile
            // `sample_offset` or `sample_count` cannot silently wrap.
            let bpv = params.bits_per_value as u128;
            let bit_start_u128 = (sample_offset as u128)
                .checked_mul(bpv)
                .ok_or_else(|| PipelineError::Range("bit start overflow".to_string()))?;
            let bit_count_u128 = (sample_count as u128)
                .checked_mul(bpv)
                .ok_or_else(|| PipelineError::Range("bit count overflow".to_string()))?;
            let bit_end_u128 = bit_start_u128
                .checked_add(bit_count_u128)
                .ok_or_else(|| PipelineError::Range("bit end overflow".to_string()))?;
            let bs = usize::try_from(bit_start_u128 / 8)
                .map_err(|_| PipelineError::Range("byte start exceeds usize".to_string()))?;
            let be = usize::try_from(bit_end_u128.div_ceil(8))
                .map_err(|_| PipelineError::Range("byte end exceeds usize".to_string()))?;
            let size = be
                .checked_sub(bs)
                .ok_or_else(|| PipelineError::Range("byte range invariant violated".to_string()))?;
            (bs, size, Some((bit_start_u128 % 8) as usize))
        }
        EncodingType::None => {
            let elem_size = config.dtype_byte_width;
            // `usize::try_from` rather than `as usize` so a `u64`
            // sample offset/count truncating on 32-bit surfaces as a
            // typed error instead of slicing into bogus memory.
            let offset_usize = usize::try_from(sample_offset).map_err(|_| {
                PipelineError::Range(format!(
                    "sample_offset {sample_offset} exceeds usize on this target"
                ))
            })?;
            let count_usize = usize::try_from(sample_count).map_err(|_| {
                PipelineError::Range(format!(
                    "sample_count {sample_count} exceeds usize on this target"
                ))
            })?;
            let bs = offset_usize
                .checked_mul(elem_size)
                .ok_or_else(|| PipelineError::Range("byte offset overflow".to_string()))?;
            let sz = count_usize
                .checked_mul(elem_size)
                .ok_or_else(|| PipelineError::Range("byte count overflow".to_string()))?;
            (bs, sz, None)
        }
    };

    // Phase 2: Get decompressed bytes for the range
    let decompressed = match build_compressor(&config.compression, config)? {
        None => {
            // No compression: slice directly from encoded buffer
            let byte_end = byte_start
                .checked_add(byte_size)
                .ok_or_else(|| PipelineError::Range("byte end overflow".to_string()))?;
            if byte_end > encoded.len() {
                return Err(PipelineError::Range(format!(
                    "range ({sample_offset}, {sample_count}) exceeds payload size"
                )));
            }
            try_clone_bytes(&encoded[byte_start..byte_end])?
        }
        Some(compressor) => {
            compressor.decompress_range(encoded, block_offsets, byte_start, byte_size)?
        }
    };

    let target_byte_order = if native_byte_order {
        ByteOrder::native()
    } else {
        config.byte_order
    };

    // Phase 3: Decode encoding from decompressed bytes
    match &config.encoding {
        EncodingType::None => {
            let mut result = decompressed;
            if native_byte_order && config.byte_order != ByteOrder::native() {
                byteswap(&mut result, config.swap_unit_size)?;
            }
            Ok(result)
        }
        EncodingType::SimplePacking(params) => {
            let count_usize = usize::try_from(sample_count).map_err(|_| {
                PipelineError::Range(format!(
                    "sample_count {sample_count} exceeds usize on this target"
                ))
            })?;
            let values = simple_packing::decode_range(
                &decompressed,
                bit_offset_in_chunk.unwrap_or(0),
                count_usize,
                params,
            )?;
            f64_to_bytes(&values, target_byte_order)
        }
    }
}

fn estimate_decompressed_size(config: &PipelineConfig) -> usize {
    match &config.encoding {
        EncodingType::None => {
            // `Dtype::Bitmask` reports `byte_width = 0` because its
            // elements are 1 bit each.  The decompressed byte count
            // for a bitmask payload is `ceil(num_values / 8)`.
            if config.dtype_byte_width == 0 {
                config.num_values.div_ceil(8)
            } else {
                config.num_values.saturating_mul(config.dtype_byte_width)
            }
        }
        EncodingType::SimplePacking(params) => {
            let total_bits =
                (config.num_values as u128).saturating_mul(params.bits_per_value as u128);
            total_bits.div_ceil(8).min(usize::MAX as u128) as usize
        }
    }
}

pub(crate) fn try_clone_bytes(src: &[u8]) -> Result<Vec<u8>, PipelineError> {
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(src.len()).map_err(|e| {
        PipelineError::Range(format!(
            "failed to reserve {} bytes for range output clone: {e}",
            src.len()
        ))
    })?;
    out.extend_from_slice(src);
    Ok(out)
}

fn bytes_to_f64(data: &[u8], byte_order: ByteOrder) -> Result<Vec<f64>, PipelineError> {
    // Reject partial trailing bytes rather than silently truncating —
    // `chunks_exact(8)` would otherwise drop them, which on the encode
    // side (simple_packing) would feed a truncated value stream into
    // the encoder without any error.
    if !data.len().is_multiple_of(8) {
        return Err(PipelineError::Range(format!(
            "byte-to-f64 input length {} is not a multiple of 8",
            data.len()
        )));
    }
    let n = data.len() / 8;
    let mut out: Vec<f64> = Vec::new();
    out.try_reserve_exact(n).map_err(|e| {
        PipelineError::Range(format!(
            "failed to reserve {} bytes for byte-to-f64 conversion: {e}",
            n.saturating_mul(std::mem::size_of::<f64>()),
        ))
    })?;
    for chunk in data.chunks_exact(8) {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        out.push(match byte_order {
            ByteOrder::Big => f64::from_be_bytes(arr),
            ByteOrder::Little => f64::from_le_bytes(arr),
        });
    }
    Ok(out)
}

fn f64_to_bytes(values: &[f64], byte_order: ByteOrder) -> Result<Vec<u8>, PipelineError> {
    let bytes_len = values.len().checked_mul(8).ok_or_else(|| {
        PipelineError::Range(format!(
            "f64-to-byte output length overflows usize: {} values x 8 bytes",
            values.len()
        ))
    })?;
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(bytes_len).map_err(|e| {
        PipelineError::Range(format!(
            "failed to reserve {bytes_len} bytes for f64-to-byte conversion: {e}"
        ))
    })?;
    for v in values {
        out.extend_from_slice(&match byte_order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_pipeline() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 1,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let result = encode_pipeline(&data, &config).unwrap();
        assert_eq!(result.encoded_bytes, data);
        let decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_simple_packing_pipeline() {
        let values: Vec<f64> = (0..50).map(|i| 200.0 + i as f64 * 0.1).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params = simple_packing::compute_params(&values, 16, 0).unwrap();

        let config = PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        let decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        let decoded_values = bytes_to_f64(&decoded, ByteOrder::Little).unwrap();

        for (orig, dec) in values.iter().zip(decoded_values.iter()) {
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_shuffle_pipeline() {
        let data: Vec<u8> = (0..16).collect();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::Shuffle { element_size: 4 },
            compression: CompressionType::None,
            num_values: 4,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 4,
            swap_unit_size: 4,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        assert_ne!(result.encoded_bytes, data); // shuffled should differ
        let decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        assert_eq!(decoded, data);
    }

    #[cfg(any(feature = "szip", feature = "szip-pure"))]
    #[test]
    fn test_szip_round_trip_pipeline() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();

        // AEC_DATA_PREPROCESS = 8 in both libaec-sys and tensogram-szip
        let preprocess_flag = 8u32;

        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Szip {
                rsi: 128,
                block_size: 16,
                flags: preprocess_flag,
                bits_per_sample: 8,
            },
            num_values: 2048,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 1,
            swap_unit_size: 1,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        assert!(result.block_offsets.is_some());

        let decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        assert_eq!(decoded, data);
    }

    // -----------------------------------------------------------------------
    // byteswap utility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_byteswap_noop_for_single_byte() {
        let mut data = vec![1, 2, 3, 4];
        let original = data.clone();
        byteswap(&mut data, 1).unwrap();
        assert_eq!(data, original);
        byteswap(&mut data, 0).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_byteswap_2_bytes() {
        let mut data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        byteswap(&mut data, 2).unwrap();
        assert_eq!(data, vec![0xBB, 0xAA, 0xDD, 0xCC]);
    }

    #[test]
    fn test_byteswap_4_bytes() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        byteswap(&mut data, 4).unwrap();
        assert_eq!(data, vec![4, 3, 2, 1, 8, 7, 6, 5]);
    }

    #[test]
    fn test_byteswap_8_bytes() {
        let mut data: Vec<u8> = (1..=16).collect();
        byteswap(&mut data, 8).unwrap();
        assert_eq!(
            data,
            vec![8, 7, 6, 5, 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9]
        );
    }

    #[test]
    fn test_byteswap_round_trip() {
        let original = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut data = original.clone();
        byteswap(&mut data, 4).unwrap();
        assert_ne!(data, original);
        byteswap(&mut data, 4).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_byteswap_misaligned_returns_error() {
        let mut data = vec![1, 2, 3, 4, 5]; // 5 bytes, not a multiple of 4
        let result = byteswap(&mut data, 4);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Native byte-order decode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_native_byte_order_encoding_none() {
        // Encode as big-endian float32 on a (likely) little-endian machine.
        let value: f32 = 42.0;
        let be_bytes = value.to_be_bytes();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 1,
            byte_order: ByteOrder::Big,
            dtype_byte_width: 4,
            swap_unit_size: 4,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };

        let result = encode_pipeline(&be_bytes, &config).unwrap();

        // Decode with native_byte_order=true: should get native-endian bytes.
        let native_decoded = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        let ne_value = f32::from_ne_bytes(native_decoded[..4].try_into().unwrap());
        assert_eq!(ne_value, value);

        // Decode with native_byte_order=false: should get big-endian bytes.
        let wire_decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        let be_value = f32::from_be_bytes(wire_decoded[..4].try_into().unwrap());
        assert_eq!(be_value, value);
    }

    #[test]
    fn test_decode_native_byte_order_simple_packing() {
        let values: Vec<f64> = vec![100.0, 200.0, 300.0, 400.0];
        // Encode with big-endian byte order.
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
        let params = simple_packing::compute_params(&values, 24, 0).unwrap();

        let config = PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::Big,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };

        let result = encode_pipeline(&data, &config).unwrap();

        // Decode with native_byte_order=true: result should be native f64.
        let native_decoded = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        let decoded_values: Vec<f64> = native_decoded
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        for (orig, dec) in values.iter().zip(decoded_values.iter()) {
            assert!((orig - dec).abs() < 1.0, "orig={orig}, dec={dec}");
        }

        // Decode with native_byte_order=false: result should be big-endian f64.
        let wire_decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        let wire_values: Vec<f64> = wire_decoded
            .chunks_exact(8)
            .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
            .collect();
        for (orig, dec) in values.iter().zip(wire_values.iter()) {
            assert!((orig - dec).abs() < 1.0, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn test_native_byte_order_same_as_wire_is_noop() {
        // When wire byte order == native, native_byte_order=true/false should
        // produce identical output (no swap needed either way).
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::native(),
            dtype_byte_width: 4,
            swap_unit_size: 4,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };

        let result = encode_pipeline(&data, &config).unwrap();
        let native_decoded = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        let wire_decoded = decode_pipeline(&result.encoded_bytes, &config, false).unwrap();
        assert_eq!(native_decoded, wire_decoded);
    }

    #[test]
    fn test_decode_native_byte_order_2byte_dtype() {
        // int16 / uint16 / float16 — 2-byte swap unit.
        let value: u16 = 0x0102;
        let be_bytes = value.to_be_bytes();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 1,
            byte_order: ByteOrder::Big,
            dtype_byte_width: 2,
            swap_unit_size: 2,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let result = encode_pipeline(&be_bytes, &config).unwrap();
        let native = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        assert_eq!(u16::from_ne_bytes(native[..2].try_into().unwrap()), value);
    }

    #[test]
    fn test_decode_native_byte_order_8byte_dtype() {
        // float64 / int64 / uint64 — 8-byte swap unit.
        let value: f64 = std::f64::consts::E;
        let be_bytes = value.to_be_bytes();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 1,
            byte_order: ByteOrder::Big,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let result = encode_pipeline(&be_bytes, &config).unwrap();
        let native = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        assert_eq!(f64::from_ne_bytes(native[..8].try_into().unwrap()), value);
    }

    #[test]
    fn test_decode_native_byte_order_complex64() {
        // complex64 = two float32 — swap_unit_size=4, dtype_byte_width=8.
        // Each 4-byte component must be swapped independently.
        let real: f32 = 1.5;
        let imag: f32 = 2.5;
        let mut be_bytes = Vec::new();
        be_bytes.extend_from_slice(&real.to_be_bytes());
        be_bytes.extend_from_slice(&imag.to_be_bytes());
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 1,
            byte_order: ByteOrder::Big,
            dtype_byte_width: 8,
            swap_unit_size: 4, // complex64: swap each float32 component
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let result = encode_pipeline(&be_bytes, &config).unwrap();
        let native = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        let decoded_real = f32::from_ne_bytes(native[0..4].try_into().unwrap());
        let decoded_imag = f32::from_ne_bytes(native[4..8].try_into().unwrap());
        assert_eq!(decoded_real, real);
        assert_eq!(decoded_imag, imag);
    }

    #[test]
    fn test_decode_native_byte_order_uint8_noop() {
        // uint8 / int8 — swap_unit_size=1, byteswap should be a no-op.
        let data = vec![1u8, 2, 3, 4, 5];
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: 5,
            byte_order: ByteOrder::Big, // cross-endian, but 1-byte → no-op
            dtype_byte_width: 1,
            swap_unit_size: 1,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let result = encode_pipeline(&data, &config).unwrap();
        let native = decode_pipeline(&result.encoded_bytes, &config, true).unwrap();
        assert_eq!(native, data); // no swap for single-byte types
    }

    // -----------------------------------------------------------------------
    // Hash-while-encoding tests — guard the invariant that
    // PipelineResult.hash (when compute_hash = true) is byte-equivalent to
    // xxh3_64(encoded_bytes) computed post-hoc.
    // -----------------------------------------------------------------------

    /// Helper: minimal passthrough config with a flag for hash.
    fn passthrough_config(num_values: usize, compute_hash: bool) -> PipelineConfig {
        PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 1,
            swap_unit_size: 1,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash,
        }
    }

    #[test]
    fn streaming_and_oneshot_xxh3_agree() {
        // Regression guard against xxhash-rust API drift: our fused path
        // relies on `Xxh3Default::new().update(chunks).digest()` producing
        // bit-identical output to `xxh3_64(concat(chunks))`.  If this ever
        // diverges, the hash-while-encoding optimisation would silently
        // corrupt hash values.
        use xxhash_rust::xxh3::{Xxh3Default, xxh3_64};

        for size in [0usize, 1, 239, 240, 1024 * 1024 + 17] {
            let data: Vec<u8> = (0..size).map(|i| (i * 31 + 7) as u8).collect();

            // Full one-shot.
            let one_shot = xxh3_64(&data);

            // Streaming, 64 KiB chunks (matches copy_and_hash).
            let mut h = Xxh3Default::new();
            for chunk in data.chunks(64 * 1024) {
                h.update(chunk);
            }
            assert_eq!(h.digest(), one_shot, "streaming vs one-shot at size {size}");

            // Streaming, 1-byte chunks — worst case for internal buffering.
            let mut h = Xxh3Default::new();
            for chunk in data.chunks(1) {
                h.update(chunk);
            }
            assert_eq!(
                h.digest(),
                one_shot,
                "streaming 1-byte chunks vs one-shot at size {size}"
            );
        }
    }

    #[test]
    fn pipeline_hash_none_when_disabled() {
        let data: Vec<u8> = (0..64).collect();
        let config = passthrough_config(data.len(), /* compute_hash = */ false);
        let result = encode_pipeline(&data, &config).unwrap();
        assert!(
            result.hash.is_none(),
            "compute_hash = false must leave PipelineResult.hash = None"
        );
    }

    #[test]
    fn pipeline_hash_matches_post_hoc_for_passthrough() {
        use xxhash_rust::xxh3::xxh3_64;

        // Exercise the sizes that hit each branch of `copy_and_hash` — below
        // one chunk, exactly one chunk, and multiple chunks.
        for size in [0usize, 1, 64 * 1024 - 1, 64 * 1024, 64 * 1024 + 1, 250_000] {
            let data: Vec<u8> = (0..size).map(|i| (i as u32 ^ 0xA5A5A5A5) as u8).collect();
            let config = passthrough_config(size, true);
            let result = encode_pipeline(&data, &config).unwrap();
            let expected = xxh3_64(&result.encoded_bytes);
            assert_eq!(
                result.hash,
                Some(expected),
                "passthrough hash-while-encoding mismatch at size {size}"
            );
            assert_eq!(
                result.encoded_bytes, data,
                "passthrough must still produce identical bytes at size {size}"
            );
        }
    }

    #[test]
    fn pipeline_hash_matches_post_hoc_for_simple_packing() {
        use xxhash_rust::xxh3::xxh3_64;

        let values: Vec<f64> = (0..10_000).map(|i| 200.0 + i as f64 * 0.1).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params = simple_packing::compute_params(&values, 16, 0).unwrap();

        let config = PipelineConfig {
            encoding: EncodingType::SimplePacking(params),
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: true,
        };
        let result = encode_pipeline(&data, &config).unwrap();
        let expected = xxh3_64(&result.encoded_bytes);
        assert_eq!(result.hash, Some(expected));
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn pipeline_hash_matches_post_hoc_for_lz4() {
        use xxhash_rust::xxh3::xxh3_64;

        let data: Vec<u8> = (0..16_000).map(|i| (i % 257) as u8).collect();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Lz4,
            num_values: data.len(),
            byte_order: ByteOrder::Little,
            dtype_byte_width: 1,
            swap_unit_size: 1,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: true,
        };
        let result = encode_pipeline(&data, &config).unwrap();
        let expected = xxh3_64(&result.encoded_bytes);
        assert_eq!(result.hash, Some(expected));
    }

    #[test]
    fn pipeline_f64_hash_matches_post_hoc() {
        use xxhash_rust::xxh3::xxh3_64;

        let values: Vec<f64> = (0..1_000).map(|i| (i as f64).sqrt()).collect();
        let config = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: true,
        };
        let result = encode_pipeline_f64(&values, &config).unwrap();
        let expected = xxh3_64(&result.encoded_bytes);
        assert_eq!(result.hash, Some(expected));
    }

    #[test]
    fn pipeline_hash_byte_identical_across_threads_transparent() {
        // Transparent codec (simple_packing): hash at threads=0 must equal
        // hash at threads=N for every N.  Opaque codecs are checked in the
        // integration suite (they allow per-run byte differences).
        let values: Vec<f64> = (0..50_000)
            .map(|i| 280.0 + (i as f64 * 0.001).sin())
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let params = simple_packing::compute_params(&values, 24, 0).unwrap();

        let mut hashes = Vec::new();
        for threads in [0u32, 1, 2, 4] {
            let config = PipelineConfig {
                encoding: EncodingType::SimplePacking(params.clone()),
                filter: FilterType::None,
                compression: CompressionType::None,
                num_values: values.len(),
                byte_order: ByteOrder::Little,
                dtype_byte_width: 8,
                swap_unit_size: 8,
                compression_backend: CompressionBackend::default(),
                intra_codec_threads: threads,
                compute_hash: true,
            };
            let result = encode_pipeline(&data, &config).unwrap();
            hashes.push(result.hash);
        }
        assert!(
            hashes.windows(2).all(|w| w[0] == w[1]),
            "transparent simple_packing must produce byte-identical hashes across thread counts: {hashes:?}"
        );
    }

    // ── Preallocation-DoS hardening (end-to-end) ────────────────────────
    //
    // Integration tests for the cross-codec `expected_size` preallocation
    // hardening.  A malformed descriptor with a hostile `num_values` must
    // surface as a `PipelineError::Compression` through the normal error
    // channel, not as a process abort.  Covers both szip backends when
    // both features are compiled in.

    #[cfg(any(feature = "szip", feature = "szip-pure"))]
    fn szip_hostile_config(num_values: usize, backend: CompressionBackend) -> PipelineConfig {
        PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Szip {
                bits_per_sample: 8,
                block_size: 16,
                rsi: 64,
                flags: 0,
            },
            num_values,
            byte_order: ByteOrder::Little,
            dtype_byte_width: 1,
            swap_unit_size: 1,
            compression_backend: backend,
            intra_codec_threads: 0,
            compute_hash: false,
        }
    }

    #[cfg(any(feature = "szip", feature = "szip-pure"))]
    fn small_szip_payload(backend: CompressionBackend) -> Vec<u8> {
        // Encode 256 bytes honestly with the selected backend so we have
        // a real szip-compressed payload the decode pipeline will accept
        // up to the reservation step.
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let honest = PipelineConfig {
            num_values: data.len(),
            ..szip_hostile_config(data.len(), backend)
        };
        encode_pipeline(&data, &honest).unwrap().encoded_bytes
    }

    #[cfg(feature = "szip")]
    #[test]
    fn pipeline_rejects_malicious_num_values_szip_ffi() {
        let payload = small_szip_payload(CompressionBackend::Ffi);
        let hostile = szip_hostile_config(usize::MAX, CompressionBackend::Ffi);

        let err = decode_pipeline(&payload, &hostile, false)
            .expect_err("expected allocation failure, not success nor abort");
        let msg = format!("{err}");
        assert!(
            msg.contains("failed to reserve"),
            "error should report allocation failure, got: {msg}"
        );
    }

    #[cfg(feature = "szip-pure")]
    #[test]
    fn pipeline_rejects_malicious_num_values_szip_pure() {
        let payload = small_szip_payload(CompressionBackend::Pure);
        let hostile = szip_hostile_config(usize::MAX, CompressionBackend::Pure);

        let err = decode_pipeline(&payload, &hostile, false)
            .expect_err("expected allocation failure, not success nor abort");
        let msg = format!("{err}");
        assert!(
            msg.contains("failed to reserve"),
            "error should report allocation failure, got: {msg}"
        );
    }

    #[test]
    fn pipeline_rejects_malicious_num_values_simple_packing() {
        // simple_packing with no compressor is the easiest descriptor-
        // driven abort path after the szip hardening: a malformed shape
        // takes `num_values` straight into simple_packing::decode_with_threads.
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sp_params = simple_packing::compute_params(&values, 16, 0).unwrap();

        let honest = PipelineConfig {
            encoding: EncodingType::SimplePacking(sp_params.clone()),
            filter: FilterType::None,
            compression: CompressionType::None,
            num_values: values.len(),
            byte_order: ByteOrder::Little,
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let payload = encode_pipeline(&data, &honest).unwrap().encoded_bytes;

        let hostile = PipelineConfig {
            num_values: usize::MAX,
            ..honest
        };
        let err = decode_pipeline(&payload, &hostile, false)
            .expect_err("pathological num_values on simple_packing must surface as an error");
        let msg = format!("{err}");
        assert!(
            msg.contains("overflow")
                || msg.contains("failed to reserve")
                || msg.contains("Insufficient")
                || msg.contains("insufficient"),
            "error should report a guard-check failure, got: {msg}"
        );
    }

    #[cfg(feature = "zfp")]
    #[test]
    fn pipeline_rejects_malicious_num_values_zfp() {
        let values: Vec<f64> = (0..64).map(|i| (i as f64) * 0.1).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

        let honest = PipelineConfig {
            encoding: EncodingType::None,
            filter: FilterType::None,
            compression: CompressionType::Zfp {
                mode: ZfpMode::FixedRate { rate: 16.0 },
            },
            num_values: values.len(),
            byte_order: ByteOrder::native(),
            dtype_byte_width: 8,
            swap_unit_size: 8,
            compression_backend: CompressionBackend::default(),
            intra_codec_threads: 0,
            compute_hash: false,
        };
        let payload = encode_pipeline(&data, &honest).unwrap().encoded_bytes;

        let hostile = PipelineConfig {
            num_values: usize::MAX,
            ..honest
        };
        let err = decode_pipeline(&payload, &hostile, false)
            .expect_err("pathological num_values on zfp must surface as an error");
        let msg = format!("{err}");
        assert!(
            msg.contains("failed to reserve"),
            "error should report allocation failure, got: {msg}"
        );
    }
}

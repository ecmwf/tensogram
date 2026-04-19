// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use crate::encode::build_pipeline_config_with_backend;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::hash;
use crate::types::{DataObjectDescriptor, DecodedObject, GlobalMetadata};
use tensogram_encodings::pipeline;

fn extract_block_offsets(
    params: &std::collections::BTreeMap<String, ciborium::Value>,
) -> Result<Vec<u64>> {
    match params.get("szip_block_offsets") {
        Some(ciborium::Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                ciborium::Value::Integer(i) => {
                    let n: i128 = (*i).into();
                    u64::try_from(n).map_err(|_| {
                        TensogramError::Metadata("szip_block_offset out of u64 range".to_string())
                    })
                }
                _ => Err(TensogramError::Metadata(
                    "szip_block_offsets must contain integers".to_string(),
                )),
            })
            .collect(),
        Some(_) => Err(TensogramError::Metadata(
            "szip_block_offsets must be an array".to_string(),
        )),
        None => Err(TensogramError::Compression(
            "missing szip_block_offsets in payload metadata (required for partial range decode)"
                .to_string(),
        )),
    }
}

/// Options for decoding.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// Whether to verify payload hashes during decode.
    pub verify_hash: bool,
    /// When true (the default), decoded payloads are converted to the
    /// caller's native byte order regardless of the wire byte order declared
    /// in the descriptor.  Set to false to receive bytes in the message's
    /// declared wire byte order (rare — useful for zero-copy forwarding).
    pub native_byte_order: bool,
    /// Which backend to use for szip / zstd when both FFI and pure-Rust
    /// implementations are compiled in.
    pub compression_backend: pipeline::CompressionBackend,
    /// Thread budget for the multi-threaded decoding pipeline.
    ///
    /// Semantics match
    /// [`EncodeOptions.threads`](crate::encode::EncodeOptions::threads):
    /// `0` means sequential (may be overridden by `TENSOGRAM_THREADS`),
    /// `1` means explicit single-threaded execution, `N ≥ 2` builds a
    /// scoped pool.  Output bytes are byte-identical to the
    /// sequential path regardless of `N`.
    pub threads: u32,
    /// Minimum total payload bytes below which the parallel path is
    /// skipped.  See
    /// [`EncodeOptions.parallel_threshold_bytes`](crate::encode::EncodeOptions::parallel_threshold_bytes).
    pub parallel_threshold_bytes: Option<usize>,
    /// When `true` (the default) AND the object carries a
    /// `NTensorMaskedFrame` `masks` sub-map, decompress the masks
    /// and write the canonical NaN / +Inf / -Inf bit pattern at
    /// every `1` position in the decoded output.  See
    /// `plans/BITMASK_FRAME.md` §7.1 for the (lossy) reconstruction
    /// caveat — only the canonical quiet-NaN / ±∞ bit pattern is
    /// restored; specific NaN payloads are not preserved.
    ///
    /// Set to `false` to skip restoration and receive the
    /// `0.0`-substituted bytes as they are on disk.  Callers who
    /// need the raw masks alongside the substituted payload use
    /// [`decode_with_masks`] instead.
    pub restore_non_finite: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            verify_hash: false,
            native_byte_order: true,
            compression_backend: pipeline::CompressionBackend::default(),
            threads: 0,
            parallel_threshold_bytes: None,
            restore_non_finite: true,
        }
    }
}

/// Decode all objects from a message buffer.
/// Returns (global_metadata, list of (descriptor, decoded_data)).
///
/// When `options.threads > 0` (or `TENSOGRAM_THREADS` is set),
/// per-object decode work is parallelised using the axis-B-first
/// policy documented in
/// `docs/src/guide/multi-threaded-pipeline.md`.  Output bytes are
/// byte-identical to the sequential path regardless of thread count.
#[tracing::instrument(skip(buf, options), fields(buf_len = buf.len()))]
pub fn decode(buf: &[u8], options: &DecodeOptions) -> Result<(GlobalMetadata, Vec<DecodedObject>)> {
    let msg = framing::decode_message(buf)?;

    let budget = crate::parallel::resolve_budget(options.threads);
    let total_bytes: usize = msg.objects.iter().map(|(_, p, _, _)| p.len()).sum();
    let parallel =
        crate::parallel::should_parallelise(budget, total_bytes, options.parallel_threshold_bytes);
    let any_axis_b = msg.objects.iter().any(|(d, _, _, _)| {
        crate::parallel::is_axis_b_friendly(&d.encoding, &d.filter, &d.compression)
    });
    let use_axis_a = parallel && crate::parallel::use_axis_a(msg.objects.len(), budget, any_axis_b);
    let intra_codec_threads = if parallel && !use_axis_a { budget } else { 0 };

    let decode_one = |(desc, payload_bytes, mask_region, _offset): &(
        DataObjectDescriptor,
        &[u8],
        &[u8],
        usize,
    )|
     -> Result<DecodedObject> {
        let mut decoded = decode_single_object_with_backend(
            desc,
            payload_bytes,
            options,
            options.compression_backend,
            intra_codec_threads,
        )?;
        if options.restore_non_finite {
            crate::restore::restore_non_finite_into(
                &mut decoded,
                desc,
                mask_region,
                output_byte_order(desc, options),
            )?;
        }
        Ok((desc.clone(), decoded))
    };

    let data_objects: Vec<DecodedObject> = if use_axis_a {
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            crate::parallel::with_pool(budget, || {
                msg.objects
                    .par_iter()
                    .map(&decode_one)
                    .collect::<Result<Vec<_>>>()
            })?
        }
        #[cfg(not(feature = "threads"))]
        {
            msg.objects.iter().map(decode_one).collect::<Result<_>>()?
        }
    } else {
        crate::parallel::run_maybe_pooled(budget, parallel, intra_codec_threads, || {
            msg.objects.iter().map(decode_one).collect::<Result<_>>()
        })?
    };

    Ok((msg.global_metadata, data_objects))
}

/// Decode only global metadata from a message buffer, skipping payloads.
pub fn decode_metadata(buf: &[u8]) -> Result<GlobalMetadata> {
    framing::decode_metadata_only(buf)
}

/// Decode all objects from a message buffer AND return the raw
/// decompressed bitmasks alongside the substituted payloads.
///
/// Like [`decode`], but the returned payloads always contain `0.0`
/// at non-finite positions — restoration is **not** applied.
/// Callers get the raw [`restore::DecodedMaskSet`] for each object
/// and can apply the masks manually (e.g. to convert to a
/// domain-specific missing-value representation, or to aggregate
/// missing-count statistics without materialising the canonical
/// NaN / Inf bytes).
///
/// See `plans/BITMASK_FRAME.md` §7.3.
pub fn decode_with_masks(
    buf: &[u8],
    options: &DecodeOptions,
) -> Result<(GlobalMetadata, Vec<DecodedObjectWithMasks>)> {
    let msg = framing::decode_message(buf)?;

    let budget = crate::parallel::resolve_budget(options.threads);
    let total_bytes: usize = msg.objects.iter().map(|(_, p, _, _)| p.len()).sum();
    let parallel =
        crate::parallel::should_parallelise(budget, total_bytes, options.parallel_threshold_bytes);
    let intra_codec_threads = if parallel { budget } else { 0 };

    // Local options snapshot with restore_non_finite forced off —
    // this API returns masks alongside a 0-substituted payload by
    // design, matching the `plans/BITMASK_FRAME.md` §7.3 contract.
    let mut decode_opts = options.clone();
    decode_opts.restore_non_finite = false;

    let decode_one = |(desc, payload_bytes, mask_region, _offset): &(
        DataObjectDescriptor,
        &[u8],
        &[u8],
        usize,
    )|
     -> Result<DecodedObjectWithMasks> {
        let payload = decode_single_object_with_backend(
            desc,
            payload_bytes,
            &decode_opts,
            options.compression_backend,
            intra_codec_threads,
        )?;
        let masks = crate::restore::decode_mask_set(desc, mask_region)?;
        Ok(DecodedObjectWithMasks {
            descriptor: desc.clone(),
            payload,
            masks,
        })
    };

    // Advanced API — axis-A parallelism is not worth the complexity
    // here since this path is intended for niche workflows (custom
    // missing-value representations, missing-count stats).  Mainline
    // consumers use `decode()` with `restore_non_finite=true` which
    // does have the full axis-A/B dispatch.
    let objects: Vec<DecodedObjectWithMasks> =
        crate::parallel::run_maybe_pooled(budget, parallel, intra_codec_threads, || {
            msg.objects.iter().map(decode_one).collect::<Result<_>>()
        })?;

    Ok((msg.global_metadata, objects))
}

pub use crate::restore::{DecodedMaskSet, DecodedObjectWithMasks};

/// Decode global metadata **and** per-object descriptors without decoding
/// any payload data.
///
/// This is cheaper than [`decode`] because the pipeline (decompression,
/// filter reversal, endian swap) is never executed.  Use it when you only
/// need shapes, dtypes, and metadata — e.g. for building xarray Datasets
/// at open time.
pub fn decode_descriptors(buf: &[u8]) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
    let msg = framing::decode_message(buf)?;
    let descriptors = msg
        .objects
        .into_iter()
        .map(|(desc, _, _, _)| desc)
        .collect();
    Ok((msg.global_metadata, descriptors))
}

/// Decode a single object by index (O(1) access via index frame).
/// Returns (global_metadata, descriptor, decoded_data).
pub fn decode_object(
    buf: &[u8],
    index: usize,
    options: &DecodeOptions,
) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
    let msg = framing::decode_message(buf)?;

    if index >= msg.objects.len() {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            index,
            msg.objects.len()
        )));
    }

    let (desc, payload_bytes, mask_region, _) = &msg.objects[index];

    // Single-object decode: axis A is impossible — spend the entire
    // budget (if any) on the codec internally (axis B).
    let budget = crate::parallel::resolve_budget(options.threads);
    let parallel = crate::parallel::should_parallelise(
        budget,
        payload_bytes.len(),
        options.parallel_threshold_bytes,
    );
    let intra_codec_threads = if parallel { budget } else { 0 };

    let mut decoded =
        crate::parallel::run_maybe_pooled(budget, parallel, intra_codec_threads, || {
            decode_single_object_with_backend(
                desc,
                payload_bytes,
                options,
                options.compression_backend,
                intra_codec_threads,
            )
        })?;

    if options.restore_non_finite {
        crate::restore::restore_non_finite_into(
            &mut decoded,
            desc,
            mask_region,
            output_byte_order(desc, options),
        )?;
    }

    Ok((msg.global_metadata, desc.clone(), decoded))
}

/// Decode partial ranges from a data object.
///
/// `ranges` is a list of (element_offset, element_count) pairs.
///
/// Returns `(descriptor, parts)` where `parts` contains one `Vec<u8>`
/// per range.  The descriptor is included so callers can determine
/// the dtype without a separate lookup.
pub fn decode_range(
    buf: &[u8],
    object_index: usize,
    ranges: &[(u64, u64)],
    options: &DecodeOptions,
) -> Result<(DataObjectDescriptor, Vec<Vec<u8>>)> {
    let msg = framing::decode_message(buf)?;

    if object_index >= msg.objects.len() {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            object_index,
            msg.objects.len()
        )));
    }

    let (desc, payload_bytes, mask_region, _) = &msg.objects[object_index];
    let mut parts = decode_range_from_payload(desc, payload_bytes, ranges, options)?;
    // Apply mask-aware NaN / Inf restoration to each requested
    // sub-range.  See `plans/BITMASK_FRAME.md` §7.4.
    if options.restore_non_finite && desc.masks.is_some() {
        let mask_set = crate::restore::decode_mask_set(desc, mask_region)?;
        crate::restore::restore_non_finite_into_ranges(
            &mut parts,
            desc,
            ranges,
            &mask_set,
            output_byte_order(desc, options),
        )?;
    }
    Ok((desc.clone(), parts))
}

/// Byte order of the bytes coming out of `decode_pipeline` for
/// `desc`, given the caller's [`DecodeOptions`].  Used to tell
/// [`crate::restore`] which endianness to write canonical NaN / Inf
/// bit patterns in.
fn output_byte_order(
    desc: &DataObjectDescriptor,
    options: &DecodeOptions,
) -> tensogram_encodings::ByteOrder {
    if options.native_byte_order {
        tensogram_encodings::ByteOrder::native()
    } else {
        desc.byte_order
    }
}

pub fn decode_range_from_payload(
    desc: &DataObjectDescriptor,
    payload_bytes: &[u8],
    ranges: &[(u64, u64)],
    options: &DecodeOptions,
) -> Result<Vec<Vec<u8>>> {
    if desc.filter != "none" {
        return Err(TensogramError::Encoding(
            "decode_range is not supported when a filter (e.g. shuffle) is applied".to_string(),
        ));
    }

    if desc.dtype.byte_width() == 0 {
        return Err(TensogramError::Encoding(
            "partial range decode not supported for bitmask dtype".to_string(),
        ));
    }

    if options.verify_hash
        && let Some(ref hash_desc) = desc.hash
    {
        hash::verify_hash(payload_bytes, hash_desc)?;
    }

    let shape_product = desc
        .shape
        .iter()
        .try_fold(1u64, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
    let num_elements = usize::try_from(shape_product)
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    // Thread-budget dispatch for range decode.
    //
    // Each range is an independent decode call; parallelism is natural
    // when the caller requests multiple ranges.  Axis B is always
    // preferred when there's only one range.
    let budget = crate::parallel::resolve_budget(options.threads);
    // Work is proportional to decoded output, not the input payload —
    // sum the requested counts × element byte width.
    let elem_bytes = desc.dtype.byte_width();
    let total_bytes: usize = ranges
        .iter()
        .map(|(_, c)| (*c as usize).saturating_mul(elem_bytes))
        .sum();
    let parallel =
        crate::parallel::should_parallelise(budget, total_bytes, options.parallel_threshold_bytes);
    let axis_b_friendly =
        crate::parallel::is_axis_b_friendly(&desc.encoding, &desc.filter, &desc.compression);
    let use_axis_a = parallel && crate::parallel::use_axis_a(ranges.len(), budget, axis_b_friendly);
    let intra_codec_threads = if parallel && !use_axis_a { budget } else { 0 };

    let config = build_pipeline_config_with_backend(
        desc,
        num_elements,
        desc.dtype,
        options.compression_backend,
        intra_codec_threads,
    )?;

    let block_offsets = if desc.compression == "szip" {
        extract_block_offsets(&desc.params)?
    } else {
        Vec::new()
    };

    let decode_one = |offset: u64, count: u64| -> Result<Vec<u8>> {
        pipeline::decode_range_pipeline(
            payload_bytes,
            &config,
            &block_offsets,
            offset,
            count,
            options.native_byte_order,
        )
        .map_err(|e| {
            TensogramError::Encoding(format!("range (offset={offset}, count={count}): {e}"))
        })
    };

    let run_seq = || -> Result<Vec<Vec<u8>>> {
        ranges
            .iter()
            .map(|&(offset, count)| decode_one(offset, count))
            .collect()
    };

    let results: Vec<Vec<u8>> = if use_axis_a {
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            crate::parallel::with_pool(budget, || {
                ranges
                    .par_iter()
                    .map(|&(offset, count)| decode_one(offset, count))
                    .collect::<Result<Vec<_>>>()
            })?
        }
        #[cfg(not(feature = "threads"))]
        {
            run_seq()?
        }
    } else {
        crate::parallel::run_maybe_pooled(budget, parallel, intra_codec_threads, run_seq)?
    };

    Ok(results)
}

#[cfg(feature = "remote")]
pub(crate) fn decode_single_object(
    desc: &DataObjectDescriptor,
    payload_bytes: &[u8],
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    decode_single_object_with_backend(desc, payload_bytes, options, options.compression_backend, 0)
}

/// Decode a single object payload using the specified compression backend
/// and intra-codec thread budget.
///
/// `intra_codec_threads == 0` preserves the pre-threads behaviour.
fn decode_single_object_with_backend(
    desc: &DataObjectDescriptor,
    payload_bytes: &[u8],
    options: &DecodeOptions,
    backend: pipeline::CompressionBackend,
    intra_codec_threads: u32,
) -> Result<Vec<u8>> {
    if options.verify_hash
        && let Some(ref hash_desc) = desc.hash
    {
        hash::verify_hash(payload_bytes, hash_desc)?;
    }

    let shape_product = desc
        .shape
        .iter()
        .try_fold(1u64, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
    let num_elements = usize::try_from(shape_product)
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    let config = build_pipeline_config_with_backend(
        desc,
        num_elements,
        desc.dtype,
        backend,
        intra_codec_threads,
    )?;
    let decoded = pipeline::decode_pipeline(payload_bytes, &config, options.native_byte_order)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::encode::{EncodeOptions, encode};
    use crate::types::ByteOrder;
    use std::collections::BTreeMap;

    fn make_global_meta() -> GlobalMetadata {
        GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
            ..Default::default()
        }
    }

    fn make_descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
        let strides = if shape.is_empty() {
            vec![]
        } else {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape,
            strides,
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
            hash: None,
        }
    }

    // ── corrupt descriptor CBOR → decode error ───────────────────────────

    #[test]
    fn test_decode_corrupt_message_bytes() {
        // Completely invalid bytes — not a valid tensogram message
        let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
        let result = decode(&garbage, &DecodeOptions::default());
        assert!(result.is_err(), "decoding garbage should fail");
    }

    #[test]
    fn test_decode_truncated_message() {
        // Encode a valid message then truncate it
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        // Truncate to half
        let truncated = &encoded[..encoded.len() / 2];
        let result = decode(truncated, &DecodeOptions::default());
        assert!(result.is_err(), "decoding truncated message should fail");
    }

    #[test]
    fn test_decode_corrupted_cbor_in_message() {
        // Encode a valid message then corrupt the metadata frame CBOR.
        // The metadata CBOR starts right after preamble (24 bytes) +
        // frame header (16 bytes). Aggressively corrupt that region.
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        let mut encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        // Preamble = 24 bytes, Frame header = 16 bytes => CBOR starts at 40
        let cbor_start = 40;
        let corrupt_end = (cbor_start + 30).min(encoded.len());
        for byte in &mut encoded[cbor_start..corrupt_end] {
            *byte = 0xFF;
        }

        let result = decode(&encoded, &DecodeOptions::default());
        // Should fail because CBOR metadata or frame structure is corrupted
        assert!(result.is_err(), "decoding corrupted CBOR should fail");
    }

    // ── object index out of range in decode_object ───────────────────────

    #[test]
    fn test_decode_object_index_out_of_range() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        // Only 1 object (index 0), request index 1
        let result = decode_object(&encoded, 1, &DecodeOptions::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "expected 'out of range', got: {msg}"
        );

        // Request a very large index
        let result = decode_object(&encoded, 999, &DecodeOptions::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of range"));
    }

    #[test]
    fn test_decode_object_valid_index() {
        let meta = make_global_meta();
        let desc0 = make_descriptor(vec![2]);
        let data0 = vec![10u8; 8];
        let desc1 = make_descriptor(vec![3]);
        let data1 = vec![20u8; 12];

        let encoded = encode(
            &meta,
            &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        // Access object 0
        let (_, ret_desc, ret_data) =
            decode_object(&encoded, 0, &DecodeOptions::default()).unwrap();
        assert_eq!(ret_desc.shape, vec![2]);
        assert_eq!(ret_data, data0);

        // Access object 1
        let (_, ret_desc, ret_data) =
            decode_object(&encoded, 1, &DecodeOptions::default()).unwrap();
        assert_eq!(ret_desc.shape, vec![3]);
        assert_eq!(ret_data, data1);
    }

    // ── decode_range invalid byte ranges ─────────────────────────────────

    #[test]
    fn test_decode_range_object_index_out_of_range() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        let result = decode_range(&encoded, 5, &[(0, 2)], &DecodeOptions::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "expected 'out of range', got: {msg}"
        );
    }

    #[test]
    fn test_decode_range_exceeds_payload() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]); // 4 float32s = 16 bytes
        let data = vec![0u8; 16];
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        // Request range offset=2, count=10 but only 4 elements
        let result = decode_range(&encoded, 0, &[(2, 10)], &DecodeOptions::default());
        assert!(result.is_err(), "range exceeding payload should fail");
    }

    #[test]
    fn test_decode_range_valid() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![8]); // 8 float32s = 32 bytes
        let data: Vec<u8> = (0..32).collect();
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        let (ret_desc, parts) =
            decode_range(&encoded, 0, &[(0, 4)], &DecodeOptions::default()).unwrap();
        assert_eq!(ret_desc.shape, vec![8]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 16); // 4 float32s = 16 bytes
    }

    #[test]
    fn test_decode_range_empty_ranges() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        let (_, parts) = decode_range(&encoded, 0, &[], &DecodeOptions::default()).unwrap();
        assert!(parts.is_empty());
    }

    // ── decode_metadata ──────────────────────────────────────────────────

    #[test]
    fn test_decode_metadata_valid() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        let decoded_meta = decode_metadata(&encoded).unwrap();
        assert_eq!(decoded_meta.version, 2);
    }

    #[test]
    fn test_decode_metadata_corrupt() {
        let garbage = vec![0xFF; 50];
        let result = decode_metadata(&garbage);
        assert!(result.is_err(), "decode_metadata on garbage should fail");
    }

    // ── decode_descriptors ───────────────────────────────────────────────

    #[test]
    fn test_decode_descriptors_valid() {
        let meta = make_global_meta();
        let desc0 = make_descriptor(vec![4]);
        let desc1 = make_descriptor(vec![2, 3]);
        let data0 = vec![0u8; 16];
        let data1 = vec![0u8; 24];
        let encoded = encode(
            &meta,
            &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        let (decoded_meta, descs) = decode_descriptors(&encoded).unwrap();
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].shape, vec![4]);
        assert_eq!(descs[1].shape, vec![2, 3]);
    }

    // ── decode_range with filter=shuffle → error ─────────────────────────

    #[test]
    fn test_decode_range_filter_shuffle_rejected() {
        let meta = make_global_meta();
        let mut desc = make_descriptor(vec![100]);
        desc.filter = "shuffle".to_string();
        desc.params.insert(
            "shuffle_element_size".to_string(),
            ciborium::Value::Integer(4.into()),
        );
        // Finite f32 data (avoid tripping the 0.17 default-reject
        // finite-check with byte-pattern NaN bits).
        let data: Vec<u8> = (0..100).flat_map(|i| (i as f32).to_ne_bytes()).collect();

        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        let result = decode_range(&encoded, 0, &[(0, 10)], &DecodeOptions::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("filter") || msg.contains("shuffle"),
            "expected filter/shuffle error, got: {msg}"
        );
    }

    // ── decode_range with bitmask dtype → error ──────────────────────────

    #[test]
    fn test_decode_range_bitmask_dtype_rejected() {
        let meta = make_global_meta();
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![16],
            strides: vec![1],
            dtype: Dtype::Bitmask,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
            hash: None,
        };
        let data = vec![0xFF; 2]; // ceil(16/8) = 2 bytes

        let encoded = encode(&meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();

        let result = decode_range(&encoded, 0, &[(0, 8)], &DecodeOptions::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("bitmask"),
            "expected bitmask error, got: {msg}"
        );
    }

    // ── DecodeOptions defaults ───────────────────────────────────────────

    #[test]
    fn test_decode_options_defaults() {
        let opts = DecodeOptions::default();
        assert!(!opts.verify_hash);
        assert!(opts.native_byte_order);
    }

    // ── decode with unknown encoding in descriptor ───────────────────────

    #[test]
    fn test_decode_unknown_encoding_in_descriptor() {
        // We need to craft a message with an unknown encoding.
        // Easiest: encode a valid message, then manually patch the CBOR
        // descriptor's encoding field. Instead, use build_pipeline_config directly.
        let mut desc = make_descriptor(vec![4]);
        desc.encoding = "foobar".to_string();

        let result = crate::encode::build_pipeline_config_with_backend(
            &desc,
            4,
            Dtype::Float32,
            pipeline::CompressionBackend::default(),
            0,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unknown encoding"),
            "expected 'unknown encoding', got: {msg}"
        );
    }

    // ── decode with unknown compression in descriptor ────────────────────

    #[test]
    fn test_decode_unknown_compression_in_descriptor() {
        let mut desc = make_descriptor(vec![4]);
        desc.compression = "quantum_compress".to_string();

        let result = crate::encode::build_pipeline_config_with_backend(
            &desc,
            4,
            Dtype::Float32,
            pipeline::CompressionBackend::default(),
            0,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unknown compression"),
            "expected 'unknown compression', got: {msg}"
        );
    }

    // ── extract_block_offsets error paths ─────────────────────────────────

    #[test]
    fn test_extract_block_offsets_missing() {
        let params = BTreeMap::new();
        let result = extract_block_offsets(&params);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("szip_block_offsets"),
            "expected szip_block_offsets error, got: {msg}"
        );
    }

    #[test]
    fn test_extract_block_offsets_wrong_type() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Text("not an array".to_string()),
        );
        let result = extract_block_offsets(&params);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must be an array"),
            "expected 'must be an array', got: {msg}"
        );
    }

    #[test]
    fn test_extract_block_offsets_non_integer_elements() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![
                ciborium::Value::Float(1.5), // not an integer
            ]),
        );
        let result = extract_block_offsets(&params);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("integers"),
            "expected integers error, got: {msg}"
        );
    }

    #[test]
    fn test_extract_block_offsets_valid() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![
                ciborium::Value::Integer(0.into()),
                ciborium::Value::Integer(100.into()),
                ciborium::Value::Integer(200.into()),
            ]),
        );
        let result = extract_block_offsets(&params).unwrap();
        assert_eq!(result, vec![0, 100, 200]);
    }
}

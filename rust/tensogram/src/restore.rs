// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Decode-side companion to [`crate::substitute_and_mask`].
//!
//! When a data-object frame carries a `masks` sub-map
//! (`NTensorMaskedFrame`, wire type 9 — see `plans/BITMASK_FRAME.md`
//! §7), this module:
//!
//! 1. Parses each present mask's [`crate::types::MaskDescriptor`]
//!    from the descriptor.
//! 2. Decompresses the mask blob via the method named in the
//!    descriptor (`rle`, `roaring`, `blosc2`, `zstd`, `lz4`, or
//!    `none`).
//! 3. Writes the canonical bit pattern for the kind
//!    (`f64::NAN` / `f64::INFINITY` / `f64::NEG_INFINITY`; and the
//!    dtype-equivalent patterns for `f16` / `bf16` / `f32` / `c64` /
//!    `c128`) at every `1` position in the already-decoded payload.
//!
//! ## Lossy reconstruction
//!
//! The restored bits are **canonical** — bit-exact NaN payloads and
//! mixed-component complex kinds are not preserved.  This is a
//! documented trade-off; see `plans/BITMASK_FRAME.md` §7.1.

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use crate::types::{DataObjectDescriptor, MaskDescriptor};
use tensogram_encodings::ByteOrder;
use tensogram_encodings::bitmask;

/// If `descriptor.masks` is `Some`, decompress each mask from
/// `mask_region` and restore canonical non-finite bit patterns in
/// `decoded_payload`.  No-op when `descriptor.masks` is `None`.
///
/// Caller is responsible for already decoding `decoded_payload`
/// through the pipeline (decompression, filter reversal, byte-order
/// swap).  `decoded_payload.len()` must match the element count
/// declared by the descriptor's shape + dtype.
///
/// `output_byte_order` must match the byte order of
/// `decoded_payload` — typically [`ByteOrder::native()`] when the
/// caller decoded with `native_byte_order = true`, otherwise the
/// descriptor's declared `byte_order`.
pub(crate) fn restore_non_finite_into(
    decoded_payload: &mut [u8],
    descriptor: &DataObjectDescriptor,
    mask_region: &[u8],
    output_byte_order: ByteOrder,
) -> Result<()> {
    let Some(masks) = descriptor.masks.as_ref() else {
        return Ok(());
    };

    let n_elements = element_count(descriptor)?;
    let elem_size = descriptor.dtype.byte_width();
    if elem_size == 0 {
        // Bitmask dtype — masks don't apply; guard against the
        // pathological case.
        return Err(TensogramError::Framing(
            "bitmask-companion masks cannot be restored on bitmask-dtype payloads".to_string(),
        ));
    }
    let expected_len = n_elements.checked_mul(elem_size).ok_or_else(|| {
        TensogramError::Metadata("n_elements * elem_size overflows usize".to_string())
    })?;
    if decoded_payload.len() != expected_len {
        return Err(TensogramError::Framing(format!(
            "decoded payload length {} does not match descriptor n_elements * elem_size ({} * {} = {})",
            decoded_payload.len(),
            n_elements,
            elem_size,
            expected_len,
        )));
    }

    // Mask-region offsets in the descriptor are relative to the
    // payload-region start.  The `mask_region` slice we receive
    // starts at the smallest mask offset (= payload length), so
    // [`decode_one_mask_at`] subtracts `mask_region_base` to
    // translate descriptor offsets into per-slice positions.
    let mask_region_base = smallest_mask_offset(masks);

    // Restore in canonical descriptor order: nan, inf+, inf-.  A
    // given element is classified into AT MOST ONE kind on encode
    // (complex priority rule), so the three mask kinds never
    // overlap; iteration order is cosmetic.
    for (md, kind) in each_mask_kind(masks) {
        let bits = decode_one_mask_at(md, mask_region, mask_region_base, n_elements)?;
        write_canonical_non_finite(
            decoded_payload,
            descriptor.dtype,
            output_byte_order,
            &bits,
            kind,
        );
    }

    Ok(())
}

/// Iterate over the three mask kinds in canonical order, yielding
/// only those that are present.  Used by both [`restore_non_finite_into`]
/// and [`decode_mask_set`] to keep the kind-handling DRY.
fn each_mask_kind(
    masks: &crate::types::MasksMetadata,
) -> impl Iterator<Item = (&MaskDescriptor, Kind)> {
    [
        (masks.nan.as_ref(), Kind::Nan),
        (masks.pos_inf.as_ref(), Kind::PosInf),
        (masks.neg_inf.as_ref(), Kind::NegInf),
    ]
    .into_iter()
    .filter_map(|(md, kind)| md.map(|m| (m, kind)))
}

fn element_count(desc: &DataObjectDescriptor) -> Result<usize> {
    let product = desc
        .shape
        .iter()
        .try_fold(1u64, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
    usize::try_from(product)
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))
}

/// Decompress one mask blob at `mask_region[md.offset - mask_region_base ..
/// md.offset - mask_region_base + md.length]`.
///
/// Every mask's `offset` field in the descriptor is relative to the
/// payload-region start; `mask_region_base` is the offset of the
/// first byte of `mask_region` relative to the same anchor.  Framing
/// returns a `mask_region` that starts at the smallest mask offset,
/// so `base == smallest offset`.
fn decode_one_mask_at(
    md: &MaskDescriptor,
    mask_region: &[u8],
    mask_region_base: u64,
    n_elements: usize,
) -> Result<Vec<bool>> {
    let offset = u64_to_usize(md.offset, "mask.offset")?;
    let length = u64_to_usize(md.length, "mask.length")?;
    let base = u64_to_usize(mask_region_base, "mask_region_base")?;
    let start = offset.checked_sub(base).ok_or_else(|| {
        TensogramError::Framing(format!(
            "mask.offset {} less than mask_region_base {}",
            md.offset, mask_region_base
        ))
    })?;
    let end = start.checked_add(length).ok_or_else(|| {
        TensogramError::Framing(format!(
            "mask.offset + length overflow (offset={offset}, length={length})"
        ))
    })?;
    if end > mask_region.len() {
        return Err(TensogramError::Framing(format!(
            "mask slice end {end} exceeds mask_region length {}",
            mask_region.len()
        )));
    }
    let blob = &mask_region[start..end];
    decode_blob(&md.method, blob, n_elements).map_err(|e| match e {
        // `UnknownMethod` already names the method — avoid duplicating it.
        bitmask::MaskError::UnknownMethod(_) => TensogramError::Encoding(e.to_string()),
        other => TensogramError::Encoding(format!("bitmask decode ({}): {other}", md.method)),
    })
}

fn u64_to_usize(v: u64, name: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| TensogramError::Framing(format!("{name} {v} overflows usize")))
}

fn decode_blob(
    method: &str,
    blob: &[u8],
    n_elements: usize,
) -> std::result::Result<Vec<bool>, bitmask::MaskError> {
    match method {
        "none" => bitmask::codecs::decode_none(blob, n_elements),
        "rle" => bitmask::rle::decode(blob, n_elements),
        "roaring" => bitmask::roaring::decode(blob, n_elements),
        "lz4" => bitmask::codecs::decode_lz4(blob, n_elements),
        "zstd" => bitmask::codecs::decode_zstd(blob, n_elements),
        #[cfg(feature = "blosc2")]
        "blosc2" => bitmask::codecs::decode_blosc2(blob, n_elements),
        #[cfg(not(feature = "blosc2"))]
        "blosc2" => Err(bitmask::MaskError::FeatureDisabled { method: "blosc2" }),
        other => Err(bitmask::MaskError::UnknownMethod(other.to_string())),
    }
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Nan,
    PosInf,
    NegInf,
}

/// Write the canonical bit pattern for `kind` at every position
/// where `bits[i]` is `true`.  Byte order matches `byte_order`;
/// element size and dispatch are driven by `dtype`.
///
/// Callers must ensure `buf.len() >= bits.len() * dtype.byte_width()`
/// — [`restore_non_finite_into`] and [`restore_non_finite_into_ranges`]
/// both validate this at entry.  A `debug_assert!` catches a caller
/// bug in tests; release builds skip out-of-bounds positions rather
/// than panicking.
fn write_canonical_non_finite(
    buf: &mut [u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    bits: &[bool],
    kind: Kind,
) {
    let elem_size = dtype.byte_width();
    debug_assert!(
        buf.len() >= bits.len() * elem_size,
        "write_canonical_non_finite: buf {} < bits {} * elem_size {}",
        buf.len(),
        bits.len(),
        elem_size,
    );
    for (i, &is_set) in bits.iter().enumerate() {
        if !is_set {
            continue;
        }
        let start = i * elem_size;
        if start + elem_size > buf.len() {
            // Debug-assert caught this above; release builds take the
            // safe fallback (skip) so a bad caller can't crash the
            // process.  All in-tree callers validate lengths upstream.
            break;
        }
        match dtype {
            Dtype::Float32 => {
                let bytes = match kind {
                    Kind::Nan => f32_nan_bytes(byte_order),
                    Kind::PosInf => f32_bytes(f32::INFINITY, byte_order),
                    Kind::NegInf => f32_bytes(f32::NEG_INFINITY, byte_order),
                };
                buf[start..start + 4].copy_from_slice(&bytes);
            }
            Dtype::Float64 => {
                let bytes = match kind {
                    Kind::Nan => f64_nan_bytes(byte_order),
                    Kind::PosInf => f64_bytes(f64::INFINITY, byte_order),
                    Kind::NegInf => f64_bytes(f64::NEG_INFINITY, byte_order),
                };
                buf[start..start + 8].copy_from_slice(&bytes);
            }
            Dtype::Float16 => {
                // IEEE half canonical patterns:
                //   NaN  = 0x7E00 (quiet, sign=0, exp=0x1F, mant=0x200)
                //   +Inf = 0x7C00
                //   -Inf = 0xFC00
                let bits_u16 = match kind {
                    Kind::Nan => 0x7E00u16,
                    Kind::PosInf => 0x7C00,
                    Kind::NegInf => 0xFC00,
                };
                let bytes = match byte_order {
                    ByteOrder::Big => bits_u16.to_be_bytes(),
                    ByteOrder::Little => bits_u16.to_le_bytes(),
                };
                buf[start..start + 2].copy_from_slice(&bytes);
            }
            Dtype::Bfloat16 => {
                // bfloat16 canonical patterns:
                //   NaN  = 0x7FC0 (quiet, sign=0, exp=0xFF, mant=0x40)
                //   +Inf = 0x7F80
                //   -Inf = 0xFF80
                let bits_u16 = match kind {
                    Kind::Nan => 0x7FC0u16,
                    Kind::PosInf => 0x7F80,
                    Kind::NegInf => 0xFF80,
                };
                let bytes = match byte_order {
                    ByteOrder::Big => bits_u16.to_be_bytes(),
                    ByteOrder::Little => bits_u16.to_le_bytes(),
                };
                buf[start..start + 2].copy_from_slice(&bytes);
            }
            Dtype::Complex64 => {
                // Both real and imag components get the same
                // canonical pattern — see §4 / §7.1.
                let comp = match kind {
                    Kind::Nan => f32_nan_bytes(byte_order),
                    Kind::PosInf => f32_bytes(f32::INFINITY, byte_order),
                    Kind::NegInf => f32_bytes(f32::NEG_INFINITY, byte_order),
                };
                buf[start..start + 4].copy_from_slice(&comp);
                buf[start + 4..start + 8].copy_from_slice(&comp);
            }
            Dtype::Complex128 => {
                let comp = match kind {
                    Kind::Nan => f64_nan_bytes(byte_order),
                    Kind::PosInf => f64_bytes(f64::INFINITY, byte_order),
                    Kind::NegInf => f64_bytes(f64::NEG_INFINITY, byte_order),
                };
                buf[start..start + 8].copy_from_slice(&comp);
                buf[start + 8..start + 16].copy_from_slice(&comp);
            }
            // Non-float dtypes: the encoder should have never
            // produced masks for these, but guard defensively.
            _ => {}
        }
    }
}

fn f32_nan_bytes(byte_order: ByteOrder) -> [u8; 4] {
    // Canonical quiet NaN: 0x7FC00000.  Matches `f32::NAN.to_bits()`
    // on every host Rust supports today, but we use the explicit
    // constant rather than `f32::NAN` to remove host variance.
    let bits = 0x7FC0_0000u32;
    match byte_order {
        ByteOrder::Big => bits.to_be_bytes(),
        ByteOrder::Little => bits.to_le_bytes(),
    }
}

fn f64_nan_bytes(byte_order: ByteOrder) -> [u8; 8] {
    // Canonical quiet NaN: 0x7FF8000000000000.
    let bits = 0x7FF8_0000_0000_0000u64;
    match byte_order {
        ByteOrder::Big => bits.to_be_bytes(),
        ByteOrder::Little => bits.to_le_bytes(),
    }
}

fn f32_bytes(v: f32, byte_order: ByteOrder) -> [u8; 4] {
    match byte_order {
        ByteOrder::Big => v.to_be_bytes(),
        ByteOrder::Little => v.to_le_bytes(),
    }
}

fn f64_bytes(v: f64, byte_order: ByteOrder) -> [u8; 8] {
    match byte_order {
        ByteOrder::Big => v.to_be_bytes(),
        ByteOrder::Little => v.to_le_bytes(),
    }
}

// ── decode_with_masks API — exposes raw MaskSet to advanced callers ────────

/// A decoded object paired with its raw NaN / Inf bitmasks.
///
/// Unlike the standard [`crate::decode`] path, the `payload` here is
/// always the `0.0`-substituted decoded bytes — masks are **not**
/// applied.  Advanced callers can apply them manually, aggregate
/// across kinds, or convert to their own domain types.  See
/// `plans/BITMASK_FRAME.md` §7.3.
#[derive(Debug, Clone)]
pub struct DecodedObjectWithMasks {
    /// Object descriptor, including the `masks` sub-map when
    /// present (a no-op when the frame carried no masks).
    pub descriptor: DataObjectDescriptor,
    /// Decoded payload with non-finite positions holding `0.0`.
    pub payload: Vec<u8>,
    /// Decompressed per-kind bitmasks.  `Vec<bool>` of length
    /// `n_elements`.  Missing kinds have `None`.
    pub masks: DecodedMaskSet,
}

/// Three-kind set of decompressed bitmasks, one entry per mask kind
/// present in the frame.  Each [`Vec<bool>`] has length
/// `n_elements` (the descriptor's shape product).  Mirrors the
/// encoder's [`crate::substitute_and_mask::MaskSet`] but carries no
/// element-count field — callers get it from the descriptor.
#[derive(Debug, Clone, Default)]
pub struct DecodedMaskSet {
    pub nan: Option<Vec<bool>>,
    pub pos_inf: Option<Vec<bool>>,
    pub neg_inf: Option<Vec<bool>>,
}

impl DecodedMaskSet {
    /// `true` when every kind is absent.
    pub fn is_empty(&self) -> bool {
        self.nan.is_none() && self.pos_inf.is_none() && self.neg_inf.is_none()
    }

    /// Mutable slot for one of the three mask kinds, used by
    /// assembly helpers that iterate over [`Kind`] values.
    fn slot_for(&mut self, kind: Kind) -> &mut Option<Vec<bool>> {
        match kind {
            Kind::Nan => &mut self.nan,
            Kind::PosInf => &mut self.pos_inf,
            Kind::NegInf => &mut self.neg_inf,
        }
    }
}

/// Apply canonical NaN / Inf restoration to pre-decoded range slices.
///
/// Each range is a `(element_offset, element_count)` pair matching the
/// `ranges` argument passed to [`crate::decode_range`].  The output
/// `parts[i]` is a `Vec<u8>` of length `count_i * elem_size` — we
/// write the canonical bit pattern at any element index that falls in
/// `[offset_i, offset_i + count_i)` AND has a `1` bit in the
/// corresponding kind's mask.
///
/// No-op when `descriptor.masks` is `None` or `mask_set.is_empty()`.
/// `output_byte_order` must match the byte order of the bytes
/// already in `parts`; see [`restore_non_finite_into`].
pub(crate) fn restore_non_finite_into_ranges(
    parts: &mut [Vec<u8>],
    descriptor: &DataObjectDescriptor,
    ranges: &[(u64, u64)],
    mask_set: &DecodedMaskSet,
    output_byte_order: ByteOrder,
) -> Result<()> {
    if descriptor.masks.is_none() || mask_set.is_empty() {
        return Ok(());
    }
    if parts.len() != ranges.len() {
        return Err(TensogramError::Framing(format!(
            "range count mismatch: parts.len()={} but ranges.len()={}",
            parts.len(),
            ranges.len()
        )));
    }

    let elem_size = descriptor.dtype.byte_width();
    if elem_size == 0 {
        return Err(TensogramError::Framing(
            "bitmask-companion masks cannot be restored on bitmask-dtype payloads".to_string(),
        ));
    }

    // For every (range, part) pair, slice each mask over the range
    // and apply canonical restoration per kind.  Kinds never overlap
    // on encode (complex priority rule) so the three passes are
    // independent.
    for (part, &(offset, count)) in parts.iter_mut().zip(ranges.iter()) {
        let start = u64_to_usize(offset, "range.offset")?;
        let count = u64_to_usize(count, "range.count")?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| TensogramError::Framing("range offset+count overflow".to_string()))?;
        let expected_part_len = count.checked_mul(elem_size).ok_or_else(|| {
            TensogramError::Framing("range count * elem_size overflows usize".to_string())
        })?;
        if part.len() != expected_part_len {
            return Err(TensogramError::Framing(format!(
                "range part length {} does not match count * elem_size ({} * {} = {})",
                part.len(),
                count,
                elem_size,
                expected_part_len,
            )));
        }
        for (kind_bits, kind) in [
            (mask_set.nan.as_ref(), Kind::Nan),
            (mask_set.pos_inf.as_ref(), Kind::PosInf),
            (mask_set.neg_inf.as_ref(), Kind::NegInf),
        ] {
            let Some(bits) = kind_bits else { continue };
            if end > bits.len() {
                return Err(TensogramError::Framing(format!(
                    "range end {end} exceeds mask length {} for descriptor shape",
                    bits.len()
                )));
            }
            let sliced = &bits[start..end];
            write_canonical_non_finite(part, descriptor.dtype, output_byte_order, sliced, kind);
        }
    }
    Ok(())
}

/// Decompress the masks referenced by `descriptor.masks` from the
/// raw `mask_region` slice.  Returns `DecodedMaskSet::default()` when
/// the descriptor has no masks.
pub(crate) fn decode_mask_set(
    descriptor: &DataObjectDescriptor,
    mask_region: &[u8],
) -> Result<DecodedMaskSet> {
    let Some(masks) = descriptor.masks.as_ref() else {
        return Ok(DecodedMaskSet::default());
    };
    let n_elements = element_count(descriptor)?;
    let mask_region_base = smallest_mask_offset(masks);
    let mut out = DecodedMaskSet::default();
    for (md, kind) in each_mask_kind(masks) {
        let bits = decode_one_mask_at(md, mask_region, mask_region_base, n_elements)?;
        *out.slot_for(kind) = Some(bits);
    }
    Ok(out)
}

fn smallest_mask_offset(masks: &crate::types::MasksMetadata) -> u64 {
    let mut smallest = u64::MAX;
    for md in [
        masks.nan.as_ref(),
        masks.pos_inf.as_ref(),
        masks.neg_inf.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if md.offset < smallest {
            smallest = md.offset;
        }
    }
    smallest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MaskDescriptor, MasksMetadata};
    use std::collections::BTreeMap;

    fn make_descriptor(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape: shape.clone(),
            strides: {
                let mut s = vec![1u64; shape.len()];
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    s[i] = s[i + 1] * shape[i + 1];
                }
                s
            },
            dtype,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            masks: None,
            params: BTreeMap::new(),
            hash: None,
        }
    }

    #[test]
    fn write_canonical_f64_nan_at_marked_positions() {
        let mut buf = vec![0u8; 8 * 4];
        let bits = vec![false, true, false, true];
        let dtype = Dtype::Float64;
        write_canonical_non_finite(&mut buf, dtype, ByteOrder::native(), &bits, Kind::Nan);

        // Element 1 and 3 must decode as NaN; 0 and 2 as 0.0.
        let mut doubles = [0f64; 4];
        for i in 0..4 {
            let bytes = [
                buf[i * 8],
                buf[i * 8 + 1],
                buf[i * 8 + 2],
                buf[i * 8 + 3],
                buf[i * 8 + 4],
                buf[i * 8 + 5],
                buf[i * 8 + 6],
                buf[i * 8 + 7],
            ];
            doubles[i] = f64::from_ne_bytes(bytes);
        }
        assert_eq!(doubles[0], 0.0);
        assert!(doubles[1].is_nan());
        assert_eq!(doubles[2], 0.0);
        assert!(doubles[3].is_nan());
    }

    #[test]
    fn write_canonical_f32_neg_inf() {
        let mut buf = vec![0u8; 4 * 3];
        let bits = vec![false, true, false];
        write_canonical_non_finite(
            &mut buf,
            Dtype::Float32,
            ByteOrder::native(),
            &bits,
            Kind::NegInf,
        );

        let v = f32::from_ne_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert!(v.is_infinite() && v.is_sign_negative());
    }

    #[test]
    fn write_canonical_complex64_writes_both_components() {
        let mut buf = vec![0u8; 8];
        let bits = vec![true];
        write_canonical_non_finite(
            &mut buf,
            Dtype::Complex64,
            ByteOrder::native(),
            &bits,
            Kind::Nan,
        );

        let real = f32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let imag = f32::from_ne_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert!(real.is_nan());
        assert!(imag.is_nan());
    }

    #[test]
    fn restore_non_finite_into_no_op_without_masks() {
        let desc = make_descriptor(vec![4], Dtype::Float64);
        let mut payload = vec![0u8; 32];
        let original = payload.clone();
        restore_non_finite_into(&mut payload, &desc, &[], ByteOrder::native()).unwrap();
        assert_eq!(payload, original);
    }

    #[test]
    fn decode_mask_set_empty_when_no_masks() {
        let desc = make_descriptor(vec![4], Dtype::Float64);
        let set = decode_mask_set(&desc, &[]).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn restore_non_finite_into_rejects_bitmask_dtype() {
        // Regression: bitmask-dtype descriptors must reject mask
        // companions at entry — no silent zero-width processing.
        let mut desc = make_descriptor(vec![4], Dtype::Bitmask);
        desc.masks = Some(MasksMetadata {
            nan: Some(MaskDescriptor {
                method: "none".to_string(),
                offset: 0,
                length: 1,
                params: BTreeMap::new(),
            }),
            ..Default::default()
        });
        let mut payload = vec![0u8; 1];
        let err = restore_non_finite_into(&mut payload, &desc, &[0u8; 1], ByteOrder::native())
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bitmask"), "got: {msg}");
    }

    #[test]
    fn restore_non_finite_into_rejects_wrong_payload_length() {
        // Regression for Pass 3 E2: catch the caller-bug silently
        // skipped before — decoded payload shorter than
        // n_elements * elem_size now hard-errors.
        let mut desc = make_descriptor(vec![4], Dtype::Float64);
        desc.masks = Some(MasksMetadata {
            nan: Some(MaskDescriptor {
                method: "none".to_string(),
                offset: 32, // bytes for 4×f64
                length: 1,
                params: BTreeMap::new(),
            }),
            ..Default::default()
        });
        let mut short_payload = vec![0u8; 16]; // only 2×f64 — wrong
        let mask_region: Vec<u8> = bitmask::codecs::encode_none(&[false, true, false, true]);
        let err =
            restore_non_finite_into(&mut short_payload, &desc, &mask_region, ByteOrder::native())
                .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("decoded payload length"),
            "expected length-mismatch error, got: {msg}"
        );
    }

    #[test]
    fn restore_non_finite_into_ranges_rejects_wrong_part_length() {
        // Regression for Pass 3 E3: decode_range consumers that
        // supply malformed per-range slices now see a clear error
        // rather than silent corruption.
        let mut desc = make_descriptor(vec![8], Dtype::Float64);
        desc.masks = Some(MasksMetadata {
            nan: Some(MaskDescriptor {
                method: "none".to_string(),
                offset: 64,
                length: 1,
                params: BTreeMap::new(),
            }),
            ..Default::default()
        });
        let mask_set = DecodedMaskSet {
            nan: Some(vec![true; 8]),
            pos_inf: None,
            neg_inf: None,
        };
        // Wrong part length: range says count=4 but part holds 3×f64.
        let mut parts = vec![vec![0u8; 24]];
        let err = restore_non_finite_into_ranges(
            &mut parts,
            &desc,
            &[(0, 4)],
            &mask_set,
            ByteOrder::native(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("range part length"), "got: {err}");
    }

    #[test]
    fn decode_one_mask_at_unknown_method_error_is_not_duplicated() {
        // Regression for Pass 3 E4: the "bitmask decode (method):"
        // prefix must not duplicate the method name inside
        // `UnknownMethod`'s own message.
        let md = MaskDescriptor {
            method: "bogus".to_string(),
            offset: 0,
            length: 1,
            params: BTreeMap::new(),
        };
        let mask_region = vec![0u8; 1];
        let err = decode_one_mask_at(&md, &mask_region, 0, 4).unwrap_err();
        let msg = err.to_string();
        // The method name appears exactly once.
        let occurrences = msg.matches("bogus").count();
        assert_eq!(
            occurrences, 1,
            "message should name method once, got: {msg}"
        );
    }

    #[test]
    fn decode_mask_set_restores_offsets_relative_to_region_base() {
        // Build an artificial frame: 4 f64 elements, nan mask at
        // offset 32 (= payload length) of 1 byte, roaring-encoded
        // mask bits [true, false, true, false].
        let mut masks = MasksMetadata::default();
        // Encode a roaring mask for [true, false, true, false]
        let bits = vec![true, false, true, false];
        let blob = bitmask::roaring::encode(&bits).unwrap();
        masks.nan = Some(MaskDescriptor {
            method: "roaring".to_string(),
            offset: 32, // payload_len
            length: blob.len() as u64,
            params: BTreeMap::new(),
        });
        let mut desc = make_descriptor(vec![4], Dtype::Float64);
        desc.masks = Some(masks);

        // mask_region slice IS just the blob (in framing.rs we give
        // the caller a slice starting at the smallest mask offset).
        let got = decode_mask_set(&desc, &blob).unwrap();
        assert_eq!(got.nan.unwrap(), bits);
    }
}

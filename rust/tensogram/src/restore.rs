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
pub(crate) fn restore_non_finite_into(
    decoded_payload: &mut [u8],
    descriptor: &DataObjectDescriptor,
    mask_region: &[u8],
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
    // `decoded_payload` is in the caller's declared byte order when
    // options.native_byte_order is true.  We write canonical bit
    // patterns in that same order so the caller sees consistent
    // bytes back.  `options.native_byte_order == false` (rare) keeps
    // the wire byte order; we respect the descriptor's declared
    // byte_order in that case — but since the decode output is
    // whichever order the caller requested, we just always write
    // native here.  The "always native" choice mirrors how
    // `decode_pipeline` handles endian on its output.
    let native = ByteOrder::native();

    // Mask-region offsets in the descriptor are relative to the
    // payload-region start.  The `mask_region` slice we receive
    // starts at the smallest mask offset — which equals the payload
    // length — so subtract that base to translate into per-slice
    // positions.  `decode_one_mask_at` handles the translation.
    let mask_region_base = smallest_mask_offset(masks);

    // Restore in canonical descriptor order: nan, inf+, inf-.  Since
    // a given element was classified into exactly ONE kind on encode
    // (complex priority rule), the three mask kinds never overlap;
    // restoration order is cosmetic.
    if let Some(md) = masks.nan.as_ref() {
        let bits = decode_one_mask_at(md, mask_region, mask_region_base, n_elements)?;
        write_canonical_non_finite(decoded_payload, descriptor.dtype, native, &bits, Kind::Nan);
    }
    if let Some(md) = masks.pos_inf.as_ref() {
        let bits = decode_one_mask_at(md, mask_region, mask_region_base, n_elements)?;
        write_canonical_non_finite(
            decoded_payload,
            descriptor.dtype,
            native,
            &bits,
            Kind::PosInf,
        );
    }
    if let Some(md) = masks.neg_inf.as_ref() {
        let bits = decode_one_mask_at(md, mask_region, mask_region_base, n_elements)?;
        write_canonical_non_finite(
            decoded_payload,
            descriptor.dtype,
            native,
            &bits,
            Kind::NegInf,
        );
    }

    Ok(())
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
    decode_blob(&md.method, blob, n_elements)
        .map_err(|e| TensogramError::Encoding(format!("bitmask decode ({}): {e}", md.method)))
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
fn write_canonical_non_finite(
    buf: &mut [u8],
    dtype: Dtype,
    byte_order: ByteOrder,
    bits: &[bool],
    kind: Kind,
) {
    let elem_size = dtype.byte_width();
    // Each present mask's length equals `n_elements`; `buf.len() ==
    // n_elements * elem_size` for non-bitmask float dtypes.  Bitmask
    // dtype is rejected upstream.
    for (i, &is_set) in bits.iter().enumerate() {
        if !is_set {
            continue;
        }
        let start = i * elem_size;
        if start + elem_size > buf.len() {
            // Defensive: a malformed mask that claims more elements
            // than the payload can hold.  Skip rather than panic so
            // a bad frame doesn't crash the process.
            return;
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
pub(crate) fn restore_non_finite_into_ranges(
    parts: &mut [Vec<u8>],
    descriptor: &DataObjectDescriptor,
    ranges: &[(u64, u64)],
    mask_set: &DecodedMaskSet,
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
    let native = ByteOrder::native();

    // For every (range, part) pair, slice each mask over the range
    // and apply canonical restoration per kind.  A single element is
    // only ever set in at most one mask (complex priority rule on
    // encode) so the three passes cannot collide.
    for (part, &(offset, count)) in parts.iter_mut().zip(ranges.iter()) {
        let start = u64_to_usize(offset, "range.offset")?;
        let end = start
            .checked_add(u64_to_usize(count, "range.count")?)
            .ok_or_else(|| TensogramError::Framing("range offset+count overflow".to_string()))?;
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
            write_canonical_non_finite(part, descriptor.dtype, native, sliced, kind);
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
    if let Some(md) = masks.nan.as_ref() {
        out.nan = Some(decode_one_mask_at(
            md,
            mask_region,
            mask_region_base,
            n_elements,
        )?);
    }
    if let Some(md) = masks.pos_inf.as_ref() {
        out.pos_inf = Some(decode_one_mask_at(
            md,
            mask_region,
            mask_region_base,
            n_elements,
        )?);
    }
    if let Some(md) = masks.neg_inf.as_ref() {
        out.neg_inf = Some(decode_one_mask_at(
            md,
            mask_region,
            mask_region_base,
            n_elements,
        )?);
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
        restore_non_finite_into(&mut payload, &desc, &[]).unwrap();
        assert_eq!(payload, original);
    }

    #[test]
    fn decode_mask_set_empty_when_no_masks() {
        let desc = make_descriptor(vec![4], Dtype::Float64);
        let set = decode_mask_set(&desc, &[]).unwrap();
        assert!(set.is_empty());
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

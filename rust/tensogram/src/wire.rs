// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use crate::error::{Result, TensogramError};

// ── Constants ────────────────────────────────────────────────────────────────

/// Message start magic: ASCII "TENSOGRM"
pub const MAGIC: &[u8; 8] = b"TENSOGRM";
/// Message end magic: ASCII "39277777"
pub const END_MAGIC: &[u8; 8] = b"39277777";
/// Frame start marker: ASCII "FR"
pub const FRAME_MAGIC: &[u8; 2] = b"FR";
/// Frame end marker: ASCII "ENDF"
pub const FRAME_END: &[u8; 4] = b"ENDF";

/// Current wire-format version.  See `plans/WIRE_FORMAT.md`.
///
/// A v3 decoder rejects any preamble whose `version` field does not
/// match this constant.  Bumping the wire version is the single
/// source of truth for backwards incompatibility — every structural
/// format change must bump this.
pub const WIRE_VERSION: u16 = 3;

/// Preamble size: magic(8) + version(2) + flags(2) + reserved(4) + total_length(8) = 24
pub const PREAMBLE_SIZE: usize = 24;
/// Frame header size: FR(2) + type(2) + version(2) + flags(2) + total_length(8) = 16
pub const FRAME_HEADER_SIZE: usize = 16;
/// Postamble size in v3: first_footer_offset(8) + total_length(8) + end_magic(8) = 24.
///
/// The `total_length` field was added in v3 to make the postamble
/// self-locating from any byte position inside a message — see
/// `plans/WIRE_FORMAT.md` §7 and §9.2.
pub const POSTAMBLE_SIZE: usize = 24;
/// Data object footer size: cbor_offset(8) + ENDF(4) = 12
pub const DATA_OBJECT_FOOTER_SIZE: usize = 12;

// ── Frame Types ──────────────────────────────────────────────────────────────

/// Frame type identifiers (uint16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum FrameType {
    HeaderMetadata = 1,
    HeaderIndex = 2,
    HeaderHash = 3,
    /// Legacy n-tensor data-object frame (pre-0.17).  Read-only for
    /// ≥0.17 decoders; new encoders emit [`FrameType::NTensorMaskedFrame`].
    ///
    /// The on-wire layout and CBOR schema are unchanged from pre-0.17.
    /// When a type-9 decoder reads a type-4 frame, the descriptor is
    /// interpreted as having no `masks` sub-map (equivalent to "all
    /// values are finite" under the new default-reject semantics).
    NTensorFrame = 4,
    FooterHash = 5,
    FooterIndex = 6,
    FooterMetadata = 7,
    /// Per-object metadata frame that immediately precedes a data-object
    /// frame (type 4 or 9).  Carries a GlobalMetadata CBOR with a
    /// single-entry `base` array containing metadata for the next data
    /// object. `_reserved_` and `_extra_` are empty in the preceder.
    PrecederMetadata = 8,
    /// N-tensor data-object frame with optional companion bitmasks
    /// identifying positions of NaN / +Inf / -Inf values in the
    /// original input.
    ///
    /// Layout: same preamble as type 4, then the encoded payload (with
    /// non-finite values substituted with `0.0`), then up to three
    /// compressed bitmask blobs, then the CBOR descriptor (with an
    /// optional `"masks"` sub-map carrying per-kind method / offset /
    /// length), then the `u64 cbor_offset` + `ENDF` end-magic.
    ///
    /// See `plans/BITMASK_FRAME.md` for the full design.
    NTensorMaskedFrame = 9,
}

impl FrameType {
    pub fn from_u16(v: u16) -> Result<Self> {
        match v {
            1 => Ok(FrameType::HeaderMetadata),
            2 => Ok(FrameType::HeaderIndex),
            3 => Ok(FrameType::HeaderHash),
            4 => Ok(FrameType::NTensorFrame),
            5 => Ok(FrameType::FooterHash),
            6 => Ok(FrameType::FooterIndex),
            7 => Ok(FrameType::FooterMetadata),
            8 => Ok(FrameType::PrecederMetadata),
            9 => Ok(FrameType::NTensorMaskedFrame),
            _ => Err(TensogramError::Framing(format!("unknown frame type: {v}"))),
        }
    }

    /// True for frames that carry an n-tensor data payload (types 4 and 9).
    /// Callers that want to treat both variants uniformly (e.g. the
    /// framing scanner) should use this instead of matching a single
    /// variant.
    pub fn is_data_object(self) -> bool {
        matches!(
            self,
            FrameType::NTensorFrame | FrameType::NTensorMaskedFrame
        )
    }
}

// ── Message Flags ────────────────────────────────────────────────────────────

/// Flags in the message preamble indicating which optional frames are present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MessageFlags(u16);

impl MessageFlags {
    pub const HEADER_METADATA: u16 = 1 << 0;
    pub const FOOTER_METADATA: u16 = 1 << 1;
    pub const HEADER_INDEX: u16 = 1 << 2;
    pub const FOOTER_INDEX: u16 = 1 << 3;
    pub const HEADER_HASHES: u16 = 1 << 4;
    pub const FOOTER_HASHES: u16 = 1 << 5;
    /// At least one PrecederMetadata frame is present in the data objects section.
    pub const PRECEDER_METADATA: u16 = 1 << 6;

    pub fn new(bits: u16) -> Self {
        Self(bits)
    }

    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn has(self, flag: u16) -> bool {
        self.0 & flag != 0
    }

    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    /// Returns true if at least one metadata frame (header or footer) is present.
    pub fn has_metadata(self) -> bool {
        self.has(Self::HEADER_METADATA) || self.has(Self::FOOTER_METADATA)
    }
}

// ── Data Object Flags ────────────────────────────────────────────────────────

/// Flags in data object frame header.
pub struct DataObjectFlags;

impl DataObjectFlags {
    /// Bit 0: CBOR descriptor position. 0 = before payload, 1 = after payload (default).
    pub const CBOR_AFTER_PAYLOAD: u16 = 1 << 0;
}

// ── Preamble ─────────────────────────────────────────────────────────────────

/// The fixed 24-byte message preamble.
#[derive(Debug, Clone)]
pub struct Preamble {
    pub version: u16,
    pub flags: MessageFlags,
    pub reserved: u32,
    /// Total message length including preamble and postamble.
    /// Zero indicates streaming mode (length unknown at write time).
    pub total_length: u64,
}

impl Preamble {
    pub fn read_from(buf: &[u8]) -> Result<Self> {
        if buf.len() < PREAMBLE_SIZE {
            return Err(TensogramError::Framing(format!(
                "buffer too short for preamble: {} < {PREAMBLE_SIZE}",
                buf.len()
            )));
        }
        if &buf[0..8] != MAGIC {
            return Err(TensogramError::Framing("invalid magic bytes".to_string()));
        }
        let version = read_u16_be(buf, 8);
        if version != WIRE_VERSION {
            return Err(TensogramError::Framing(format!(
                "unsupported message version {version} (required = {WIRE_VERSION}); \
                 v3 is a clean break from v2 with no backward compatibility — \
                 re-encode with tensogram ≥ 0.17.0"
            )));
        }
        Ok(Preamble {
            version,
            flags: MessageFlags::new(read_u16_be(buf, 10)),
            reserved: read_u32_be(buf, 12),
            total_length: read_u64_be(buf, 16),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.version.to_be_bytes());
        out.extend_from_slice(&self.flags.bits().to_be_bytes());
        out.extend_from_slice(&self.reserved.to_be_bytes());
        out.extend_from_slice(&self.total_length.to_be_bytes());
    }
}

// ── Frame Header ─────────────────────────────────────────────────────────────

/// The fixed 16-byte frame header.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub version: u16,
    pub flags: u16,
    /// Total length from start of frame header to end of ENDF marker (inclusive).
    pub total_length: u64,
}

impl FrameHeader {
    pub fn read_from(buf: &[u8]) -> Result<Self> {
        if buf.len() < FRAME_HEADER_SIZE {
            return Err(TensogramError::Framing(format!(
                "buffer too short for frame header: {} < {FRAME_HEADER_SIZE}",
                buf.len()
            )));
        }
        if &buf[0..2] != FRAME_MAGIC {
            return Err(TensogramError::Framing(format!(
                "invalid frame magic: {:?}",
                &buf[0..2]
            )));
        }
        let type_val = read_u16_be(buf, 2);
        let frame_type = FrameType::from_u16(type_val)?;
        Ok(FrameHeader {
            frame_type,
            version: read_u16_be(buf, 4),
            flags: read_u16_be(buf, 6),
            total_length: read_u64_be(buf, 8),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(FRAME_MAGIC);
        out.extend_from_slice(&(self.frame_type as u16).to_be_bytes());
        out.extend_from_slice(&self.version.to_be_bytes());
        out.extend_from_slice(&self.flags.to_be_bytes());
        out.extend_from_slice(&self.total_length.to_be_bytes());
    }
}

// ── Postamble ────────────────────────────────────────────────────────────────

/// The fixed 24-byte message postamble (footer terminator).
///
/// Layout (v3):
/// ```text
///   0 ..  8   first_footer_offset  (uint64 BE)
///   8 .. 16   total_length         (uint64 BE)  ← new in v3
///  16 .. 24   end magic "39277777" (8 bytes)
/// ```
///
/// The mirrored `total_length` lets a reader backward-scan for
/// `END_MAGIC` and then subtract `total_length` to locate the
/// message's `TENSOGRM` byte directly.  Zero indicates streaming
/// mode where the sink was not seekable at `finish()` time;
/// readers fall back to forward scan in that case.
#[derive(Debug, Clone)]
pub struct Postamble {
    /// Byte offset from message start to the first footer frame,
    /// or to the postamble itself if no footer frames exist.
    pub first_footer_offset: u64,
    /// Total byte length of the message, mirroring the preamble's
    /// `total_length` (new in v3).  `0` means "length unknown at
    /// finish-time" (non-seekable streaming sink).
    pub total_length: u64,
}

impl Postamble {
    pub fn read_from(buf: &[u8]) -> Result<Self> {
        if buf.len() < POSTAMBLE_SIZE {
            return Err(TensogramError::Framing(format!(
                "buffer too short for postamble: {} < {POSTAMBLE_SIZE}",
                buf.len()
            )));
        }
        let first_footer_offset = read_u64_be(buf, 0);
        let total_length = read_u64_be(buf, 8);
        if &buf[16..24] != END_MAGIC {
            return Err(TensogramError::Framing(
                "invalid end magic in postamble".to_string(),
            ));
        }
        Ok(Postamble {
            first_footer_offset,
            total_length,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.first_footer_offset.to_be_bytes());
        out.extend_from_slice(&self.total_length.to_be_bytes());
        out.extend_from_slice(END_MAGIC);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Read a big-endian u16 from `buf` at `offset`.
///
/// # Safety invariant
/// Callers must ensure `offset + 2 <= buf.len()`.  All call sites
/// in this crate validate buffer length before calling these helpers.
pub(crate) fn read_u16_be(buf: &[u8], offset: usize) -> u16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&buf[offset..offset + 2]);
    u16::from_be_bytes(bytes)
}

/// Read a big-endian u32 from `buf` at `offset`.
/// See [`read_u16_be`] for safety invariant.
pub(crate) fn read_u32_be(buf: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[offset..offset + 4]);
    u32::from_be_bytes(bytes)
}

/// Read a big-endian u64 from `buf` at `offset`.
/// See [`read_u16_be`] for safety invariant.
pub(crate) fn read_u64_be(buf: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[offset..offset + 8]);
    u64::from_be_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preamble_round_trip() {
        let preamble = Preamble {
            version: WIRE_VERSION,
            flags: MessageFlags::new(MessageFlags::HEADER_METADATA | MessageFlags::HEADER_INDEX),
            reserved: 0,
            total_length: 4096,
        };
        let mut buf = Vec::new();
        preamble.write_to(&mut buf);
        assert_eq!(buf.len(), PREAMBLE_SIZE);

        let parsed = Preamble::read_from(&buf).unwrap();
        assert_eq!(parsed.version, WIRE_VERSION);
        assert!(parsed.flags.has(MessageFlags::HEADER_METADATA));
        assert!(parsed.flags.has(MessageFlags::HEADER_INDEX));
        assert!(!parsed.flags.has(MessageFlags::FOOTER_INDEX));
        assert_eq!(parsed.total_length, 4096);
    }

    #[test]
    fn test_frame_header_round_trip() {
        let fh = FrameHeader {
            frame_type: FrameType::NTensorFrame,
            version: 1,
            flags: DataObjectFlags::CBOR_AFTER_PAYLOAD,
            total_length: 1024,
        };
        let mut buf = Vec::new();
        fh.write_to(&mut buf);
        assert_eq!(buf.len(), FRAME_HEADER_SIZE);

        let parsed = FrameHeader::read_from(&buf).unwrap();
        assert_eq!(parsed.frame_type, FrameType::NTensorFrame);
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.flags, DataObjectFlags::CBOR_AFTER_PAYLOAD);
        assert_eq!(parsed.total_length, 1024);
    }

    #[test]
    fn test_postamble_round_trip() {
        let pa = Postamble {
            first_footer_offset: 8192,
            total_length: 16384,
        };
        let mut buf = Vec::new();
        pa.write_to(&mut buf);
        assert_eq!(buf.len(), POSTAMBLE_SIZE);

        let parsed = Postamble::read_from(&buf).unwrap();
        assert_eq!(parsed.first_footer_offset, 8192);
        assert_eq!(parsed.total_length, 16384);
    }

    #[test]
    fn test_postamble_zero_total_length_streaming() {
        // Streaming-mode non-seekable sink: total_length left at 0 at finish().
        let pa = Postamble {
            first_footer_offset: 500,
            total_length: 0,
        };
        let mut buf = Vec::new();
        pa.write_to(&mut buf);
        assert_eq!(buf.len(), POSTAMBLE_SIZE);
        let parsed = Postamble::read_from(&buf).unwrap();
        assert_eq!(parsed.total_length, 0);
    }

    #[test]
    fn test_postamble_end_magic_at_fixed_offset() {
        // Pins the wire contract: the last 8 bytes of any postamble
        // are always the END_MAGIC, regardless of the preceding
        // fields.  Backward scanners rely on this.
        let pa = Postamble {
            first_footer_offset: 1,
            total_length: 2,
        };
        let mut buf = Vec::new();
        pa.write_to(&mut buf);
        assert_eq!(&buf[16..24], END_MAGIC);
    }

    #[test]
    fn test_invalid_magic() {
        let buf = vec![0u8; PREAMBLE_SIZE];
        assert!(Preamble::read_from(&buf).is_err());
    }

    #[test]
    fn test_v2_preamble_is_rejected() {
        // A hand-built preamble with version=2 must hard-fail v3 decoders.
        // Covers the clean-break contract: no backward compatibility with v2.
        let mut buf = Vec::with_capacity(PREAMBLE_SIZE);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes()); // flags
        buf.extend_from_slice(&0u32.to_be_bytes()); // reserved
        buf.extend_from_slice(&64u64.to_be_bytes()); // total_length
        let err = Preamble::read_from(&buf).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported message version 2"),
            "expected v2-rejection message, got: {msg}"
        );
        assert!(
            msg.contains("required = 3"),
            "expected required-version banner, got: {msg}"
        );
    }

    #[test]
    fn test_v1_preamble_is_rejected() {
        // Ditto for v1.
        let mut buf = Vec::with_capacity(PREAMBLE_SIZE);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&64u64.to_be_bytes());
        let err = Preamble::read_from(&buf).unwrap_err();
        assert!(err.to_string().contains("unsupported message version 1"));
    }

    #[test]
    fn test_future_version_is_rejected() {
        // A future version bump must also be rejected by this decoder.
        let mut buf = Vec::with_capacity(PREAMBLE_SIZE);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&99u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&64u64.to_be_bytes());
        assert!(Preamble::read_from(&buf).is_err());
    }

    #[test]
    fn test_invalid_frame_magic() {
        let buf = vec![0u8; FRAME_HEADER_SIZE];
        assert!(FrameHeader::read_from(&buf).is_err());
    }

    #[test]
    fn test_invalid_end_magic() {
        let mut buf = vec![0u8; POSTAMBLE_SIZE];
        // Valid offsets but bad magic at [16..24]
        buf[0..8].copy_from_slice(&100u64.to_be_bytes());
        buf[8..16].copy_from_slice(&200u64.to_be_bytes());
        assert!(Postamble::read_from(&buf).is_err());
    }

    #[test]
    fn test_frame_type_parse() {
        assert_eq!(FrameType::from_u16(1).unwrap(), FrameType::HeaderMetadata);
        assert_eq!(FrameType::from_u16(4).unwrap(), FrameType::NTensorFrame);
        assert_eq!(FrameType::from_u16(7).unwrap(), FrameType::FooterMetadata);
        assert_eq!(FrameType::from_u16(8).unwrap(), FrameType::PrecederMetadata);
        assert_eq!(
            FrameType::from_u16(9).unwrap(),
            FrameType::NTensorMaskedFrame
        );
        assert!(FrameType::from_u16(0).is_err());
        assert!(FrameType::from_u16(10).is_err());
    }

    #[test]
    fn test_is_data_object() {
        assert!(FrameType::NTensorFrame.is_data_object());
        assert!(FrameType::NTensorMaskedFrame.is_data_object());
        assert!(!FrameType::HeaderMetadata.is_data_object());
        assert!(!FrameType::PrecederMetadata.is_data_object());
        assert!(!FrameType::FooterHash.is_data_object());
    }

    #[test]
    fn test_message_flags() {
        let mut flags = MessageFlags::default();
        assert!(!flags.has_metadata());

        flags.set(MessageFlags::HEADER_METADATA);
        assert!(flags.has_metadata());
        assert!(flags.has(MessageFlags::HEADER_METADATA));
        assert!(!flags.has(MessageFlags::FOOTER_METADATA));

        flags.set(MessageFlags::FOOTER_INDEX);
        assert!(flags.has(MessageFlags::FOOTER_INDEX));
    }

    #[test]
    fn test_preceder_metadata_flag() {
        let mut flags = MessageFlags::default();
        assert!(!flags.has(MessageFlags::PRECEDER_METADATA));

        flags.set(MessageFlags::PRECEDER_METADATA);
        assert!(flags.has(MessageFlags::PRECEDER_METADATA));
        assert_eq!(flags.bits() & (1 << 6), 1 << 6);
    }

    #[test]
    fn test_preceder_metadata_frame_header_round_trip() {
        let fh = FrameHeader {
            frame_type: FrameType::PrecederMetadata,
            version: 1,
            flags: 0,
            total_length: 256,
        };
        let mut buf = Vec::new();
        fh.write_to(&mut buf);
        assert_eq!(buf.len(), FRAME_HEADER_SIZE);

        let parsed = FrameHeader::read_from(&buf).unwrap();
        assert_eq!(parsed.frame_type, FrameType::PrecederMetadata);
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.flags, 0);
        assert_eq!(parsed.total_length, 256);
    }

    #[test]
    fn test_truncated_preamble() {
        let buf = vec![0u8; 10];
        assert!(Preamble::read_from(&buf).is_err());
    }
}

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

/// Preamble size: magic(8) + version(2) + flags(2) + reserved(4) + total_length(8) = 24
pub const PREAMBLE_SIZE: usize = 24;
/// Frame header size: FR(2) + type(2) + version(2) + flags(2) + total_length(8) = 16
pub const FRAME_HEADER_SIZE: usize = 16;
/// Postamble size: first_footer_offset(8) + end_magic(8) = 16
pub const POSTAMBLE_SIZE: usize = 16;
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
    DataObject = 4,
    FooterHash = 5,
    FooterIndex = 6,
    FooterMetadata = 7,
    /// Per-object metadata frame that immediately precedes a DataObject frame.
    /// Carries a GlobalMetadata CBOR with `common` empty and a single-entry
    /// `payload` array containing metadata for the next data object.
    PrecederMetadata = 8,
}

impl FrameType {
    pub fn from_u16(v: u16) -> Result<Self> {
        match v {
            1 => Ok(FrameType::HeaderMetadata),
            2 => Ok(FrameType::HeaderIndex),
            3 => Ok(FrameType::HeaderHash),
            4 => Ok(FrameType::DataObject),
            5 => Ok(FrameType::FooterHash),
            6 => Ok(FrameType::FooterIndex),
            7 => Ok(FrameType::FooterMetadata),
            8 => Ok(FrameType::PrecederMetadata),
            _ => Err(TensogramError::Framing(format!("unknown frame type: {v}"))),
        }
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
        if version < 2 {
            return Err(TensogramError::Framing(format!(
                "unsupported message version {version} (versions 0 and 1 are deprecated, minimum is 2)"
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

/// The fixed 16-byte message postamble (footer terminator).
#[derive(Debug, Clone)]
pub struct Postamble {
    /// Byte offset from message start to the first footer frame,
    /// or to the postamble itself if no footer frames exist.
    pub first_footer_offset: u64,
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
        if &buf[8..16] != END_MAGIC {
            return Err(TensogramError::Framing(
                "invalid end magic in postamble".to_string(),
            ));
        }
        Ok(Postamble {
            first_footer_offset,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.first_footer_offset.to_be_bytes());
        out.extend_from_slice(END_MAGIC);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub(crate) fn read_u16_be(buf: &[u8], offset: usize) -> u16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&buf[offset..offset + 2]);
    u16::from_be_bytes(bytes)
}

pub(crate) fn read_u32_be(buf: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[offset..offset + 4]);
    u32::from_be_bytes(bytes)
}

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
            version: 2,
            flags: MessageFlags::new(MessageFlags::HEADER_METADATA | MessageFlags::HEADER_INDEX),
            reserved: 0,
            total_length: 4096,
        };
        let mut buf = Vec::new();
        preamble.write_to(&mut buf);
        assert_eq!(buf.len(), PREAMBLE_SIZE);

        let parsed = Preamble::read_from(&buf).unwrap();
        assert_eq!(parsed.version, 2);
        assert!(parsed.flags.has(MessageFlags::HEADER_METADATA));
        assert!(parsed.flags.has(MessageFlags::HEADER_INDEX));
        assert!(!parsed.flags.has(MessageFlags::FOOTER_INDEX));
        assert_eq!(parsed.total_length, 4096);
    }

    #[test]
    fn test_frame_header_round_trip() {
        let fh = FrameHeader {
            frame_type: FrameType::DataObject,
            version: 1,
            flags: DataObjectFlags::CBOR_AFTER_PAYLOAD,
            total_length: 1024,
        };
        let mut buf = Vec::new();
        fh.write_to(&mut buf);
        assert_eq!(buf.len(), FRAME_HEADER_SIZE);

        let parsed = FrameHeader::read_from(&buf).unwrap();
        assert_eq!(parsed.frame_type, FrameType::DataObject);
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.flags, DataObjectFlags::CBOR_AFTER_PAYLOAD);
        assert_eq!(parsed.total_length, 1024);
    }

    #[test]
    fn test_postamble_round_trip() {
        let pa = Postamble {
            first_footer_offset: 8192,
        };
        let mut buf = Vec::new();
        pa.write_to(&mut buf);
        assert_eq!(buf.len(), POSTAMBLE_SIZE);

        let parsed = Postamble::read_from(&buf).unwrap();
        assert_eq!(parsed.first_footer_offset, 8192);
    }

    #[test]
    fn test_invalid_magic() {
        let buf = vec![0u8; PREAMBLE_SIZE];
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
        // Valid offset but bad magic
        buf[0..8].copy_from_slice(&100u64.to_be_bytes());
        assert!(Postamble::read_from(&buf).is_err());
    }

    #[test]
    fn test_frame_type_parse() {
        assert_eq!(FrameType::from_u16(1).unwrap(), FrameType::HeaderMetadata);
        assert_eq!(FrameType::from_u16(4).unwrap(), FrameType::DataObject);
        assert_eq!(FrameType::from_u16(7).unwrap(), FrameType::FooterMetadata);
        assert_eq!(FrameType::from_u16(8).unwrap(), FrameType::PrecederMetadata);
        assert!(FrameType::from_u16(0).is_err());
        assert!(FrameType::from_u16(9).is_err());
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

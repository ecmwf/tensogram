// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Structural introspection of a message: walk its frames and read its
//! envelope, without decoding payloads or CBOR.
//!
//! This is the reference implementation every binding mirrors (see
//! `plans/INTERFACE_SYMMETRY.md` §9). Two entry points:
//!
//! - [`message_header`] — the 24-byte preamble as typed values: wire
//!   [`version`](MessageHeader::version), [`total_length`](MessageHeader::total_length),
//!   and the eight structural predicates (`has_header_metadata`, …). Tells you
//!   whether a message is random-access (metadata/index/hashes in the *header*)
//!   or streaming (in the *footer*) without reading a single frame.
//! - [`frames`] — a lazy iterator over the message's frames (types 1–9). The
//!   preamble and postamble are *not* frames and are not yielded; use
//!   [`message_header`] for the envelope.
//!
//! # Borrowing
//!
//! [`FrameInfo::payload`] borrows the caller's message buffer — it is a view,
//! not a copy, and is valid for as long as that buffer lives. The iterator
//! itself is just a cursor.

use crate::error::{Result, TensogramError};
use crate::wire::{
    END_MAGIC, FRAME_HEADER_SIZE, FrameFlags, FrameHeader, FrameType, MessageFlags, POSTAMBLE_SIZE,
    PREAMBLE_SIZE, Preamble, footer_size_for,
};

/// One frame's structural description plus a borrowed view of its content.
#[derive(Debug, Clone, Copy)]
pub struct FrameInfo<'a> {
    /// The frame's type (1–9).
    pub frame_type: FrameType,
    /// Frame-type-specific version field from the frame header.
    pub version: u16,
    /// Raw 16-bit frame flags (bit 1 is `HASH_PRESENT`; see
    /// [`has_hash`](Self::has_hash)).
    pub flags: u16,
    /// Byte offset of the frame header, **relative to the start of the
    /// message** that was passed to [`frames`].
    pub offset: usize,
    /// Whole-frame span in bytes: frame header through `ENDF`, excluding any
    /// alignment padding that follows.
    pub length: usize,
    /// The frame's content: everything between the 16-byte frame header and
    /// the type-specific footer.
    ///
    /// For a data-object frame (type 9) this is the encoded payload *and* the
    /// trailing CBOR descriptor (the 20-byte footer `[cbor_offset][hash][ENDF]`
    /// is excluded). For every other frame type it is the CBOR body (the
    /// 12-byte footer `[hash][ENDF]` is excluded). Use
    /// [`offset`](Self::offset)/[`length`](Self::length) for the whole frame.
    pub payload: &'a [u8],
}

impl FrameInfo<'_> {
    /// `true` if this frame's `HASH_PRESENT` flag is set, i.e. its hash slot
    /// holds a meaningful digest.
    #[must_use]
    pub fn has_hash(&self) -> bool {
        self.flags & FrameFlags::HASH_PRESENT != 0
    }
}

/// A message's envelope (preamble), decoded into typed values.
#[derive(Debug, Clone, Copy)]
pub struct MessageHeader {
    /// Wire-format version (3 in v3).
    pub version: u16,
    /// Total message length in bytes, preamble through postamble. `0` in a
    /// streaming message whose length was never back-filled.
    pub total_length: u64,
    /// Raw preamble flag bits; prefer the `has_*` predicates.
    pub flags: MessageFlags,
}

impl MessageHeader {
    /// A `HeaderMetadata` frame is present (random-access mode).
    #[must_use]
    pub fn has_header_metadata(&self) -> bool {
        self.flags.has(MessageFlags::HEADER_METADATA)
    }
    /// A `FooterMetadata` frame is present (streaming mode).
    #[must_use]
    pub fn has_footer_metadata(&self) -> bool {
        self.flags.has(MessageFlags::FOOTER_METADATA)
    }
    /// A `HeaderIndex` frame is present.
    #[must_use]
    pub fn has_header_index(&self) -> bool {
        self.flags.has(MessageFlags::HEADER_INDEX)
    }
    /// A `FooterIndex` frame is present.
    #[must_use]
    pub fn has_footer_index(&self) -> bool {
        self.flags.has(MessageFlags::FOOTER_INDEX)
    }
    /// A `HeaderHash` frame is present.
    #[must_use]
    pub fn has_header_hashes(&self) -> bool {
        self.flags.has(MessageFlags::HEADER_HASHES)
    }
    /// A `FooterHash` frame is present.
    #[must_use]
    pub fn has_footer_hashes(&self) -> bool {
        self.flags.has(MessageFlags::FOOTER_HASHES)
    }
    /// At least one `PrecederMetadata` frame appears in the body.
    #[must_use]
    pub fn has_preceder_metadata(&self) -> bool {
        self.flags.has(MessageFlags::PRECEDER_METADATA)
    }
    /// Advisory: every frame in this message has its per-frame `HASH_PRESENT`
    /// bit set. The per-frame bit remains authoritative for any single frame.
    #[must_use]
    pub fn has_hashes_present(&self) -> bool {
        self.flags.has(MessageFlags::HASHES_PRESENT)
    }
}

/// Read a message's envelope without walking its frames.
///
/// `message` must start at the `TENSOGRM` preamble magic.
pub fn message_header(message: &[u8]) -> Result<MessageHeader> {
    let p = Preamble::read_from(message)?;
    Ok(MessageHeader {
        version: p.version,
        total_length: p.total_length,
        flags: p.flags,
    })
}

/// Lazily iterate the frames of one message.
///
/// `message` must start at the `TENSOGRM` preamble magic — typically a slice
/// obtained from [`crate::scan`]. Yields `Err` once and then stops if the frame
/// chain is malformed.
pub fn frames(message: &[u8]) -> Result<FrameIter<'_>> {
    let p = Preamble::read_from(message)?;
    // Bound the walk to the frame region so the postamble is never mistaken
    // for a frame. A streaming message that was never back-filled reports
    // total_length == 0; fall back to the buffer end. Clamp to the buffer so a
    // truncated message surfaces a framing error rather than reading past it.
    let end = if p.total_length > 0 {
        usize::try_from(p.total_length)
            .unwrap_or(usize::MAX)
            .saturating_sub(POSTAMBLE_SIZE)
            .min(message.len())
    } else if message.len() >= POSTAMBLE_SIZE && message.ends_with(END_MAGIC) {
        // Streaming message whose total_length was never back-filled: the
        // postamble is still the trailing POSTAMBLE_SIZE bytes (its END_MAGIC
        // sits at the very end), so exclude it rather than scanning into it.
        message.len() - POSTAMBLE_SIZE
    } else {
        message.len()
    };
    Ok(FrameIter {
        buf: message,
        pos: PREAMBLE_SIZE,
        end,
        done: false,
    })
}

/// Lazy cursor over a message's frames. See [`frames`].
#[derive(Debug)]
pub struct FrameIter<'a> {
    buf: &'a [u8],
    pos: usize,
    end: usize,
    done: bool,
}

impl<'a> Iterator for FrameIter<'a> {
    type Item = Result<FrameInfo<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // Skip inter-frame alignment padding (zero bytes after ENDF).
        while self.pos + 2 <= self.end && &self.buf[self.pos..self.pos + 2] != b"FR" {
            self.pos += 1;
        }
        if self.pos + 2 > self.end {
            // Only padding left — a clean end of the frame region.
            self.done = true;
            return None;
        }
        // A frame starts here, but the region ends before its header does.
        if self.pos + FRAME_HEADER_SIZE > self.end {
            self.done = true;
            return Some(Err(TensogramError::Framing(format!(
                "truncated frame header at offset {}: {} bytes available, {FRAME_HEADER_SIZE} required",
                self.pos,
                self.end - self.pos
            ))));
        }
        let fh = match FrameHeader::read_from(&self.buf[self.pos..]) {
            Ok(fh) => fh,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        let footer = footer_size_for(fh.frame_type);
        let min_total = FRAME_HEADER_SIZE + footer;
        let total = usize::try_from(fh.total_length).unwrap_or(usize::MAX);
        if total < min_total || self.pos + total > self.end {
            self.done = true;
            return Some(Err(TensogramError::Framing(format!(
                "frame at offset {} declares total_length {} which does not fit the message",
                self.pos, fh.total_length
            ))));
        }
        let info = FrameInfo {
            frame_type: fh.frame_type,
            version: fh.version,
            flags: fh.flags,
            offset: self.pos,
            length: total,
            payload: &self.buf[self.pos + FRAME_HEADER_SIZE..self.pos + total - footer],
        };
        self.pos += total;
        Some(Ok(info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata, StreamingEncoder,
        encode,
    };
    use std::collections::BTreeMap;

    fn desc(shape: Vec<u64>) -> DataObjectDescriptor {
        let strides = {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
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
            byte_order: ByteOrder::Little,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        }
    }

    /// Buffered encode → random-access layout (metadata/index in the header).
    fn buffered_message(n_objects: usize) -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let d = desc(vec![4]);
        let data = vec![0u8; 16];
        let objs: Vec<(&DataObjectDescriptor, &[u8])> =
            (0..n_objects).map(|_| (&d, data.as_slice())).collect();
        encode(&meta, &objs, &EncodeOptions::default()).expect("encode")
    }

    /// Streaming encode → footer-side index/hashes.
    fn streamed_message() -> Vec<u8> {
        let meta = GlobalMetadata::default();
        let d = desc(vec![4]);
        let data = vec![0u8; 16];
        let mut enc = StreamingEncoder::new(
            std::io::Cursor::new(Vec::new()),
            &meta,
            &EncodeOptions::default(),
        )
        .expect("streaming new");
        enc.write_object(&d, &data).expect("write");
        enc.finish().expect("finish").into_inner()
    }

    fn collect(msg: &[u8]) -> Vec<FrameInfo<'_>> {
        frames(msg)
            .expect("frames")
            .map(|r| r.expect("frame ok"))
            .collect()
    }

    #[test]
    fn walks_frames_and_finds_one_data_object_per_object() {
        let msg = buffered_message(3);
        let fs = collect(&msg);
        assert!(!fs.is_empty(), "expected frames");
        let data_objects = fs.iter().filter(|f| f.frame_type.is_data_object()).count();
        assert_eq!(data_objects, 3, "one data-object frame per encoded object");
        // The preamble/postamble are NOT frames.
        assert!(
            fs[0].offset >= PREAMBLE_SIZE,
            "first frame starts after the preamble"
        );
    }

    #[test]
    fn frames_are_ordered_in_bounds_and_do_not_overlap() {
        let msg = buffered_message(2);
        let fs = collect(&msg);
        let mut prev_end = PREAMBLE_SIZE;
        for f in &fs {
            assert!(f.offset >= prev_end, "frames must not overlap");
            assert!(f.offset + f.length <= msg.len(), "frame must be in bounds");
            assert!(
                f.length >= FRAME_HEADER_SIZE,
                "frame spans at least its header"
            );
            prev_end = f.offset + f.length;
        }
    }

    #[test]
    fn payload_excludes_header_and_footer_and_lies_inside_the_frame() {
        let msg = buffered_message(1);
        for f in collect(&msg) {
            // Content starts after the 16-byte frame header ...
            let frame = &msg[f.offset..f.offset + f.length];
            let start = FRAME_HEADER_SIZE;
            let end = start + f.payload.len();
            assert!(end <= frame.len(), "payload must fit inside the frame");
            assert_eq!(
                f.payload,
                &frame[start..end],
                "payload is the content slice"
            );
            // ... and stops before the footer, which always ends with ENDF.
            assert_eq!(&frame[frame.len() - 4..], b"ENDF");
            assert!(
                f.payload.len() < f.length - FRAME_HEADER_SIZE,
                "payload must exclude the type-specific footer"
            );
        }
    }

    #[test]
    fn buffered_message_header_flags_match_its_frames_exactly() {
        // In buffered (random-access) mode the encoder knows the whole message
        // up front, so every preamble flag is an exact statement about the
        // frames present.
        let msg = buffered_message(2);
        let h = message_header(&msg).expect("header");
        assert_eq!(h.version, crate::WIRE_VERSION);
        assert_eq!(h.total_length as usize, msg.len());
        let types: Vec<FrameType> = collect(&msg).iter().map(|f| f.frame_type).collect();
        let has = |t: FrameType| types.contains(&t);

        assert_eq!(h.has_header_metadata(), has(FrameType::HeaderMetadata));
        assert_eq!(h.has_footer_metadata(), has(FrameType::FooterMetadata));
        assert_eq!(h.has_header_index(), has(FrameType::HeaderIndex));
        assert_eq!(h.has_footer_index(), has(FrameType::FooterIndex));
        assert_eq!(h.has_header_hashes(), has(FrameType::HeaderHash));
        assert_eq!(h.has_footer_hashes(), has(FrameType::FooterHash));
        assert_eq!(h.has_preceder_metadata(), has(FrameType::PrecederMetadata));
    }

    #[test]
    fn streaming_message_header_flags_are_advisory_but_never_understate() {
        // In streaming mode the preamble is written before any object, so the
        // encoder cannot know what will follow: it sets PRECEDER_METADATA
        // optimistically (see `build_preamble_and_header_bytes`) and may leave
        // total_length at 0. The invariant that still holds in BOTH directions
        // for the frames it can predict, and one-directionally for the rest, is:
        // a frame that IS present must be advertised.
        let msg = streamed_message();
        let h = message_header(&msg).expect("header");
        assert_eq!(h.version, crate::WIRE_VERSION);
        let types: Vec<FrameType> = collect(&msg).iter().map(|f| f.frame_type).collect();
        let implies = |present: bool, flag: bool| !present || flag;

        assert!(implies(
            types.contains(&FrameType::HeaderMetadata),
            h.has_header_metadata()
        ));
        assert!(implies(
            types.contains(&FrameType::FooterMetadata),
            h.has_footer_metadata()
        ));
        assert!(implies(
            types.contains(&FrameType::HeaderIndex),
            h.has_header_index()
        ));
        assert!(implies(
            types.contains(&FrameType::FooterIndex),
            h.has_footer_index()
        ));
        assert!(implies(
            types.contains(&FrameType::HeaderHash),
            h.has_header_hashes()
        ));
        assert!(implies(
            types.contains(&FrameType::FooterHash),
            h.has_footer_hashes()
        ));
        assert!(implies(
            types.contains(&FrameType::PrecederMetadata),
            h.has_preceder_metadata()
        ));
        // The walk must still stop cleanly at the postamble even though
        // total_length was never back-filled.
        assert!(
            types.contains(&FrameType::FooterIndex),
            "footer frames are reachable"
        );
    }

    #[test]
    fn has_hash_matches_the_per_frame_flag() {
        // Default options hash frames, so at least one frame must advertise it,
        // and every frame's has_hash() must mirror its raw flag bit.
        let msg = buffered_message(1);
        let fs = collect(&msg);
        assert!(
            fs.iter().any(FrameInfo::has_hash),
            "default encode hashes frames"
        );
        for f in &fs {
            assert_eq!(f.has_hash(), f.flags & (1 << 1) != 0);
        }
    }

    #[test]
    fn rejects_a_buffer_that_is_not_a_message() {
        assert!(message_header(b"not a tensogram message").is_err());
        assert!(frames(b"not a tensogram message").is_err());
        assert!(message_header(&[]).is_err());
    }

    #[test]
    fn truncated_frame_chain_yields_an_error_item_then_stops() {
        let msg = buffered_message(2);
        // Cut *inside* the last frame: the preamble stays valid, the chain does not.
        let last = *collect(&msg).last().expect("at least one frame");
        let truncated = &msg[..last.offset + 8];
        let mut it = frames(truncated).expect("preamble still parses");
        let mut saw_err = false;
        for item in it.by_ref() {
            if item.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "a truncated chain must surface an error");
        assert!(it.next().is_none(), "iteration stops after an error");
    }
}

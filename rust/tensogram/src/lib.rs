// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

pub mod decode;
pub mod doctor;
pub mod dtype;
pub mod encode;
pub mod error;
pub mod file;
pub mod framing;
pub mod hash;
pub mod iter;
pub mod metadata;
// Internal thread-dispatch helpers for the multi-threaded coding
// pipeline.  Callers configure threading via `EncodeOptions.threads`
// and `DecodeOptions.threads`; the constants
// `DEFAULT_PARALLEL_THRESHOLD_BYTES` and `ENV_THREADS` are re-exported
// at the crate root for documentation.
mod parallel;
pub mod pipeline;
#[cfg(feature = "remote")]
pub mod remote;
pub mod remote_scan_parse;
pub(crate) mod restore;
pub mod scan_opts;
pub mod streaming;
pub(crate) mod substitute_and_mask;
pub mod types;
pub mod validate;
pub mod wire;

pub use decode::{
    DecodeOptions, DecodedMaskSet, DecodedObjectWithMasks, decode, decode_descriptors,
    decode_metadata, decode_object, decode_range, decode_range_from_payload, decode_with_masks,
};
pub use dtype::Dtype;
pub use encode::{AggregateHashPolicy, EncodeOptions, encode, encode_pre_encoded};
pub use error::{Result, TensogramError};
pub use file::{MessageLayout, TensogramFile};
pub use framing::{
    ScanOptions, data_object_inline_hashes, scan, scan_file, scan_file_with_options,
    scan_with_options,
};
pub use hash::{HASH_ALGORITHM_NAME, compute_hash, parse_hash_name};
pub use iter::{FileMessageIter, MessageIter, ObjectIter, messages, objects, objects_metadata};
pub use metadata::{RESERVED_KEY, compute_common, verify_canonical_cbor};
pub use parallel::{DEFAULT_PARALLEL_THRESHOLD_BYTES, ENV_THREADS};
pub use pipeline::{DEFAULT_SIMPLE_PACKING_BITS, DataPipeline, apply_pipeline};
pub use remote_scan_parse::{
    BackwardCommit, BackwardOutcome, ForwardOutcome, footer_region_present,
    parse_backward_postamble, parse_forward_preamble, same_message_check,
    validate_backward_preamble,
};
pub use scan_opts::RemoteScanOptions;
pub use streaming::StreamingEncoder;
pub use tensogram_encodings::bitmask::MaskMethod;
pub use tensogram_encodings::pipeline::CompressionBackend;
pub use types::{
    ByteOrder, DataObjectDescriptor, DecodedObject, GlobalMetadata, HashFrame, IndexFrame,
    MaskDescriptor, MasksMetadata,
};
pub use validate::{
    FileIssue, FileValidationReport, IssueCode, IssueSeverity, ValidateOptions, ValidationIssue,
    ValidationLevel, ValidationReport, validate_buffer, validate_file, validate_message,
};
pub use wire::{FrameType, MessageFlags, WIRE_VERSION};

#[cfg(feature = "remote")]
pub use remote::is_remote_url;

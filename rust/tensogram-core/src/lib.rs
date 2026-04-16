// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

pub mod decode;
pub mod dtype;
pub mod encode;
pub mod error;
pub mod file;
pub mod framing;
pub mod hash;
pub mod iter;
pub mod metadata;
pub mod parallel;
pub mod pipeline;
#[cfg(feature = "remote")]
pub mod remote;
pub mod streaming;
pub mod types;
pub mod validate;
pub mod wire;

pub use decode::{
    decode, decode_descriptors, decode_metadata, decode_object, decode_range,
    decode_range_from_payload, DecodeOptions,
};
pub use dtype::Dtype;
pub use encode::{encode, encode_pre_encoded, EncodeOptions};
pub use error::{Result, TensogramError};
pub use file::TensogramFile;
pub use framing::{scan, scan_file};
pub use hash::{compute_hash, verify_hash, HashAlgorithm};
pub use iter::{messages, objects, objects_metadata, FileMessageIter, MessageIter, ObjectIter};
pub use metadata::{compute_common, verify_canonical_cbor, RESERVED_KEY};
pub use pipeline::{apply_pipeline, DataPipeline};
pub use streaming::StreamingEncoder;
pub use tensogram_encodings::pipeline::CompressionBackend;
pub use types::{
    ByteOrder, DataObjectDescriptor, DecodedObject, GlobalMetadata, HashDescriptor, HashFrame,
    IndexFrame,
};
pub use validate::{
    validate_buffer, validate_file, validate_message, FileIssue, FileValidationReport, IssueCode,
    IssueSeverity, ValidateOptions, ValidationIssue, ValidationLevel, ValidationReport,
};
pub use wire::{FrameType, MessageFlags};

#[cfg(feature = "remote")]
pub use remote::is_remote_url;

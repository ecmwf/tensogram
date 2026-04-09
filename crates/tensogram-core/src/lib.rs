pub mod decode;
pub mod dtype;
pub mod encode;
pub mod error;
pub mod file;
pub mod framing;
pub mod hash;
pub mod iter;
pub mod metadata;
pub mod pipeline;
pub mod streaming;
pub mod types;
pub mod wire;

pub use decode::{
    decode, decode_descriptors, decode_metadata, decode_object, decode_range, DecodeOptions,
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
pub use wire::{FrameType, MessageFlags};

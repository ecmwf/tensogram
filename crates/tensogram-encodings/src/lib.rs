pub mod compression;
#[cfg(feature = "szip")]
pub mod libaec;
pub mod pipeline;
pub mod shuffle;
pub mod simple_packing;
#[cfg(feature = "zfp")]
pub mod zfp_ffi;

pub use pipeline::{
    ByteOrder, CompressionBackend, CompressionType, EncodingType, FilterType, PipelineConfig,
    PipelineResult,
};

#[cfg(feature = "blosc2")]
pub use pipeline::Blosc2Codec;
#[cfg(feature = "sz3")]
pub use pipeline::Sz3ErrorBound;
#[cfg(feature = "zfp")]
pub use pipeline::ZfpMode;

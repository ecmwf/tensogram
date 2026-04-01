use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Float16,
    Bfloat16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bitmask,
}

impl Dtype {
    /// Returns byte width per element. Bitmask returns 0 (sub-byte; callers handle specially).
    pub fn byte_width(&self) -> usize {
        match self {
            Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Float32 => 4,
            Dtype::Float64 => 8,
            Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
            Dtype::Int8 | Dtype::Uint8 => 1,
            Dtype::Int16 | Dtype::Uint16 => 2,
            Dtype::Int32 | Dtype::Uint32 => 4,
            Dtype::Int64 | Dtype::Uint64 => 8,
            Dtype::Bitmask => 0,
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Dtype::Float16 => "float16",
            Dtype::Bfloat16 => "bfloat16",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
            Dtype::Complex64 => "complex64",
            Dtype::Complex128 => "complex128",
            Dtype::Int8 => "int8",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Uint8 => "uint8",
            Dtype::Uint16 => "uint16",
            Dtype::Uint32 => "uint32",
            Dtype::Uint64 => "uint64",
            Dtype::Bitmask => "bitmask",
        };
        write!(f, "{s}")
    }
}

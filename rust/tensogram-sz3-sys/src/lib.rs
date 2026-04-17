// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Clean-room FFI bindings for the SZ3 lossy compression library.
//!
//! This crate provides low-level C FFI bindings to the SZ3 header-only C++
//! library via a thin C++ shim (`cpp/sz3_ffi.cpp`).

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/// Flat C-compatible configuration struct mirroring the SZ3 C++ `Config` class.
///
/// The `dims` pointer is heap-allocated on the C++ side when returned from
/// [`sz3_decompress_config`]; the caller must free it with [`sz3_dealloc_size_t`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SZ3_Config {
    pub N: u8,
    pub dims: *mut usize,
    pub num: usize,
    pub errorBoundMode: u8,
    pub absErrorBound: f64,
    pub relErrorBound: f64,
    pub l2normErrorBound: f64,
    pub psnrErrorBound: f64,
    pub cmprAlgo: u8,
    pub lorenzo: bool,
    pub lorenzo2: bool,
    pub regression: bool,
    pub openmp: bool,
    pub dataType: u8,
    pub blockSize: i32,
    pub quantbinCnt: i32,
}

// ---------------------------------------------------------------------------
// Per-type modules – each exposes compress_size_bound, compress, decompress
// ---------------------------------------------------------------------------

macro_rules! impl_sz3_type {
    ($mod_name:ident, $rust_ty:ty, $data_type:expr_2021, $suffix:ident) => {
        pub mod $mod_name {
            use super::SZ3_Config;

            /// The Rust primitive type this module operates on.
            pub type ty = $rust_ty;

            /// SZ3 `dataType` discriminant for this type.
            pub const DATA_TYPE_TYPE: u8 = $data_type;

            unsafe extern "C" {
                #[link_name = concat!("sz3_compress_size_bound_", stringify!($suffix))]
                pub fn compress_size_bound(config: SZ3_Config) -> usize;

                #[link_name = concat!("sz3_compress_", stringify!($suffix))]
                pub fn compress(
                    config: SZ3_Config,
                    data: *const ty,
                    compressed_data: *mut i8,
                    compressed_capacity: usize,
                ) -> usize;

                #[link_name = concat!("sz3_decompress_", stringify!($suffix))]
                pub fn decompress(
                    compressed_data: *const i8,
                    compressed_len: usize,
                    decompressed_data: *mut ty,
                );
            }
        }
    };
}

impl_sz3_type!(impl_f32, f32, 0, f32);
impl_sz3_type!(impl_f64, f64, 1, f64);
impl_sz3_type!(impl_u8, u8, 2, u8);
impl_sz3_type!(impl_i8, i8, 3, i8);
impl_sz3_type!(impl_u16, u16, 4, u16);
impl_sz3_type!(impl_i16, i16, 5, i16);
impl_sz3_type!(impl_u32, u32, 6, u32);
impl_sz3_type!(impl_i32, i32, 7, i32);
impl_sz3_type!(impl_u64, u64, 8, u64);
impl_sz3_type!(impl_i64, i64, 9, i64);

// ---------------------------------------------------------------------------
// Top-level helpers
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Parse a compressed SZ3 blob and return its configuration **without**
    /// decompressing the payload.  The returned `SZ3_Config::dims` pointer is
    /// heap-allocated; free it with [`sz3_dealloc_size_t`].
    pub fn sz3_decompress_config(data: *const i8, len: usize) -> SZ3_Config;

    /// Free a `size_t[]` pointer that was allocated on the C++ side (e.g. the
    /// `dims` field of [`SZ3_Config`]).
    pub fn sz3_dealloc_size_t(ptr: *mut usize);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// SZ3 algorithm and error-bound mode constants.
///
/// Named to match the C++ `SZ3::` namespace; the `ALGO_` and `EB_` prefixes
/// distinguish compression-algorithm IDs from error-bound mode IDs.
#[allow(non_snake_case)]
pub mod SZ3 {
    // Compression algorithms
    pub const ALGO_ALGO_LORENZO_REG: u32 = 0;
    pub const ALGO_ALGO_INTERP_LORENZO: u32 = 1;
    pub const ALGO_ALGO_INTERP: u32 = 2;
    pub const ALGO_ALGO_NOPRED: u32 = 3;
    pub const ALGO_ALGO_LOSSLESS: u32 = 4;
    pub const ALGO_ALGO_BIOMD: u32 = 5;
    pub const ALGO_ALGO_BIOMDXTC: u32 = 6;

    // Error-bound modes
    pub const EB_EB_ABS: u32 = 0;
    pub const EB_EB_REL: u32 = 1;
    pub const EB_EB_PSNR: u32 = 2;
    pub const EB_EB_L2NORM: u32 = 3;
    pub const EB_EB_ABS_AND_REL: u32 = 4;
    pub const EB_EB_ABS_OR_REL: u32 = 5;
}

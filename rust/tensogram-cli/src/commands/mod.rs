// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

#[cfg(feature = "grib")]
pub mod convert_grib;
#[cfg(feature = "netcdf")]
pub mod convert_netcdf;
pub mod copy;
pub mod doctor;
pub mod dump;
pub mod get;
pub mod info;
pub mod ls;
pub mod merge;
pub mod reshuffle;
pub mod set;
pub mod split;
pub mod validate;

/// NaN / Inf mask-companion options collected from the global CLI
/// flags in `main.rs` and forwarded to every encoding-capable
/// subcommand (`merge`, `split`, `reshuffle`, `convert-grib`,
/// `convert-netcdf`).
///
/// Defaults correspond to the library's default-reject policy — pass
/// `--allow-nan` / `--allow-inf` on the CLI (or set the matching env
/// vars, see `docs/src/guide/nan-inf-handling.md`) to opt in to the
/// bitmask companion frame.
#[derive(Debug, Clone, Default)]
pub struct MaskCliOptions {
    pub allow_nan: bool,
    pub allow_inf: bool,
    pub nan_mask_method: Option<String>,
    pub pos_inf_mask_method: Option<String>,
    pub neg_inf_mask_method: Option<String>,
    pub small_mask_threshold_bytes: Option<usize>,
}

impl MaskCliOptions {
    /// Apply the CLI-side NaN / Inf mask settings to an
    /// [`tensogram::EncodeOptions`] struct.  Unknown mask method
    /// names produce a clean error string the caller can surface.
    pub fn apply(&self, opts: &mut tensogram::EncodeOptions) -> Result<(), String> {
        use tensogram::encode::MaskMethod;
        opts.allow_nan = self.allow_nan;
        opts.allow_inf = self.allow_inf;
        // Delegate the error message to `MaskError::UnknownMethod`'s
        // Display so the accepted-names list stays in one place
        // across every binding.
        let parse = |name: &Option<String>, default: &MaskMethod| -> Result<MaskMethod, String> {
            let Some(name) = name.as_deref() else {
                return Ok(default.clone());
            };
            MaskMethod::from_name(name).map_err(|e| e.to_string())
        };
        let defaults = tensogram::EncodeOptions::default();
        opts.nan_mask_method = parse(&self.nan_mask_method, &defaults.nan_mask_method)?;
        opts.pos_inf_mask_method = parse(&self.pos_inf_mask_method, &defaults.pos_inf_mask_method)?;
        opts.neg_inf_mask_method = parse(&self.neg_inf_mask_method, &defaults.neg_inf_mask_method)?;
        if let Some(t) = self.small_mask_threshold_bytes {
            opts.small_mask_threshold_bytes = t;
        }
        Ok(())
    }
}

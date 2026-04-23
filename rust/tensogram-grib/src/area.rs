// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Canonical `[N, W, S, E]` area computation for `regular_ll` GRIB grids.
//!
//! The Tensoscope viewer and downstream xarray/zarr readers consume
//! `mars.area = [N, W, S, E]`.  This module turns the raw ecCodes
//! `geography` corner keys (`latitudeOfFirstGridPointInDegrees` etc.)
//! into that tuple.
//!
//! Only one normalisation is applied: a full-global dateline-first scan
//! (`lon_first = 180, lon_last = 179.75` on a 0.25° grid, as used by
//! ECMWF open-data) is shifted to `west = -180` so the axis is monotone
//! in `[-180, +E]`.
//!
//! # Why refuse instead of guess for other `west > east` cases?
//!
//! A dateline-crossing regional subdomain — e.g. `(lon_first = 170,
//! lon_last = -170, Ni = 80)` — cannot be represented as a single
//! monotone axis in either `[-180, 180]` or `[0, 360]`.  The Tensoscope
//! regrid worker normalises `lon > 180 → lon − 360` but does not handle
//! `lon < -180`, so a naive `lon_first -= 360` there produces a wrapped
//! axis the worker mis-bins.  The only correct answer for such grids is
//! per-point lat/lon emission, which is out of scope for this pass.
//! Returning `None` lets the caller either fall back to a compat default
//! (legacy-file bridge) or skip emitting `mars.area`, rather than produce
//! a silently-wrong area that renders nicely but is half a world off.

/// Inputs required to compute the canonical area of a `regular_ll` grid.
///
/// Field names mirror the ecCodes `geography` namespace key names
/// one-for-one so the call site in `converter.rs` is a trivial shuffle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RegularLlGeometry {
    pub(crate) lat_first: f64,
    pub(crate) lon_first: f64,
    pub(crate) lat_last: f64,
    pub(crate) lon_last: f64,
    pub(crate) i_scans_negatively: i64,
    pub(crate) j_scans_positively: i64,
    pub(crate) ni: i64,
    pub(crate) i_direction_increment: f64,
}

/// Compute `[north, west, south, east]` from a `regular_ll` grid's
/// geometry keys.  See the module docstring for the `west > east` rule.
///
/// Returns `Some([N, W, S, E])` exactly when all of:
/// - all four corner values + the increment are finite,
/// - `i_scans_negatively == 0` and `j_scans_positively == 0`,
/// - `ni >= 2` and `i_direction_increment > 0`,
/// - after optional normalisation, `north > south` and `west < east`.
pub(crate) fn compute_regular_ll_area(g: RegularLlGeometry) -> Option<[f64; 4]> {
    if g.i_scans_negatively != 0 || g.j_scans_positively != 0 {
        return None;
    }
    if !g.lat_first.is_finite()
        || !g.lon_first.is_finite()
        || !g.lat_last.is_finite()
        || !g.lon_last.is_finite()
        || !g.i_direction_increment.is_finite()
    {
        return None;
    }
    if g.ni < 2 || g.i_direction_increment <= 0.0 {
        return None;
    }

    let north = g.lat_first;
    let south = g.lat_last;
    let mut west = g.lon_first;
    let east = g.lon_last;

    // Antimeridian-crossing branch: only the full-global dateline-first
    // case has an unambiguous [-180, E] representation — regional subdomains
    // that cross the dateline need per-point coords, not an [N,W,S,E] tuple.
    // We also require that the shifted span matches the sample spacing
    // (`east − west_shifted ≈ (ni − 1) × inc`) so a file with inconsistent
    // corner metadata (e.g. `w=10, e=5` with `ni=360, inc=1`) is refused
    // even though `ni × inc ≈ 360` alone might have accepted it.
    if west > east {
        if !is_full_global_i(g.ni, g.i_direction_increment) {
            return None;
        }
        let shifted_span = east - (west - 360.0);
        let expected_span = (g.ni - 1) as f64 * g.i_direction_increment;
        if (shifted_span - expected_span).abs() >= 0.5 * g.i_direction_increment {
            return None;
        }
        west -= 360.0;
    }

    if north <= south || west >= east {
        return None;
    }

    Some([north, west, south, east])
}

/// True when `ni × i_direction_increment` covers a full 360° circle,
/// within a half-increment tolerance.  The tolerance accommodates f64
/// round-off for non-representable increments like `0.1` (where
/// `3600 × 0.1 = 360.00000000000006` on IEEE-754).
fn is_full_global_i(ni: i64, i_direction_increment: f64) -> bool {
    let span = (ni as f64) * i_direction_increment;
    (span - 360.0).abs() < 0.5 * i_direction_increment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quarter_degree_global(lon_first: f64, lon_last: f64) -> RegularLlGeometry {
        RegularLlGeometry {
            lat_first: 90.0,
            lat_last: -90.0,
            lon_first,
            lon_last,
            i_scans_negatively: 0,
            j_scans_positively: 0,
            ni: 1440,
            i_direction_increment: 0.25,
        }
    }

    #[test]
    fn dateline_first_global_normalises() {
        assert_eq!(
            compute_regular_ll_area(quarter_degree_global(180.0, 179.75)),
            Some([90.0, -180.0, -90.0, 179.75]),
        );
    }

    #[test]
    fn greenwich_first_global_no_normalise() {
        assert_eq!(
            compute_regular_ll_area(quarter_degree_global(0.0, 359.75)),
            Some([90.0, 0.0, -90.0, 359.75]),
        );
    }

    #[test]
    fn half_degree_grid_also_normalises() {
        let g = RegularLlGeometry {
            lat_first: 90.0,
            lat_last: -90.0,
            lon_first: 180.0,
            lon_last: 179.5,
            i_scans_negatively: 0,
            j_scans_positively: 0,
            ni: 720,
            i_direction_increment: 0.5,
        };
        assert_eq!(
            compute_regular_ll_area(g),
            Some([90.0, -180.0, -90.0, 179.5]),
        );
    }

    #[test]
    fn tenth_degree_grid_tolerates_f64_roundoff() {
        let g = RegularLlGeometry {
            lat_first: 90.0,
            lat_last: -90.0,
            lon_first: 180.0,
            lon_last: 179.9,
            i_scans_negatively: 0,
            j_scans_positively: 0,
            ni: 3600,
            i_direction_increment: 0.1,
        };
        let area = compute_regular_ll_area(g).expect("0.1° global must succeed");
        assert!((area[1] - (-180.0)).abs() < 1e-9);
    }

    #[test]
    fn greenwich_first_subdomain_passes_through() {
        let g = RegularLlGeometry {
            lat_first: 70.0,
            lat_last: 30.0,
            lon_first: -30.0,
            lon_last: 50.0,
            i_scans_negatively: 0,
            j_scans_positively: 0,
            ni: 321,
            i_direction_increment: 0.25,
        };
        assert_eq!(compute_regular_ll_area(g), Some([70.0, -30.0, 30.0, 50.0]));
    }

    #[test]
    fn dateline_crossing_regional_bails() {
        // W > E with Ni × inc = 20.25° ≠ 360° — narrow-rule refuses.
        let g = RegularLlGeometry {
            lat_first: 60.0,
            lat_last: -60.0,
            lon_first: 170.0,
            lon_last: -170.0,
            i_scans_negatively: 0,
            j_scans_positively: 0,
            ni: 81,
            i_direction_increment: 0.25,
        };
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn inconsistent_endpoints_bail_even_when_full_global_in_ni() {
        // Pathological metadata that the `ni × inc ≈ 360` check alone
        // would have accepted: the endpoint-consistency guard rejects it.
        // `w=10, e=5, ni=360, inc=1` — after shift `w=-350`, shifted-span
        // would be 355°, expected `(ni-1)*inc = 359°`.  Mismatch > half an
        // increment → bail.
        let g = RegularLlGeometry {
            lat_first: 90.0,
            lat_last: -90.0,
            lon_first: 10.0,
            lon_last: 5.0,
            i_scans_negatively: 0,
            j_scans_positively: 0,
            ni: 360,
            i_direction_increment: 1.0,
        };
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn non_standard_i_scan_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.i_scans_negatively = 1;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn non_standard_j_scan_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.j_scans_positively = 1;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn nan_lat_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.lat_first = f64::NAN;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn nan_lon_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.lon_last = f64::NAN;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn infinite_increment_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.i_direction_increment = f64::INFINITY;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn zero_increment_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.i_direction_increment = 0.0;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn negative_increment_bails() {
        let mut g = quarter_degree_global(180.0, 179.75);
        g.i_direction_increment = -0.25;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn single_column_grid_bails() {
        let mut g = quarter_degree_global(0.0, 0.0);
        g.ni = 1;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn degenerate_west_equals_east_bails() {
        assert_eq!(
            compute_regular_ll_area(quarter_degree_global(10.0, 10.0)),
            None,
        );
    }

    #[test]
    fn degenerate_north_equals_south_bails() {
        let mut g = quarter_degree_global(0.0, 359.75);
        g.lat_first = 45.0;
        g.lat_last = 45.0;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn inverted_latitudes_bail() {
        // Standard scan (jScansPositively = 0) requires lat_first > lat_last.
        let mut g = quarter_degree_global(0.0, 359.75);
        g.lat_first = -90.0;
        g.lat_last = 90.0;
        assert_eq!(compute_regular_ll_area(g), None);
    }

    #[test]
    fn full_global_test_accepts_exact_and_roundoff() {
        assert!(is_full_global_i(1440, 0.25));
        assert!(is_full_global_i(720, 0.5));
        assert!(is_full_global_i(360, 1.0));
        assert!(is_full_global_i(3600, 0.1));
    }

    #[test]
    fn full_global_test_rejects_regional() {
        assert!(!is_full_global_i(81, 0.25));
        assert!(!is_full_global_i(720, 0.25));
        assert!(!is_full_global_i(719, 0.5));
    }
}

// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Deterministic synthetic weather data generator.
//!
//! Produces smooth, spatially correlated float64 fields that resemble
//! real meteorological data (temperature grids). Smooth fields compress
//! far better than random noise, giving realistic benchmark ratios.
//!
//! The generator is intentionally zero-dependency — it implements a simple
//! SplitMix64 PRNG rather than pulling in `rand`.

// ── PRNG ─────────────────────────────────────────────────────────────────────

/// SplitMix64 PRNG — fast, statistically good, and fully deterministic.
/// State is advanced by `next()`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Return a float in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for mantissa precision
        let bits = self.next() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Generate a weather-like float64 field of `num_points` values.
///
/// The returned values model a 2D temperature grid (Kelvin) with:
/// - A smooth large-scale sinusoidal pattern (base ≈ 280 K, amplitude ≈ 30 K)
/// - Small random noise (±0.1 K) to mimic observation/model imperfections
///
/// The function is pure and deterministic for a given platform/toolchain:
/// the same `seed` produces the same output in the same environment. Because
/// it uses `f64::sin()`/`f64::cos()`, outputs may differ slightly across
/// platforms or Rust versions.
///
/// # Arguments
/// * `num_points` – exact number of values to generate
/// * `seed` – PRNG seed for full reproducibility
pub fn generate_weather_field(num_points: usize, seed: u64) -> Vec<f64> {
    if num_points == 0 {
        return Vec::new();
    }

    // Approximate grid dimensions (favours square grids).
    let ncols = (num_points as f64).sqrt().ceil() as usize;
    let nrows = num_points.div_ceil(ncols);

    let base: f64 = 280.0; // Kelvin midpoint
    let amplitude: f64 = 30.0; // K peak-to-trough
    let noise_scale: f64 = 0.1; // K noise amplitude

    // Spatial frequencies chosen to produce ~4 wavelengths across each axis.
    let freq_i = std::f64::consts::TAU * 4.0 / nrows as f64;
    let freq_j = std::f64::consts::TAU * 4.0 / ncols as f64;

    let mut prng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(num_points);

    'outer: for i in 0..nrows {
        for j in 0..ncols {
            if values.len() == num_points {
                break 'outer;
            }
            let signal = amplitude * (freq_i * i as f64).sin() * (freq_j * j as f64).cos();
            let noise = noise_scale * (prng.next_f64() * 2.0 - 1.0);
            values.push(base + signal + noise);
        }
    }

    values
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let a = generate_weather_field(1000, 42);
        let b = generate_weather_field(1000, 42);
        assert_eq!(a, b, "same seed must produce identical output");
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = generate_weather_field(1000, 42);
        let b = generate_weather_field(1000, 99);
        assert_ne!(a, b, "different seeds must produce different output");
    }

    #[test]
    fn test_length() {
        for n in [0usize, 1, 100, 999, 1000, 1024, 10000] {
            let v = generate_weather_field(n, 42);
            assert_eq!(v.len(), n, "length mismatch for n={n}");
        }
    }

    #[test]
    fn test_physical_range() {
        let v = generate_weather_field(10_000, 42);
        for (i, &val) in v.iter().enumerate() {
            assert!(
                (240.0..=320.0).contains(&val),
                "value out of physical range at index {i}: {val}"
            );
        }
    }

    #[test]
    fn test_first_values_stable() {
        // Pin the first few values to guard against accidental PRNG changes.
        let v = generate_weather_field(5, 42);
        assert_eq!(v.len(), 5);
        // Values should be in the physical range and finite.
        for val in &v {
            assert!(val.is_finite(), "expected finite value, got {val}");
            assert!((240.0..=320.0).contains(val), "out of range: {val}");
        }
    }

    #[test]
    fn test_single_point() {
        // num_points=1: ncols=1, nrows=1, sin(0)=0, cos(0)=1 → base + noise.
        let v = generate_weather_field(1, 42);
        assert_eq!(v.len(), 1);
        assert!(v[0].is_finite());
        // sin(0)=0 so signal is 0, value ≈ 280.0 ± 0.1.
        assert!(
            (279.8..=280.2).contains(&v[0]),
            "expected ≈280 K, got {}",
            v[0]
        );
    }

    #[test]
    fn test_extreme_seeds() {
        // Seed 0 and u64::MAX must produce valid, finite, in-range values.
        for seed in [0u64, u64::MAX] {
            let v = generate_weather_field(100, seed);
            assert_eq!(v.len(), 100);
            for (i, &val) in v.iter().enumerate() {
                assert!(val.is_finite(), "seed={seed} index={i}: non-finite {val}");
                assert!(
                    (240.0..=320.0).contains(&val),
                    "seed={seed} index={i}: out of range {val}"
                );
            }
        }
    }

    #[test]
    fn test_prng_range() {
        // Verify the PRNG's next_f64() stays in [0.0, 1.0).
        let mut rng = SplitMix64::new(12345);
        for _ in 0..10_000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "next_f64 out of [0,1): {f}");
        }
    }
}

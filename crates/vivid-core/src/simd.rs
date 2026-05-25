// crates/vivid-core/src/simd.rs
#![cfg(feature = "nightly")]

use std::simd::num::SimdFloat;
use std::simd::Simd;

/// Override lane width to 16 (suitable for AVX-512 targets).
#[cfg(feature = "simd-lanes-16")]
const LANES: usize = 16;

/// Override lane width to 4 (suitable for SSE-only targets).
#[cfg(feature = "simd-lanes-4")]
const LANES: usize = 4;

/// Auto-detect lane width from target features.
///
/// - 16 on AVX-512 targets
/// - 8  on AVX/AVX2 targets
/// - 4  on SSE-only targets (default fallback)
#[cfg(not(any(feature = "simd-lanes-16", feature = "simd-lanes-4")))]
mod lanes {
    #[cfg(target_feature = "avx512f")]
    pub const LANES: usize = 16;

    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx"))]
    pub const LANES: usize = 8;

    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx")))]
    pub const LANES: usize = 4;
}

#[cfg(not(any(feature = "simd-lanes-16", feature = "simd-lanes-4")))]
use lanes::LANES;

/// Computes L2 (Euclidean) distance using manual SIMD instructions.
pub fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut diff_sum = Simd::<f32, LANES>::splat(0.0);

    let len = a.len();
    let len_rounded = len - (len % LANES);

    let mut i = 0;
    while i < len_rounded {
        let va = Simd::from_slice(&a[i..i + LANES]);
        let vb = Simd::from_slice(&b[i..i + LANES]);

        let diff = va - vb;
        diff_sum += diff * diff;

        i += LANES;
    }

    let mut sum = diff_sum.reduce_sum();

    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }

    sum.sqrt()
}

/// Computes cosine distance using manual SIMD instructions.
pub fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_sum = Simd::<f32, LANES>::splat(0.0);
    let mut norm_a_sum = Simd::<f32, LANES>::splat(0.0);
    let mut norm_b_sum = Simd::<f32, LANES>::splat(0.0);

    let len = a.len();
    let len_rounded = len - (len % LANES);

    let mut i = 0;
    while i < len_rounded {
        let va = Simd::from_slice(&a[i..i + LANES]);
        let vb = Simd::from_slice(&b[i..i + LANES]);

        dot_sum += va * vb;
        norm_a_sum += va * va;
        norm_b_sum += vb * vb;

        i += LANES;
    }

    let mut dot = dot_sum.reduce_sum();
    let mut norm_a = norm_a_sum.reduce_sum();
    let mut norm_b = norm_b_sum.reduce_sum();

    while i < len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let cosine_similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    1.0 - cosine_similarity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| { let d = x - y; d * d }).sum::<f32>().sqrt()
    }

    fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum();
        let nb: f32 = b.iter().map(|&x| x * x).sum();
        if na == 0.0 || nb == 0.0 { return 1.0; }
        1.0 - dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn test_simd_l2_matches_scalar() {
        for dim in [1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 64, 100] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.05).collect();
            let expected = scalar_l2(&a, &b);
            let got = l2_distance_simd(&a, &b);
            assert!((got - expected).abs() < 1e-5, "dim={}: expected {} got {}", dim, expected, got);
        }
    }

    #[test]
    fn test_simd_cosine_matches_scalar() {
        for dim in [1, 2, 3, 4, 7, 8, 15, 16, 17, 32, 64, 100] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.05).collect();
            let expected = scalar_cosine(&a, &b);
            let got = cosine_distance_simd(&a, &b);
            assert!((got - expected).abs() < 1e-5, "dim={}: expected {} got {}", dim, expected, got);
        }
    }

    #[test]
    fn test_simd_l2_tail_handling() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = scalar_l2(&a, &b);
        let got = l2_distance_simd(&a, &b);
        assert!((got - expected).abs() < 1e-5);
    }

    #[test]
    fn test_simd_cosine_tail_handling() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let expected = scalar_cosine(&a, &b);
        let got = cosine_distance_simd(&a, &b);
        assert!((got - expected).abs() < 1e-5);
    }

    #[test]
    fn test_simd_l2_identity() {
        let a = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        assert_eq!(l2_distance_simd(&a, &a), 0.0);
    }

    #[test]
    fn test_simd_cosine_identity() {
        let a = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        assert_eq!(cosine_distance_simd(&a, &a), 0.0);
    }

    #[test]
    fn test_simd_l2_symmetry() {
        let a: Vec<f32> = (0..20).map(|i| (i as f32) * 0.3).collect();
        let b: Vec<f32> = (0..20).map(|i| 10.0 - (i as f32) * 0.2).collect();
        assert!((l2_distance_simd(&a, &b) - l2_distance_simd(&b, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_distance_simd(&a, &b), 1.0);
        assert_eq!(cosine_distance_simd(&b, &a), 1.0);
        assert_eq!(cosine_distance_simd(&a, &a), 1.0);
    }
}

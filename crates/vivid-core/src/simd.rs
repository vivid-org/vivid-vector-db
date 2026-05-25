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

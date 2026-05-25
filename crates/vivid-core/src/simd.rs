// crates/vivid-core/src/simd.rs
#![cfg(feature = "nightly")]

use std::simd::num::SimdFloat;
use std::simd::Simd;

const LANES: usize = 8;

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

    // Main loop for SIMD lanes
    let mut i = 0;
    while i < len_rounded {
        // Load chunks safely using slice references
        let va = Simd::from_slice(&a[i..i + LANES]);
        let vb = Simd::from_slice(&b[i..i + LANES]);

        dot_sum += va * vb;
        norm_a_sum += va * va;
        norm_b_sum += vb * vb;

        i += LANES;
    }

    // Horizontal reduction
    let mut dot = dot_sum.reduce_sum();
    let mut norm_a = norm_a_sum.reduce_sum();
    let mut norm_b = norm_b_sum.reduce_sum();

    // Process the tail (remainder) sequentially
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

//! # Vector Metrics Performance Benchmarks
//!
//! This module measures the execution speed of vector metrics
//! under different compilation targets (Stable auto-vectorization vs Nightly `std::simd`).
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vivid_core::{CosineSpace, L2Space, VectorSpace};

/// Measures the execution time of cosine distance calculation for 1536-dimensional vectors.
pub fn bench_cosine_distance(c: &mut Criterion) {

    let dimension = 1536;
    let v1 = vec![0.5f32; dimension];
    let v2 = vec![0.2f32; dimension];

    c.bench_function("cosine_distance_1536", |b| {
        b.iter(|| {
            CosineSpace::distance(black_box(&v1), black_box(&v2)).unwrap()
        })
    });
}

/// Measures the execution time of L2 distance calculation for 1536-dimensional vectors.
pub fn bench_l2_distance(c: &mut Criterion) {
    let dimension = 1536;
    let v1 = vec![0.5f32; dimension];
    let v2 = vec![0.2f32; dimension];

    c.bench_function("l2_distance_1536", |b| {
        b.iter(|| {
            L2Space::distance(black_box(&v1), black_box(&v2)).unwrap()
        })
    });
}

criterion_group!(benches, bench_cosine_distance, bench_l2_distance);
criterion_main!(benches);

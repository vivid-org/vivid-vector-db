#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use std::hint::black_box as bb;
use std::path::Path;
use vivid_index::{FlatIndex, HnswIndex};

fn random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

fn build_hnsw(vecs: &[Vec<f32>]) -> HnswIndex {
    let dim = vecs[0].len();
    let mut index = HnswIndex::new(dim);
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v.clone()).unwrap();
    }
    index
}

// ── HNSW Insert ──

fn bench_hnsw_insert(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("hnsw_insert");
    for &n in &[100, 500] {
        group.bench_with_input(format!("{n}_128d"), &n, |b, &n| {
            b.iter_batched(
                || random_vectors(n, dim),
                |vecs| {
                    let mut index = HnswIndex::new(dim);
                    for (i, v) in vecs.into_iter().enumerate() {
                        bb(index.insert(bb(i as u64), bb(v)).unwrap());
                    }
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ── HNSW Search ──

fn bench_hnsw_search(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("hnsw_search");
    for &n in &[100, 1000, 5000] {
        let vecs = random_vectors(n, dim);
        let index = build_hnsw(&vecs);
        let query = random_vectors(1, dim).remove(0);

        group.bench_with_input(format!("{n}_128d_top10"), &n, |b, _| {
            b.iter(|| bb(index.search(bb(&query), bb(10)).unwrap()));
        });
    }
    group.finish();
}

// ── HNSW Batch Insert ──

fn bench_hnsw_batch_insert(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("hnsw_batch_insert");
    for &n in &[100, 500] {
        group.bench_with_input(format!("{n}_128d"), &n, |b, &n| {
            b.iter_batched(
                || {
                    let vecs = random_vectors(n, dim);
                    let items: Vec<(u64, Vec<f32>)> =
                        vecs.into_iter().enumerate().map(|(i, v)| (i as u64, v)).collect();
                    items
                },
                |items| {
                    let mut index = HnswIndex::new(dim);
                    bb(index.insert_batch(bb(&items)).unwrap());
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ── HNSW Update (in-place) ──

fn bench_hnsw_update(c: &mut Criterion) {
    let dim = 128;
    let n = 1000;
    let vecs = random_vectors(n, dim);
    let mut index = HnswIndex::new(dim);
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v.clone()).unwrap();
    }
    let new_vec = random_vectors(1, dim).remove(0);

    let mut group = c.benchmark_group("hnsw_update");
    group.bench_with_input("1000_128d", &(), |b, _| {
        b.iter(|| {
            bb(index.update(bb(0), bb(new_vec.clone())).unwrap());
        });
    });
    group.finish();
}

// ── FlatIndex Search (brute-force baseline) ──

fn bench_flat_search(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("flat_search");
    for &n in &[100, 1000, 5000] {
        let path = format!("__bench_flat_{n}.bin");
        let vecs = random_vectors(n, dim);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = FlatIndex::create(Path::new(&path), dim, &ids, &vecs).unwrap();
        let query = random_vectors(1, dim).remove(0);

        group.bench_with_input(format!("{n}_128d_top10"), &n, |b, _| {
            b.iter(|| bb(index.search(bb(&query), bb(10)).unwrap()));
        });
        let _ = std::fs::remove_file(&path);
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_hnsw_batch_insert,
    bench_hnsw_update,
    bench_flat_search,
);
criterion_main!(benches);

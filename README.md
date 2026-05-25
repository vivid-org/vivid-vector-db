# Vivid — High-Performance Vector Search Engine

Vivid is an embedded vector search engine written in Rust (Edition 2024) with Python bindings, optimized for AI/LLM infrastructure — RAG, semantic search, and similarity matching.

## Architecture

The project is a Cargo workspace of 4 decoupled crates:

| Crate | Description |
|---|---|
| **vivid-core** | Core vector mathematics — distance metrics (L2, Cosine), manual SIMD via `std::simd` |
| **vivid-index** | Indexing structures — brute-force `FlatIndex` for exact search, `HnswIndex` for approximate logarithmic search |
| **vivid-cli** | Async CLI for index management (upsert, search) |
| **vivid-python** | PyO3 native Python module |

## Features

- **SIMD-accelerated distance computation** — Manual `std::simd` portablesimd with 80% perf gain over auto-vectorization (~270 ns for L2, ~400 ns for Cosine on 1536-dim vectors)
- **Dual-index architecture** — Choose exact brute-force (`FlatIndex`) or approximate logarithmic (`HnswIndex`) search
- **Multithreaded search** — Rayon-powered parallelism across all CPU cores
- **Binary persistence** — Zero-copy-like serialization with `serde` + `bincode`
- **100% safe Rust** — `#![forbid(unsafe_code)]`, no `unsafe` blocks anywhere
- **Nightly/Stable dual build** — Optional `nightly` feature flag enables manual SIMD; stable builds use idiomatic iterators
- **Python bindings** — Native PyO3 extension module, installable via `maturin`

## Usage

### Rust

```rust
use vivid_index::{FlatIndex, SearchResult};

// Exact search (brute-force)
let mut index = FlatIndex::new(768);
index.insert(42, vec![0.5; 768]).unwrap();
let results = index.search(&[0.1; 768], 5).unwrap();
```

```rust
use vivid_index::hnsw::HnswIndex;

// Approximate search (logarithmic)
let mut index = HnswIndex::with_params(768, 16, 32, 200);
index.insert(42, vec![0.5; 768]).unwrap();
let results = index.search(&[0.1; 768], 5).unwrap();
```

### CLI

```bash
# Insert a vector
vivid upsert --index my_index.bin --dimension 768 --id 1 --vector "[0.1, 0.2, 0.3]"

# Search nearest neighbors
vivid search --index my_index.bin --query "[0.1, 0.2, 0.3]" --top-k 10
```

### Python

```python
import vivid

index = vivid.PyFlatIndex(1536)
index.insert(1, [0.1] * 1536)
hits = index.search([0.1] * 1536, top_k=5)
# hits = [{"id": 1, "score": 0.0}]
```

## Building

### Stable Rust

```bash
cargo build --workspace
```

### Nightly (SIMD acceleration)

Requires `rustup toolchain install nightly`.

```bash
cargo build --features nightly
```

### Python bindings

```bash
cd crates/vivid-python
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --features nightly
python test.py
```

## Index Types

### FlatIndex

Brute-force O(n) exact search. Best for:
- Datasets < 100K vectors
- When 100% recall is required
- Batch/offline workloads

### HnswIndex

Hierarchical Navigable Small World graph — O(log n) approximate search. Best for:
- Datasets > 100K vectors
- Low-latency online queries
- High-throughput RAG pipelines

**Parameters:**
- `m` (default: 16) — max connections per node per layer (except layer 0)
- `m_max` (default: 32) — max connections per node at layer 0
- `ef_construction` (default: 200) — candidate list size during construction; higher = better recall, slower build

## Distance Metrics

| Metric | Struct | SIMD (nightly) |
|---|---|---|
| Euclidean (L2) | `L2Space` | ✅ `l2_distance_simd` (~270 ns / 1536-dim) |
| Cosine | `CosineSpace` | ✅ `cosine_distance_simd` (~400 ns / 1536-dim) |

## Project Structure

```
vivid/
├── .cargo/config.toml          # Strict linting: deny unsafe, clippy pedantic, missing docs
├── crates/
│   ├── vivid-core/             # Vector math, distance metrics, SIMD engine
│   │   ├── src/simd.rs         # Manual SIMD (std::simd, nightly only)
│   │   └── benches/            # Criterion benchmarks
│   ├── vivid-index/            # FlatIndex, HnswIndex, persistence
│   │   └── src/hnsw.rs         # HNSW graph algorithm
│   ├── vivid-cli/              # Async CLI (clap + tokio)
│   └── vivid-python/           # PyO3 Python bindings
│       └── test.py
└── Cargo.toml                  # Workspace root
```

## Performance

Benchmarked with Criterion on 1536-dimensional vectors (nightly SIMD):

| Operation | Time |
|---|---|
| Cosine distance | ~400 ns |
| L2 distance | ~270 ns |
| HNSW search (top-5, 200 vectors) | ~5-15 µs |

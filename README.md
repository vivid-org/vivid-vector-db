# Vivid — High-Performance Vector Search Engine

Vivid is an embedded vector search engine written in Rust (Edition 2024) with Python bindings, optimised for AI/LLM infrastructure — RAG, semantic search, and similarity matching.

## Architecture

The project is a Cargo workspace of 4 decoupled crates:

| Crate | Description |
|---|---|
| **vivid-core** | Core vector mathematics — distance metrics (L2, Cosine), configurable SIMD via `std::simd` |
| **vivid-index** | Indexing structures — file-backed `FlatIndex` for exact search, `HnswIndex` for approximate logarithmic search |
| **vivid-cli** | CLI for index management (create, insert, update, delete, get, search, info) |
| **vivid-python** | PyO3 native Python module |

## Features

- **SIMD-accelerated distance computation** — Manual `std::simd` portable simd; auto-detects lane width from target features (16 on AVX-512, 8 on AVX, 4 on SSE); overridable via `simd-lanes-16` / `simd-lanes-4` features
- **Dual-index architecture** — File-backed exact brute-force (`FlatIndex`), or incremental approximate logarithmic (`HnswIndex`)
- **Multithreaded search** — Rayon-powered parallelism across all CPU cores
- **Binary persistence** — Combined file format (bincode graph + raw `f32` vectors) with zero-copy access via `bytemuck`
- **100% safe Rust** — `#![forbid(unsafe_code)]`, no `unsafe` blocks anywhere
- **Nightly/Stable dual build** — Optional `nightly` feature flag enables manual SIMD; stable builds use idiomatic iterators
- **Python bindings** — Native PyO3 extension module, installable via `maturin`
- **Duplicate ID detection** — `insert()` rejects duplicate IDs with `IndexError::DuplicateId`
- **Vector deletion / update** — `remove(id)` and `update(id, new_vector)` for live index mutation
- **ID lookup** — `get(id) -> Option<&[f32]>` for zero-copy vector retrieval
- **Bulk insert** — `insert_batch()` pre-allocates storage for efficient batch loading

## Usage

### Rust — HnswIndex (approximate search)

```rust
use vivid_index::HnswIndex;

let mut index = HnswIndex::with_params(768, 16, 32, 200);
index.insert(42, vec![0.5; 768]).unwrap();
index.insert(99, vec![0.1; 768]).unwrap();

let results = index.search(&[0.1; 768], 5).unwrap();
// results = [SearchResult { id: 99, score: 0.0 }, ...]

index.update(42, vec![0.2; 768]).unwrap();
index.remove(99).unwrap();

if let Some(vec) = index.get(42) {
    println!("{:?}", vec);
}
```

### Rust — FlatIndex (file-backed exact search)

```rust
use vivid_index::FlatIndex;

// Create a new index file from existing data
let index = FlatIndex::create("vectors.bin", 768, &[1, 2], &[vec![0.5; 768], vec![0.1; 768]]).unwrap();

// Open an existing file
let index = FlatIndex::open("vectors.bin").unwrap();

let results = index.search(&[0.1; 768], 5).unwrap();
```

### CLI

```bash
# Create an empty index
vivid-cli create -i my_index.bin -d 768

# Insert a vector (fails on duplicate ID)
vivid-cli insert -i my_index.bin -n 1 -v "[0.1, 0.2, 0.3]"

# Update a vector (fails if ID not found)
vivid-cli update -i my_index.bin -n 1 -v "[0.9, 0.8, 0.7]"

# Delete a vector
vivid-cli delete -i my_index.bin -n 1

# Lookup a vector by ID
vivid-cli get -i my_index.bin -n 1

# Search nearest neighbours
vivid-cli search -i my_index.bin -q "[0.1, 0.2, 0.3]" -k 10

# Upsert (insert or update)
vivid-cli upsert -i my_index.bin -d 768 -n 1 -v "[0.1, 0.2, 0.3]"

# Index statistics
vivid-cli info -i my_index.bin
```

| Flag | Long | Purpose |
|---|---|---|
| `-i` | `--index` | Path to index file |
| `-d` | `--dimension` | Vector dimension |
| `-n` | `--id` | Vector numeric ID |
| `-v` | `--vector` | Vector as JSON array |
| `-q` | `--query` | Query vector as JSON array |
| `-k` | `--top-k` | Number of results |
| `-f` | `--force` | Overwrite existing file |

### Python

```python
import vivid

index = vivid.PyVividIndex(1536)
index.insert(1, [0.1] * 1536)
index.insert(2, [0.9] * 1536)
hits = index.search([0.1] * 1536, top_k=5)
# hits = [{"id": 1, "score": 0.0}, {"id": 2, "score": 0.8}]

index.save_to_file("index.bin")
loaded = vivid.PyVividIndex.load_from_file("index.bin")
```

## Building

### Stable Rust

```bash
cargo build --workspace
```

### Nightly (SIMD acceleration — auto-detects lane width)

Requires `rustup toolchain install nightly`.

```bash
cargo build --features nightly
```

### Override SIMD lane width

```bash
# Force 16 lanes (AVX-512 class)
cargo build --features nightly,simd-lanes-16

# Force 4 lanes (SSE class)
cargo build --features nightly,simd-lanes-4
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

File-backed brute-force O(n) exact search. Read-only after creation — use `FlatIndex::create()` to build from existing data.

Best for:
- Datasets < 100K vectors
- When 100% recall is required
- Reference / ground-truth comparisons

### HnswIndex

Incremental Hierarchical Navigable Small World graph — O(log n) approximate search with `insert()`, `remove()`, `update()`, `get()`, and `insert_batch()`.

Best for:
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

## File Format

Both index types use a combined binary format optimised for zero-copy access:

**HnswIndex (VIDH):**
```
[4 bytes  magic: b"VIDH"]
[8 bytes  graph_size (u64 LE)]
[N bytes  bincode-encoded graph (topology, IDs, metadata)]
[0-3 byte padding (4-byte alignment)]
[rest    raw f32 vectors: num_vectors × dimension × 4]
```

**FlatIndex (VIDV):**
```
[4 bytes  magic: b"VIDV"]
[4 bytes  dimension (u32 LE)]
[8 bytes  num_vectors (u64 LE)]
[8×N     ids: [u64; num_vectors]]
[4×D×N   vectors: [f32; num_vectors × dimension]]
```

## Project Structure

```
vivid/
├── .cargo/config.toml          # Strict linting: deny unsafe, clippy pedantic, missing docs
├── crates/
│   ├── vivid-core/             # Vector math, distance metrics, SIMD engine
│   │   ├── src/simd.rs         # Manual SIMD (std::simd, nightly only)
│   │   └── benches/            # Criterion benchmarks
│   ├── vivid-index/            # FlatIndex, HnswIndex, persistence
│   │   └── src/hnsw.rs         # HNSW graph algorithm + file format
│   ├── vivid-cli/              # CLI (clap)
│   └── vivid-python/           # PyO3 Python bindings
└── Cargo.toml                  # Workspace root
```

## Performance

Benchmarked with Criterion on 1536-dimensional vectors (nightly SIMD):

| Operation | Time |
|---|---|
| Cosine distance | ~400 ns |
| L2 distance | ~270 ns |
| HNSW search (top-5, 200 vectors) | ~5-15 µs |

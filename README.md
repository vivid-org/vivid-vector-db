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
- **100% safe Rust** — `#![forbid(unsafe_code)]`, no `unsafe` blocks anywhere, verified by Clippy pedantic
- **Criterion benchmarks** — Full benchmark suite for distance metrics (stable + nightly SIMD) and index operations (insert, search, update, batch insert, FlatIndex search)
- **Nightly/Stable dual build** — Optional `nightly` feature flag enables manual SIMD; stable builds use idiomatic iterators
- **Python bindings** — Native PyO3 extension module, installable via `maturin`
- **Duplicate ID detection** — `insert()` rejects duplicate IDs with `IndexError::DuplicateId`
- **Vector deletion / update** — `remove(id)` and `update(id, new_vector)` for live index mutation
- **ID lookup** — `get(id) -> Option<&[f32]>` for zero-copy vector retrieval
- **Bulk insert** — `insert_batch()` pre-allocates storage for efficient batch loading
- **O(1) ID lookup** — `HashMap`-backed external-to-internal ID resolution (was O(n) linear scan)
- **In-place vector update** — `update()` overwrites raw bytes in `O(1)` without rebuilding the graph
- **Flat adjacency storage** — HNSW graph uses a CSR-like `Vec<u32>` + offsets + counts, eliminating per-node `Vec` heap allocations and 50% edge memory savings vs `Vec<Vec<Vec<usize>>>`

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
let mut index = FlatIndex::create("vectors.bin", 768, &[1, 2], &[vec![0.5; 768], vec![0.1; 768]]).unwrap();

// Open an existing file
let index = FlatIndex::open("vectors.bin").unwrap();

let results = index.search(&[0.1; 768], 5).unwrap();

// FlatIndex now supports full CRUD (insert, update, remove, get, contains)
index.insert(42, vec![0.3; 768]).unwrap();
index.update(1, vec![0.9; 768]).unwrap();
index.remove(2).unwrap();
```

### CLI

Index type is auto-detected from file magic bytes (`VIDH` → HNSW, `VIDV` → Flat). All mutation commands (insert, update, delete, upsert) work with both index types.

```bash
# Create an empty index (default: HNSW, use -t flat for exact search)
vivid-cli create -i my_index.bin -d 768
vivid-cli create -i my_flat.bin -d 768 -t flat

# Insert a vector (fails on duplicate ID)
vivid-cli insert -i my_index.bin -n 1 -v "[0.1, 0.2, 0.3]"

# Update a vector (fails if ID not found)
vivid-cli update -i my_index.bin -n 1 -v "[0.9, 0.8, 0.7]"

# Delete a vector
vivid-cli delete -i my_index.bin -n 1

# Lookup a vector by ID
vivid-cli get -i my_index.bin -n 1

# Search nearest neighbours (auto-detects index type from file magic bytes)
vivid-cli search -i my_index.bin -q "[0.1, 0.2, 0.3]" -k 10

# Upsert (insert or update — creates HNSW index if file doesn't exist)
vivid-cli upsert -i my_index.bin -d 768 -n 1 -v "[0.1, 0.2, 0.3]"

# Batch insert from JSON file (fails if any ID already exists)
vivid-cli batch-insert -i my_index.bin -f batch.json
# batch.json: [[1, [0.1, 0.2, 0.3]], [2, [0.4, 0.5, 0.6]]]

# Index statistics
vivid-cli info -i my_index.bin
```

| Flag | Long | Purpose |
|---|---|---|---|
| `-i` | `--index` | Path to index file |
| `-d` | `--dimension` | Vector dimension |
| `-n` | `--id` | Vector numeric ID |
| `-v` | `--vector` | Vector as JSON array |
| `-q` | `--query` | Query vector as JSON array |
| `-k` | `--top-k` | Number of results |
| `-t` | `--type` | Index type: `hnsw` (default) or `flat` (for `create`) |
| `-f` | `--file` | Path to JSON file (for `batch-insert`) |
| `-f` | `--force` | Overwrite existing file (for `create`) |

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

### Release (LTO-optimised)

```bash
cargo build --release --workspace
# Release binary: ~674 KB, with LTO, panic=abort, stripped
```

### Benchmarks

```bash
# Distance metrics (vivid-core)
cargo bench -p vivid-core

# Distance metrics with SIMD (nightly)
cargo bench -p vivid-core --features nightly

# Index operations (vivid-index — insert, search, update, batch, flat search)
cargo bench -p vivid-index

# HTML reports available at target/criterion/report/
```

### Nightly (SIMD acceleration — auto-detects lane width)

Requires `rustup toolchain install nightly`.

```bash
cargo build --features nightly
```

### Override SIMD lane width

```bash
# Force 4 lanes (SSE class)
cargo build --features nightly,simd-lanes-4
```

### CLI Integration Tests

```powershell
# Run full CLI test suite (48 tests across both HNSW and Flat index types)
.\local-tests\run_cli_tests.ps1
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

File-backed brute-force O(n) exact search. Supports full CRUD after creation (insert, update, remove, get, contains, create, open, save_to_file).

Best for:
- Datasets < 100K vectors
- When 100% recall is required
- Reference / ground-truth comparisons

### HnswIndex

Incremental Hierarchical Navigable Small World graph — O(log n) approximate search with `insert()`, `remove()`, `update()`, `get()`, and `insert_batch()`. `update()` overwrites vector data in-place (`O(1)`) without rebuilding graph connections.

Best for:
- Datasets > 100K vectors
- Low-latency online queries
- High-throughput RAG pipelines

**Parameters:**
- `m` (default: 16) — max connections per node per layer (except layer 0)
- `m_max` (default: 32) — max connections per node at layer 0
- `ef_construction` (default: 200) — candidate list size during construction; higher = better recall, slower build

## Distance Metrics

| Metric | Struct | Stable | Nightly SIMD | Speedup |
|---|---|---|---|---|
| Euclidean (L2) | `L2Space` | ~1.89 µs | ~498 ns | ~3.8× |
| Cosine | `CosineSpace` | ~1.93 µs | ~503 ns | ~3.8× |

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
├── local-tests/                # CLI integration test script (PowerShell)
├── crates/
│   ├── vivid-core/             # Vector math, distance metrics, SIMD engine
│   │   ├── src/simd.rs         # Manual SIMD (std::simd, nightly only)
│   │   └── benches/            # Criterion benchmarks
│   ├── vivid-index/            # FlatIndex, HnswIndex, persistence
│   │   ├── benches/            # Criterion benchmarks (insert/search/update/batch)
│   │   └── src/hnsw.rs         # HNSW graph algorithm + file format
│   ├── vivid-cli/              # CLI (clap)
│   └── vivid-python/           # PyO3 Python bindings
└── Cargo.toml                  # Workspace root
```

## Performance

### Distance Metrics (1536-dim, nightly SIMD)

| Operation | Time |
|---|---|
| Cosine distance | ~503 ns (~3.8× vs stable iterators) |
| L2 distance | ~498 ns (~3.8× vs stable iterators) |

### Index Operations (128-dim, release build)

All benchmarks measured via Criterion on 128-dimensional random vectors.

| Operation | Scale | Time |
|---|---|---|
| **HNSW insert** | 100 vectors | 18.6 ms (186 µs/vec) |
| | 500 vectors | 195 ms (390 µs/vec) |
| **HNSW batch insert** | 100 vectors | 20.2 ms |
| | 500 vectors | 198 ms |
| **HNSW search** (top-10) | 100 vectors | 94 µs |
| | 1,000 vectors | 278 µs |
| | 5,000 vectors | 590 µs |
| **HNSW update** (in-place) | 1,000 vectors | **118 ns** |
| **FlatIndex search** (brute-force, top-10) | 100 vectors | 44 µs |
| | 1,000 vectors | 202 µs |
| | 5,000 vectors | 588 µs |

HNSW update is a constant-time memcpy (~118 ns at 128-dim), independent of index size. HNSW search scales sub-linearly with vector count but requires larger datasets (>100K) to fully exploit logarithmic advantages over brute-force FlatIndex.

//! HNSW (Hierarchical Navigable Small World) graph index.
//!
//! Implementation of the algorithm described in:
//! "Efficient and robust approximate nearest neighbor search using
//!  Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2016).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use bytemuck::cast_slice;
use rand::Rng;
use serde::{Deserialize, Serialize};
use vivid_core::{CosineSpace, VectorError, VectorSpace};

use crate::{IndexError, SearchResult, VectorId};

const DEFAULT_M: usize = 16;
const DEFAULT_M_MAX: usize = 32;
const DEFAULT_EF_CONSTRUCTION: usize = 200;

const VIDH_MAGIC: [u8; 4] = *b"VIDH";

const fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Flat-packed adjacency layer for HNSW.
///
/// Stores all neighbor relationships in a single `Vec<u32>` with per-node
/// offsets and counts, eliminating per-node `Vec` heap allocations and
/// capacity slack. Node indices use `u32` (up to ~4B nodes).
#[derive(Serialize, Deserialize, Clone)]
struct AdjLayer {
    /// Flat array of all neighbor IDs in this layer.
    neighbors: Vec<u32>,
    /// Start index in `neighbors` for each node's neighbor list.
    offsets: Vec<u32>,
    /// Neighbor count per node. Length = `offsets.len()`.
    /// Node i's neighbors: `neighbors[offsets[i]..offsets[i] + counts[i] as u32]`
    counts: Vec<u16>,
}

impl AdjLayer {
    /// Creates a new layer with `num_nodes` empty neighbor lists.
    fn with_nodes(num_nodes: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            offsets: vec![0; num_nodes],
            counts: vec![0; num_nodes],
        }
    }

    /// Returns the neighbors of `node` as a slice.
    fn nbrs(&self, node: usize) -> &[u32] {
        let start = self.offsets[node] as usize;
        let end = start + self.counts[node] as usize;
        &self.neighbors[start..end]
    }

    /// Adds a neighbor to `node`'s neighbor list.
    fn add_nbr(&mut self, node: usize, nbr: u32) {
        let insert = self.offsets[node] as usize + self.counts[node] as usize;
        self.neighbors.insert(insert, nbr);
        self.counts[node] += 1;
        for o in &mut self.offsets[node + 1..] {
            *o += 1;
        }
    }

    /// Replaces the neighbor list of `node` with `new_nbrs`.
    fn set_nbrs(&mut self, node: usize, new_nbrs: &[u32]) {
        let start = self.offsets[node] as usize;
        let old_count = self.counts[node] as usize;
        let delta = new_nbrs.len() as i64 - old_count as i64;

        self.neighbors.splice(start..start + old_count, new_nbrs.iter().copied());
        self.counts[node] = new_nbrs.len() as u16;

        if delta != 0 {
            for o in &mut self.offsets[node + 1..] {
                *o = (*o as i64 + delta) as u32;
            }
        }
    }

    /// Appends an empty neighbor list for a new node.
    fn push_empty(&mut self) {
        let end = match (self.offsets.last(), self.counts.last()) {
            (Some(&off), Some(&cnt)) => off + cnt as u32,
            _ => 0,
        };
        self.offsets.push(end);
        self.counts.push(0);
    }

    /// Removes node at `pos` and decrements all neighbor references > `pos`.
    /// O(E) where E = total edges in this layer.
    fn remove_node(&mut self, pos: usize) {
        // Collect all (adjusted_src, adjusted_target) edges except those involving `pos`.
        let mut edges: Vec<(usize, u32)> = Vec::with_capacity(self.neighbors.len());
        for src in 0..self.counts.len() {
            if src == pos {
                continue;
            }
            let start = self.offsets[src] as usize;
            let end = start + self.counts[src] as usize;
            let adjusted_src = if src > pos { src - 1 } else { src };
            for &tgt in &self.neighbors[start..end] {
                if tgt != pos as u32 {
                    let adjusted_tgt = if tgt > pos as u32 { tgt - 1 } else { tgt };
                    edges.push((adjusted_src, adjusted_tgt));
                }
            }
        }

        let num_nodes = self.counts.len() - 1;
        let mut new_offsets = Vec::with_capacity(num_nodes);
        let mut new_counts = Vec::with_capacity(num_nodes);
        let mut new_neighbors = Vec::with_capacity(edges.len());

        for src in 0..num_nodes {
            new_offsets.push(new_neighbors.len() as u32);
            let cnt = edges.iter().filter(|(s, _)| *s == src).count();
            new_counts.push(cnt as u16);
            for (s, t) in &edges {
                if *s == src {
                    new_neighbors.push(*t);
                }
            }
        }

        self.offsets = new_offsets;
        self.counts = new_counts;
        self.neighbors = new_neighbors;
    }
}

/// A candidate node with its distance to a query vector.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) node_id: usize,
    pub(crate) distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Serializable portion of the HNSW index (graph topology, metadata, IDs).
#[derive(Serialize, Deserialize)]
struct GraphData {
    dimension: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    ml: f32,
    ids: Vec<VectorId>,
    levels: Vec<usize>,
    adjacency: Vec<AdjLayer>,
    max_layer: usize,
    entry_point: Option<usize>,
}

/// HNSW (Hierarchical Navigable Small World) graph index.
///
/// Provides approximate nearest neighbor search with logarithmic time complexity.
///
/// Vectors are stored as raw `f32` bytes in a `Vec<u8>` for zero-copy access via
/// `bytemuck`. The file format combines a bincode-encoded graph section with raw
/// vector data, matching the on-disk layout of the mmap-backed index.
///
/// # Parameters
/// - `m`: Maximum number of connections per element per layer (default: 16)
/// - `m_max`: Maximum number of connections for the bottom layer (default: 32)
/// - `ef_construction`: Size of the dynamic candidate list during construction (default: 200)
pub struct HnswIndex {
    dimension: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    ml: f32,
    vector_data: Vec<u8>,
    ids: Vec<VectorId>,
    id_to_node: HashMap<VectorId, usize>,
    adjacency: Vec<AdjLayer>,
    levels: Vec<usize>,
    max_layer: usize,
    entry_point: Option<usize>,
}

impl HnswIndex {
    /// Creates a new empty `HnswIndex` with the specified vector dimension and default parameters.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self::with_params(dimension, DEFAULT_M, DEFAULT_M_MAX, DEFAULT_EF_CONSTRUCTION)
    }

    /// Creates a new empty `HnswIndex` with custom parameters.
    #[must_use]
    pub fn with_params(
        dimension: usize,
        m: usize,
        m_max: usize,
        ef_construction: usize,
    ) -> Self {
        Self {
            dimension,
            m,
            m_max,
            ef_construction,
            ml: 1.0 / (m as f32).ln(),
            vector_data: Vec::new(),
            ids: Vec::new(),
            id_to_node: HashMap::new(),
            adjacency: Vec::new(),
            levels: Vec::new(),
            max_layer: 0,
            entry_point: None,
        }
    }

    /// Returns the i-th vector as a slice.
    #[inline]
    fn vector_at(&self, index: usize) -> &[f32] {
        let start = index * self.dimension;
        let bytes = &self.vector_data[start * 4..(start + self.dimension) * 4];
        cast_slice(bytes)
    }

    /// Returns the internal node index for a given external ID.
    fn node_id_by_id(&self, id: VectorId) -> Option<usize> {
        self.id_to_node.get(&id).copied()
    }

    /// Returns `true` if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        self.node_id_by_id(id).is_some()
    }

    /// Retrieves the vector associated with the given ID, if it exists.
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<&[f32]> {
        let pos = self.node_id_by_id(id)?;
        Some(self.vector_at(pos))
    }

    /// Inserts a vector into the index with a specific ID.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DuplicateId`] if the ID already exists.
    /// Returns [`IndexError::DimensionMismatch`] if the vector length is invalid.
    pub fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError> {
        if self.contains(id) {
            return Err(IndexError::DuplicateId(id));
        }
        if vector.is_empty() {
            return Err(IndexError::EmptyVector);
        }
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                found: vector.len(),
            });
        }

        let new_level = self.random_level();
        let node_id = self.ids.len();

        let vec_bytes: &[u8] = cast_slice(vector.as_slice());
        self.vector_data.extend_from_slice(vec_bytes);
        self.ids.push(id);
        self.id_to_node.insert(id, node_id);
        self.levels.push(new_level);

        for layer in &mut self.adjacency {
            layer.push_empty();
        }

        while self.adjacency.len() <= new_level {
            self.adjacency.push(AdjLayer::with_nodes(self.ids.len()));
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = new_level;
            return Ok(());
        }

        let owned_query = self.vector_at(node_id).to_vec();
        let mut curr_entry = self.entry_point.unwrap();

        for level in (new_level + 1..=self.max_layer).rev() {
            let candidates = self
                .search_layer(&owned_query, &[curr_entry], level, 1)
                .expect("search_layer should not fail with valid vectors");
            curr_entry = candidates[0].node_id;
        }

        let top_layer = new_level.min(self.max_layer);
        for level in (0..=top_layer).rev() {
            let candidates = self
                .search_layer(&owned_query, &[curr_entry], level, self.ef_construction)
                .expect("search_layer should not fail with valid vectors");

            let m_curr = if level == 0 { self.m_max } else { self.m };
            let neighbors = Self::select_neighbors_simple(&candidates, m_curr);

            for &neighbor in &neighbors {
                self.adjacency[level].add_nbr(node_id, neighbor as u32);
            }

            for &neighbor in &neighbors {
                self.adjacency[level].add_nbr(neighbor, node_id as u32);
            }

            for &neighbor in &neighbors {
                let limit = if level == 0 { self.m_max } else { self.m };
                if self.adjacency[level].counts[neighbor] as usize > limit {
                    let new_nbrs = {
                        let node_vec = self.vector_at(neighbor);
                        let mut cand: Vec<Candidate> = self.adjacency[level]
                            .nbrs(neighbor)
                            .iter()
                            .map(|&n| {
                                let dist = CosineSpace::distance(node_vec, self.vector_at(n as usize))
                                    .expect("vectors share the same dimension");
                                Candidate { node_id: n as usize, distance: dist }
                            })
                            .collect();
                        cand.sort_unstable_by(|a, b| {
                            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        cand.truncate(limit);
                        cand.into_iter().map(|c| c.node_id as u32).collect::<Vec<_>>()
                    };
                    self.adjacency[level].set_nbrs(neighbor, &new_nbrs);
                }
            }

            curr_entry = candidates[0].node_id;
        }

        if new_level > self.max_layer {
            self.max_layer = new_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Inserts multiple vectors in batch, pre-allocating storage to minimise reallocations.
    ///
    /// Each batch call still builds the graph incrementally and honors uniqueness:
    /// if any ID in the batch already exists, the entire batch is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DuplicateId`] if any ID already exists.
    /// Returns [`IndexError::DimensionMismatch`] if any vector length is invalid.
    pub fn insert_batch(&mut self, items: &[(VectorId, Vec<f32>)]) -> Result<(), IndexError> {
        for (id, _) in items {
            if self.contains(*id) {
                return Err(IndexError::DuplicateId(*id));
            }
        }

        self.vector_data.reserve(items.len() * self.dimension * 4);
        self.ids.reserve(items.len());
        self.levels.reserve(items.len());

        for (id, vector) in items {
            self.insert(*id, vector.clone())?;
        }
        Ok(())
    }

    /// Removes a vector from the index by its ID.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::IdNotFound`] if the ID is not present.
    pub fn remove(&mut self, id: VectorId) -> Result<(), IndexError> {
        let pos = self.node_id_by_id(id).ok_or(IndexError::IdNotFound(id))?;

        self.ids.remove(pos);

        let byte_start = pos * self.dimension * 4;
        self.vector_data.drain(byte_start..byte_start + self.dimension * 4);

        self.levels.remove(pos);

        for layer in &mut self.adjacency {
            layer.remove_node(pos);
        }

        self.fix_entry_point_after_removal(pos);

        self.id_to_node = self.ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();

        Ok(())
    }

    /// Replaces the vector for an existing ID in-place (graph structure is preserved).
    ///
    /// Only the raw vector bytes in `vector_data` are overwritten — the graph
    /// topology (neighbor lists) is left unchanged. This is O(1) and avoids the
    /// cost of tearing down and rebuilding graph connections.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::IdNotFound`] if the ID is not present.
    pub fn update(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError> {
        let pos = self.node_id_by_id(id).ok_or(IndexError::IdNotFound(id))?;
        if vector.is_empty() {
            return Err(IndexError::EmptyVector);
        }
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                found: vector.len(),
            });
        }

        let byte_start = pos * self.dimension * 4;
        let bytes = cast_slice(vector.as_slice());
        self.vector_data[byte_start..byte_start + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn fix_entry_point_after_removal(&mut self, removed_pos: usize) {
        if self.ids.is_empty() {
            self.entry_point = None;
            self.max_layer = 0;
            return;
        }

        if let Some(ep) = self.entry_point {
            if ep == removed_pos {
                self.entry_point = Some(0);
                self.max_layer = *self.levels.iter().max().unwrap_or(&0);
            } else if ep > removed_pos {
                self.entry_point = Some(ep - 1);
            }
        }
    }

    /// Searches for the top-K nearest neighbours using the HNSW algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`VectorError::EmptyVector`] if the query vector is empty.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, VectorError> {
        if self.ids.is_empty() {
            return Ok(Vec::new());
        }

        let ef = top_k.max(100);
        let mut curr_entry = self.entry_point.unwrap();

        for level in (1..=self.max_layer).rev() {
            let candidates = self.search_layer(query, &[curr_entry], level, 1)?;
            curr_entry = candidates[0].node_id;
        }

        let candidates = self.search_layer(query, &[curr_entry], 0, ef)?;

        Ok(candidates
            .into_iter()
            .take(top_k)
            .map(|c| SearchResult {
                id: self.ids[c.node_id],
                score: c.distance,
            })
            .collect())
    }

    /// Searches a single layer for the `ef` nearest neighbours starting from `entry_points`.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        layer: usize,
        ef: usize,
    ) -> Result<Vec<Candidate>, VectorError> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut result = BinaryHeap::new();

        for &ep in entry_points {
            let dist = CosineSpace::distance(query, self.vector_at(ep))?;
            let cand = Candidate { node_id: ep, distance: dist };
            candidates.push(Reverse(cand.clone()));
            result.push(cand);
            visited.insert(ep);
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let furthest = result.peek().ok_or(VectorError::EmptyVector)?;
            if c.distance > furthest.distance {
                break;
            }

            for &neighbor in self.adjacency[layer].nbrs(c.node_id) {
                let nb = neighbor as usize;
                if visited.insert(nb) {
                    let dist = CosineSpace::distance(query, self.vector_at(nb))?;

                    if result.len() < ef {
                        let cand = Candidate { node_id: nb, distance: dist };
                        candidates.push(Reverse(cand.clone()));
                        result.push(cand);
                    } else {
                        let furthest = result.peek().ok_or(VectorError::EmptyVector)?;
                        if dist < furthest.distance {
                            let cand = Candidate { node_id: nb, distance: dist };
                            candidates.push(Reverse(cand.clone()));
                            result.push(cand);
                            result.pop();
                        }
                    }
                }
            }
        }

        Ok(result.into_sorted_vec())
    }

    fn select_neighbors_simple(candidates: &[Candidate], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|c| c.node_id).collect()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        (-r.ln() * self.ml as f64).floor() as usize
    }

    /// Saves the index to a binary file in the combined VIDH format
    /// (bincode graph + padding + raw `f32` vectors).
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError> {
        let graph = GraphData {
            dimension: self.dimension,
            m: self.m,
            m_max: self.m_max,
            ef_construction: self.ef_construction,
            ml: self.ml,
            ids: self.ids.clone(),
            levels: self.levels.clone(),
            adjacency: self.adjacency.clone(),
            max_layer: self.max_layer,
            entry_point: self.entry_point,
        };

        let graph_bytes = bincode::serialize(&graph)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;

        let raw_end = 12 + graph_bytes.len();
        let aligned_start = align_up(raw_end, 4);
        let padding = aligned_start - raw_end;

        let file = std::fs::File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&VIDH_MAGIC)?;
        writer.write_all(&(graph_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&graph_bytes)?;

        for _ in 0..padding {
            writer.write_all(&[0])?;
        }

        writer.write_all(&self.vector_data)?;
        writer.flush()?;
        Ok(())
    }

    /// Loads the index from a binary file created by [`save_to_file`](Self::save_to_file)
    /// (VIDH combined format).
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, IndexError> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let file_size = file.metadata()?.len() as usize;

        let mut data = Vec::with_capacity(file_size);
        file.read_to_end(&mut data)?;

        if data.len() < 12 {
            return Err(IndexError::Serialization("file too small for header".into()));
        }

        if data[0..4] != VIDH_MAGIC {
            return Err(IndexError::Serialization("invalid magic bytes".into()));
        }

        let gs_bytes: [u8; 8] = data[4..12].try_into().unwrap();
        let graph_size = u64::from_le_bytes(gs_bytes) as usize;

        let graph_end = 12 + graph_size;
        if data.len() < graph_end {
            return Err(IndexError::Serialization("file too small for graph data".into()));
        }

        let graph: GraphData = bincode::deserialize(&data[12..graph_end])
            .map_err(|e| IndexError::Serialization(e.to_string()))?;

        let vector_data_start = align_up(graph_end, 4);
        let expected_vec_bytes = graph.ids.len() * graph.dimension * 4;
        if data.len() < vector_data_start + expected_vec_bytes {
            return Err(IndexError::Serialization("file too small for vector data".into()));
        }

        let vector_data = data[vector_data_start..vector_data_start + expected_vec_bytes].to_vec();

        let id_to_node = graph.ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();

        Ok(Self {
            dimension: graph.dimension,
            m: graph.m,
            m_max: graph.m_max,
            ef_construction: graph.ef_construction,
            ml: graph.ml,
            vector_data,
            ids: graph.ids,
            id_to_node,
            levels: graph.levels,
            adjacency: graph.adjacency,
            max_layer: graph.max_layer,
            entry_point: graph.entry_point,
        })
    }

    /// Returns the vector dimension of the index.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the total number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Checks if the index contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_search_basic() {
        let mut index = HnswIndex::with_params(3, 8, 16, 100);
        index.insert(101, vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(102, vec![0.0, 1.0, 0.0]).unwrap();
        index.insert(103, vec![1.0, 1.0, 0.0]).unwrap();

        let query = [0.9, 0.1, 0.0];
        let hits = index.search(&query, 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 101);
    }

    #[test]
    fn test_hnsw_search_exact() {
        let mut index = HnswIndex::with_params(2, 16, 32, 200);
        index.insert(10, vec![1.0, 0.0]).unwrap();
        index.insert(20, vec![0.0, 1.0]).unwrap();
        index.insert(30, vec![1.0, 1.0]).unwrap();

        let query = [1.0, 0.0];
        let hits = index.search(&query, 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert!(hits[0].score < 0.001);
        assert_eq!(hits[0].id, 10);
    }

    #[test]
    fn test_hnsw_empty() {
        let index = HnswIndex::new(3);
        let hits = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_hnsw_accuracy_against_flat() {
        let dir = std::env::temp_dir();
        let path = dir.join("vivid_hnsw_acc_test.bin");

        let dim = 32;
        let n = 200;
        let top_k = 5;

        let mut hnsw = HnswIndex::with_params(dim, 16, 32, 200);

        let mut rng = rand::thread_rng();
        let mut flat_ids = Vec::new();
        let mut flat_vectors = Vec::new();

        for i in 0..n {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            hnsw.insert(i as u64, vec.clone()).unwrap();
            flat_ids.push(i as u64);
            flat_vectors.push(vec);
        }

        let flat = crate::FlatIndex::create(&path, dim, &flat_ids, &flat_vectors).unwrap();

        let mut total_recall = 0.0f64;
        for _ in 0..20 {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let hnsw_hits = hnsw.search(&query, top_k).unwrap();
            let flat_hits = flat.search(&query, top_k).unwrap();

            let hnsw_ids: Vec<u64> = hnsw_hits.iter().map(|h| h.id).collect();
            let flat_ids: Vec<u64> = flat_hits.iter().map(|h| h.id).collect();

            let overlap = hnsw_ids.iter().filter(|id| flat_ids.contains(id)).count();
            total_recall += overlap as f64 / top_k as f64;
        }

        let avg_recall = total_recall / 20.0;
        assert!(avg_recall > 0.90, "HNSW recall too low: {avg_recall:.3}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hnsw_persistence() {
        let dir = std::env::temp_dir();
        let path = dir.join("vivid_hnsw_persist.bin");

        let mut original = HnswIndex::with_params(3, 8, 16, 100);
        original.insert(42, vec![0.1, 0.9, 0.0]).unwrap();
        original.insert(99, vec![0.8, 0.2, 0.0]).unwrap();

        original.save_to_file(&path).unwrap();
        let loaded = HnswIndex::load_from_file(&path).unwrap();

        assert_eq!(loaded.len(), original.len());

        let query = [0.15, 0.85, 0.0];
        let original_hits = original.search(&query, 1).unwrap();
        let loaded_hits = loaded.search(&query, 1).unwrap();
        assert_eq!(original_hits[0].id, loaded_hits[0].id);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut index = HnswIndex::new(3);
        index.insert(1, vec![0.1, 0.2, 0.3]).unwrap();
        let err = index.insert(1, vec![0.4, 0.5, 0.6]).unwrap_err();
        assert!(matches!(err, IndexError::DuplicateId(1)));
    }

    #[test]
    fn test_get_vector_by_id() {
        let mut index = HnswIndex::new(3);
        index.insert(42, vec![0.1, 0.2, 0.3]).unwrap();
        let v = index.get(42);
        assert!(v.is_some());
        assert!((v.unwrap()[0] - 0.1).abs() < 1e-6);
        assert!(index.get(99).is_none());
    }

    #[test]
    fn test_remove_vector() {
        let mut index = HnswIndex::with_params(3, 8, 16, 100);
        index.insert(10, vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(20, vec![0.0, 1.0, 0.0]).unwrap();
        index.insert(30, vec![0.0, 0.0, 1.0]).unwrap();

        assert_eq!(index.len(), 3);
        index.remove(20).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.get(20).is_none());
        assert!(index.get(10).is_some());
        assert!(index.get(30).is_some());

        let hits = index.search(&[0.0, 1.0, 0.0], 2).unwrap();
        assert!(hits.iter().any(|r| r.id == 10 || r.id == 30));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut index = HnswIndex::new(3);
        index.insert(1, vec![0.1, 0.2, 0.3]).unwrap();
        let err = index.remove(99).unwrap_err();
        assert!(matches!(err, IndexError::IdNotFound(99)));
    }

    #[test]
    fn test_update_vector() {
        let mut index = HnswIndex::with_params(2, 8, 16, 100);
        index.insert(1, vec![0.0, 1.0]).unwrap();
        index.update(1, vec![1.0, 0.0]).unwrap();

        let hits = index.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].score < 0.001);
    }

    #[test]
    fn test_update_nonexistent() {
        let mut index = HnswIndex::new(2);
        let err = index.update(99, vec![0.1, 0.2]).unwrap_err();
        assert!(matches!(err, IndexError::IdNotFound(99)));
    }

    #[test]
    fn test_insert_batch() {
        let mut index = HnswIndex::with_params(3, 8, 16, 100);
        let items = vec![
            (10, vec![1.0, 0.0, 0.0]),
            (20, vec![0.0, 1.0, 0.0]),
            (30, vec![0.0, 0.0, 1.0]),
        ];
        index.insert_batch(&items).unwrap();
        assert_eq!(index.len(), 3);

        let hits = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 10);
    }

    #[test]
    fn test_insert_batch_duplicate_rejected() {
        let mut index = HnswIndex::new(3);
        index.insert(10, vec![1.0, 0.0, 0.0]).unwrap();

        let items = vec![
            (20, vec![0.0, 1.0, 0.0]),
            (10, vec![0.0, 0.0, 1.0]),
        ];
        let err = index.insert_batch(&items).unwrap_err();
        assert!(matches!(err, IndexError::DuplicateId(10)));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_contains() {
        let mut index = HnswIndex::new(2);
        assert!(!index.contains(42));
        index.insert(42, vec![0.1, 0.2]).unwrap();
        assert!(index.contains(42));
    }

    #[test]
    fn test_remove_last_vector() {
        let mut index = HnswIndex::new(2);
        index.insert(1, vec![0.1, 0.2]).unwrap();
        index.remove(1).unwrap();
        assert!(index.is_empty());
        assert!(index.search(&[0.1, 0.2], 5).unwrap().is_empty());
    }
}

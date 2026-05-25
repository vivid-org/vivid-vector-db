//! HNSW (Hierarchical Navigable Small World) graph index.
//!
//! Implementation of the algorithm described in:
//! "Efficient and robust approximate nearest neighbor search using
//!  Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2016).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

use rand::Rng;
use serde::{Deserialize, Serialize};
use vivid_core::{CosineSpace, VectorError, VectorSpace};

use crate::{IndexError, SearchResult, VectorId};

const DEFAULT_M: usize = 16;
const DEFAULT_M_MAX: usize = 32;
const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// A candidate node with its distance to a query vector.
#[derive(Clone, Debug)]
struct Candidate {
    node_id: usize,
    distance: f32,
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

/// HNSW (Hierarchical Navigable Small World) graph index.
///
/// Provides approximate nearest neighbor search with logarithmic time complexity,
/// replacing the brute-force O(n) FlatIndex for large-scale vector search.
///
/// # Parameters
/// - `m`: Maximum number of connections per element per layer (default: 16)
/// - `m_max`: Maximum number of connections for the bottom layer (default: 32)
/// - `ef_construction`: Size of the dynamic candidate list during construction (default: 200)
#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    dimension: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    ml: f32,
    vectors: Vec<Vec<f32>>,
    ids: Vec<VectorId>,
    /// adjacency[layer][node_id] = list of neighbor node_ids
    adjacency: Vec<Vec<Vec<usize>>>,
    /// The highest layer each node belongs to
    levels: Vec<usize>,
    /// Current top-most layer in the graph
    max_layer: usize,
    /// Entry point node ID for search traversal
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
            vectors: Vec::new(),
            ids: Vec::new(),
            adjacency: Vec::new(),
            levels: Vec::new(),
            max_layer: 0,
            entry_point: None,
        }
    }

    /// Inserts a vector into the index with a specific ID.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DimensionMismatch`] if the vector length is invalid.
    pub fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError> {
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
        let node_id = self.vectors.len();

        self.vectors.push(vector);
        self.ids.push(id);
        self.levels.push(new_level);

        for layer in &mut self.adjacency {
            layer.push(Vec::new());
        }

        while self.adjacency.len() <= new_level {
            let mut layer = Vec::with_capacity(self.vectors.len());
            for _ in 0..self.vectors.len() {
                layer.push(Vec::new());
            }
            self.adjacency.push(layer);
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = new_level;
            return Ok(());
        }

        let query = &self.vectors[node_id];
        let mut curr_entry = self.entry_point.unwrap();

        for level in (new_level + 1..=self.max_layer).rev() {
            let candidates = self
                .search_layer(query, &[curr_entry], level, 1)
                .expect("search_layer should not fail with valid vectors");
            curr_entry = candidates[0].node_id;
        }

        let top_layer = new_level.min(self.max_layer);
        for level in (0..=top_layer).rev() {
            let candidates = self
                .search_layer(query, &[curr_entry], level, self.ef_construction)
                .expect("search_layer should not fail with valid vectors");

            let m_curr = if level == 0 { self.m_max } else { self.m };
            let neighbors = Self::select_neighbors_simple(&candidates, m_curr);

            for &neighbor in &neighbors {
                self.adjacency[level][node_id].push(neighbor);
            }

            for &neighbor in &neighbors {
                self.adjacency[level][neighbor].push(node_id);
            }

            for &neighbor in &neighbors {
                let limit = if level == 0 { self.m_max } else { self.m };
                if self.adjacency[level][neighbor].len() > limit {
                    let new_adj = {
                        let node_vec = &self.vectors[neighbor];
                        let mut cand: Vec<Candidate> = self.adjacency[level][neighbor]
                            .iter()
                            .map(|&n| {
                                let dist = CosineSpace::distance(node_vec, &self.vectors[n])
                                    .expect("vectors share the same dimension");
                                Candidate {
                                    node_id: n,
                                    distance: dist,
                                }
                            })
                            .collect();
                        cand.sort_unstable_by(|a, b| {
                            a.distance
                                .partial_cmp(&b.distance)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        cand.truncate(limit);
                        cand.into_iter().map(|c| c.node_id).collect::<Vec<_>>()
                    };
                    self.adjacency[level][neighbor] = new_adj;
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

    /// Searches for the top-K nearest neighbors of the query vector using the HNSW algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`VectorError::EmptyVector`] if the query vector is empty.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, VectorError> {
        if self.vectors.is_empty() {
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

    /// Searches a single layer of the graph for the `ef` nearest neighbors to `query`,
    /// starting from the given entry points.
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
            let dist = CosineSpace::distance(query, &self.vectors[ep])?;
            let cand = Candidate {
                node_id: ep,
                distance: dist,
            };
            candidates.push(Reverse(cand.clone()));
            result.push(cand);
            visited.insert(ep);
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let furthest = result
                .peek()
                .ok_or(VectorError::EmptyVector)?;
            if c.distance > furthest.distance {
                break;
            }

            for &neighbor in &self.adjacency[layer][c.node_id] {
                if visited.insert(neighbor) {
                    let dist = CosineSpace::distance(query, &self.vectors[neighbor])?;

                    if result.len() < ef {
                        let cand = Candidate {
                            node_id: neighbor,
                            distance: dist,
                        };
                        candidates.push(Reverse(cand.clone()));
                        result.push(cand);
                    } else {
                        let furthest = result
                            .peek()
                            .ok_or(VectorError::EmptyVector)?;
                        if dist < furthest.distance {
                            let cand = Candidate {
                                node_id: neighbor,
                                distance: dist,
                            };
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

    /// Selects the `m` nearest neighbors from a sorted candidate list.
    fn select_neighbors_simple(candidates: &[Candidate], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|c| c.node_id).collect()
    }

    /// Generates a random level for a new node based on the normalization factor `ml`.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        (-r.ln() * self.ml as f64).floor() as usize
    }

    /// Saves the current index state to a binary file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::Io`] or [`IndexError::Serialization`] if storage fails.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError> {
        let file = std::fs::File::create(path.as_ref())?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Loads the index state from a binary file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, is corrupted, or has an invalid format.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, IndexError> {
        let file = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        let index = bincode::deserialize_from(reader)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;
        Ok(index)
    }

    /// Returns the current total number of indexed vectors.
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
    fn test_hnsw_insert_duplicate_id() {
        let mut index = HnswIndex::with_params(2, 8, 16, 100);
        index.insert(1, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![1.0, 1.0]).unwrap();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_hnsw_accuracy_against_flat() {
        use crate::FlatIndex;

        let dim = 32;
        let n = 200;
        let top_k = 5;

        let mut hnsw = HnswIndex::with_params(dim, 16, 32, 200);
        let mut flat = FlatIndex::new(dim);

        let mut rng = rand::thread_rng();
        let mut queries = Vec::new();

        for i in 0..n {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            hnsw.insert(i as u64, vec.clone()).unwrap();
            flat.insert(i as u64, vec).unwrap();
        }

        for _ in 0..20 {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            queries.push(query);
        }

        let mut total_recall = 0.0;
        for query in &queries {
            let hnsw_hits = hnsw.search(query, top_k).unwrap();
            let flat_hits = flat.search(query, top_k).unwrap();

            let hnsw_ids: Vec<u64> = hnsw_hits.iter().map(|h| h.id).collect();
            let flat_ids: Vec<u64> = flat_hits.iter().map(|h| h.id).collect();

            let overlap = hnsw_ids.iter().filter(|id| flat_ids.contains(id)).count();
            let recall = overlap as f64 / top_k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / queries.len() as f64;
        assert!(avg_recall > 0.90, "HNSW recall too low: {avg_recall:.3}");
    }

    #[test]
    fn test_hnsw_persistence() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("vivid_hnsw_test.bin");

        let mut original = HnswIndex::with_params(3, 8, 16, 100);
        original.insert(42, vec![0.1, 0.9, 0.0]).unwrap();
        original.insert(99, vec![0.8, 0.2, 0.0]).unwrap();

        original.save_to_file(&file_path).unwrap();
        let loaded = HnswIndex::load_from_file(&file_path).unwrap();

        assert_eq!(loaded.len(), original.len());

        let query = [0.15, 0.85, 0.0];
        let original_hits = original.search(&query, 1).unwrap();
        let loaded_hits = loaded.search(&query, 1).unwrap();

        assert_eq!(original_hits[0].id, loaded_hits[0].id);

        let _ = std::fs::remove_file(file_path);
    }
}

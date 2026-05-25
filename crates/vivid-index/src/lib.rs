//! # Vivid Index
//!
//! This module provides vector indexing mechanisms, starting with
//! a high-performance multithreaded Flat Index with persistence support.

pub mod hnsw;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;
use vivid_core::{CosineSpace, VectorSpace};

/// Custom type for vector identifiers.
pub type VectorId = u64;

/// Errors related to index operations.
#[derive(Debug, Error)]
pub enum IndexError {
    /// Triggered when the inserted vector dimension does not match the index dimension.
    #[error("Dimension mismatch: index expects {expected}, got {found}")]
    DimensionMismatch {
        /// Expected dimension set at index initialization.
        expected: usize,
        /// Actual dimension of the provided vector.
        found: usize,
    },
    /// Triggered when the vector payload is empty.
    #[error("Vector data cannot be empty")]
    EmptyVector,
    /// IO errors when reading/writing from/to disk.
    #[error("Disk IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Errors during binary serialization/deserialization.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl PartialEq for IndexError {
    fn eq(&self, interstate: &Self) -> bool {
        match (self, interstate) {
            (Self::EmptyVector, Self::EmptyVector) => true,
            (
                Self::DimensionMismatch { expected: e1, found: f1 },
                Self::DimensionMismatch { expected: e2, found: f2 },
            ) => e1 == e2 && f1 == f2,
            (Self::Io(..), Self::Io(..)) => true, // Упрощенное сравнение для IO
            (Self::Serialization(s1), Self::Serialization(s2)) => s1 == s2,
            _ => false,
        }
    }
}

/// Represents a single search result match.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The unique identifier of the matched vector.
    pub id: VectorId,
    /// The distance score (lower means closer/more similar).
    pub score: f32,
}

/// Simple Flat Index that stores vectors in a contiguous array
/// and performs brute-force K-Nearest Neighbors (KNN) search.
#[derive(Serialize, Deserialize)]
pub struct FlatIndex {
    dimension: usize,
    vectors: Vec<Vec<f32>>,
    ids: Vec<VectorId>,
}

impl FlatIndex {
    /// Creates a new empty `FlatIndex` with the specified vector dimension.
    #[must_use]
    pub const fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
            ids: Vec::new(),
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

        self.vectors.push(vector);
        self.ids.push(id);
        Ok(())
    }

    /// Searches for the top-K closest vectors relative to the query vector.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, vivid_core::VectorError> {
        let mut results: Vec<SearchResult> = self
            .ids
            .par_iter()
            .zip(self.vectors.par_iter())
            .map(|(&id, vec)| {
                let dist = CosineSpace::distance(query, vec)?;
                Ok(SearchResult { id, score: dist })
            })
            .collect::<Result<Vec<_>, vivid_core::VectorError>>()?;

        results.sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    /// Saves the current index state to a binary file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::Io`] or [`IndexError::Serialization`] if storage fails.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
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
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
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
    fn test_flat_index_search() {
        let mut index = FlatIndex::new(3);
        index.insert(101, vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(102, vec![0.0, 1.0, 0.0]).unwrap();
        index.insert(103, vec![1.0, 1.0, 0.0]).unwrap();

        let query = [0.9, 0.1, 0.0];
        let hits = index.search(&query, 2).unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 101);
        assert_eq!(hits[1].id, 103);
    }

    #[test]
    fn test_index_persistence() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("vivid_test_index.bin");

        let mut original_index = FlatIndex::new(2);
        original_index.insert(42, vec![0.1, 0.9]).unwrap();
        original_index.insert(99, vec![0.8, 0.2]).unwrap();

        original_index.save_to_file(&file_path).unwrap();

        let loaded_index = FlatIndex::load_from_file(&file_path).unwrap();

        assert_eq!(loaded_index.len(), original_index.len());
        
        let query = [0.15, 0.85];
        let original_hits = original_index.search(&query, 1).unwrap();
        let loaded_hits = loaded_index.search(&query, 1).unwrap();
        
        assert_eq!(original_hits, loaded_hits);

        let _ = std::fs::remove_file(file_path);
    }
}

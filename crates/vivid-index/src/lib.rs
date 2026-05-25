//! # Vivid Index
//!
//! This module provides vector indexing mechanisms:
//! - `FlatIndex`: file-backed exact search (brute-force)
//! - `HnswIndex`: in-memory graph index for approximate logarithmic search

pub mod hnsw;
pub use hnsw::HnswIndex;

use bytemuck::cast_slice;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;
use vivid_core::{CosineSpace, VectorError, VectorSpace};

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
    /// Triggered when a vector with the same ID already exists.
    #[error("Duplicate ID: {0}")]
    DuplicateId(VectorId),
    /// Triggered when the requested ID is not found in the index.
    #[error("ID not found: {0}")]
    IdNotFound(VectorId),
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
            (Self::DuplicateId(a), Self::DuplicateId(b)) => a == b,
            (Self::IdNotFound(a), Self::IdNotFound(b)) => a == b,
            (Self::Io(..), Self::Io(..)) => true,
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

// ---------------------------------------------------------------------------
// FlatIndex — file-backed exact search
// ---------------------------------------------------------------------------

const FLAT_MAGIC: [u8; 4] = *b"VIDV";
const FLAT_HEADER: usize = 16;

/// File-backed flat index for exact brute-force search.
///
/// The entire file is loaded into a single `Vec<u8>` and parsed zero-copy via
/// `bytemuck`, avoiding per-vector `Vec` overhead.
///
/// # File format
///
/// ```text
/// Offset  | Size          | Field
/// 0       | 4             | magic: b"VIDV"
/// 4       | 4             | dimension (u32 LE)
/// 8       | 8             | num_vectors (u64 LE)
/// 16      | num_vectors*8 | ids: [u64; num_vectors]
/// 16+8*N  | dim*N*4       | vectors: [f32; num_vectors * dimension]
/// ```
pub struct FlatIndex {
    data: Vec<u8>,
    dimension: usize,
    num_vectors: usize,
}

impl FlatIndex {
    /// Opens an existing index file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, IndexError> {
        let mut file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len() as usize;
        let mut data = Vec::with_capacity(file_size);
        file.read_to_end(&mut data)?;

        if data.len() < FLAT_HEADER {
            return Err(IndexError::Serialization("file too small for header".into()));
        }

        if data[0..4] != FLAT_MAGIC {
            return Err(IndexError::Serialization("invalid magic bytes".into()));
        }

        let dim_bytes: [u8; 4] = data[4..8].try_into().unwrap();
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let n_bytes: [u8; 8] = data[8..16].try_into().unwrap();
        let num_vectors = u64::from_le_bytes(n_bytes) as usize;

        let expected = FLAT_HEADER + num_vectors * 8 + num_vectors * dimension * 4;
        if data.len() < expected {
            return Err(IndexError::Serialization("file too small".into()));
        }

        Ok(Self { data, dimension, num_vectors })
    }

    /// Creates a new index file from the given vectors and opens it.
    pub fn create<P: AsRef<Path>>(
        path: P,
        dimension: usize,
        ids: &[VectorId],
        vectors: &[Vec<f32>],
    ) -> Result<Self, IndexError> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&FLAT_MAGIC)?;
        writer.write_all(&(dimension as u32).to_le_bytes())?;
        writer.write_all(&(vectors.len() as u64).to_le_bytes())?;

        for &id in ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        for vec in vectors {
            writer.write_all(cast_slice(vec.as_slice()))?;
        }

        writer.flush()?;
        drop(writer);
        Self::open(path)
    }

    /// Searches for the top-K nearest neighbors using brute-force with rayon.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, VectorError> {
        let ids_start = FLAT_HEADER;

        let mut results: Vec<SearchResult> = (0..self.num_vectors)
            .into_par_iter()
            .map(|i| {
                let ns = ids_start + i * 8;
                let id_bytes: [u8; 8] = self.data[ns..ns + 8].try_into().unwrap();
                let id = u64::from_le_bytes(id_bytes);

                let vs = ids_start + self.num_vectors * 8 + i * self.dimension * 4;
                let ve = vs + self.dimension * 4;
                let vec: &[f32] = cast_slice(&self.data[vs..ve]);

                let dist = CosineSpace::distance(query, vec)?;
                Ok(SearchResult { id, score: dist })
            })
            .collect::<Result<Vec<_>, VectorError>>()?;

        results.sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    /// Returns the vector dimension of the index.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns `true` if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        let ids_start = FLAT_HEADER;
        for i in 0..self.num_vectors {
            let ns = ids_start + i * 8;
            let stored_id = u64::from_le_bytes(self.data[ns..ns + 8].try_into().unwrap());
            if stored_id == id {
                return true;
            }
        }
        false
    }

    /// Retrieves the vector associated with the given ID by linear scan.
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<&[f32]> {
        let ids_start = FLAT_HEADER;
        for i in 0..self.num_vectors {
            let ns = ids_start + i * 8;
            let stored_id = u64::from_le_bytes(self.data[ns..ns + 8].try_into().ok()?);
            if stored_id == id {
                let vs = ids_start + self.num_vectors * 8 + i * self.dimension * 4;
                let ve = vs + self.dimension * 4;
                return Some(cast_slice(&self.data[vs..ve]));
            }
        }
        None
    }

    /// Inserts a new vector.
    ///
    /// The ID is spliced into the ID block (before the vector block) and the
    /// vector data is appended at the end.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::DuplicateId`] if the ID already exists.
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

        let id_insert = FLAT_HEADER + self.num_vectors * 8;
        self.data.splice(id_insert..id_insert, id.to_le_bytes());
        self.data.extend_from_slice(cast_slice(vector.as_slice()));
        self.num_vectors += 1;
        self.update_header_count();
        Ok(())
    }

    /// Replaces the vector for an existing ID.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::IdNotFound`] if the ID is not present.
    pub fn update(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                found: vector.len(),
            });
        }
        let ids_start = FLAT_HEADER;
        for i in 0..self.num_vectors {
            let ns = ids_start + i * 8;
            let stored_id = u64::from_le_bytes(self.data[ns..ns + 8].try_into().unwrap());
            if stored_id == id {
                let vs = ids_start + self.num_vectors * 8 + i * self.dimension * 4;
                let bytes = cast_slice(vector.as_slice());
                self.data[vs..vs + bytes.len()].copy_from_slice(bytes);
                return Ok(());
            }
        }
        Err(IndexError::IdNotFound(id))
    }

    /// Removes a vector by ID.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::IdNotFound`] if the ID is not present.
    pub fn remove(&mut self, id: VectorId) -> Result<(), IndexError> {
        let ids_start = FLAT_HEADER;
        for i in 0..self.num_vectors {
            let ns = ids_start + i * 8;
            let stored_id = u64::from_le_bytes(self.data[ns..ns + 8].try_into().unwrap());
            if stored_id == id {
                // Remove the ID entry (8 bytes at ns).
                self.data.splice(ns..ns + 8, []);
                // After ID removal, the vector section start shifted left by 8.
                let vec_byte_len = self.dimension * 4;
                let vec_section = ids_start + (self.num_vectors - 1) * 8;
                let vec_start = vec_section + i * vec_byte_len;
                self.data.splice(vec_start..vec_start + vec_byte_len, []);
                self.num_vectors -= 1;
                self.update_header_count();
                return Ok(());
            }
        }
        Err(IndexError::IdNotFound(id))
    }

    /// Writes the current index state to a file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), IndexError> {
        std::fs::write(path.as_ref(), &self.data)?;
        Ok(())
    }

    /// Rewrites the num_vectors field in the header.
    fn update_header_count(&mut self) {
        let bytes = (self.num_vectors as u64).to_le_bytes();
        self.data[8..16].copy_from_slice(&bytes);
    }

    /// Returns the total number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Checks if the index contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let p = dir.join(name);
        let _ = std::fs::remove_file(&p);
        p
    }

    // ── existing tests ──

    #[test]
    fn test_flat_index_search() {
        let path = flat_path("vivid_flat_test.bin");

        let ids = vec![101u64, 102, 103];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];

        let index = FlatIndex::create(&path, 3, &ids, &vectors).unwrap();
        assert_eq!(index.len(), 3);

        let query = [0.9, 0.1, 0.0];
        let hits = index.search(&query, 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 101);
        assert_eq!(hits[1].id, 103);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_index_persistence() {
        let path = flat_path("vivid_flat_persist.bin");

        let ids = vec![42u64, 99];
        let vectors = vec![vec![0.1, 0.9], vec![0.8, 0.2]];

        let created = FlatIndex::create(&path, 2, &ids, &vectors).unwrap();
        let loaded = FlatIndex::open(&path).unwrap();

        assert_eq!(created.len(), loaded.len());

        let query = [0.15, 0.85];
        let a = created.search(&query, 1).unwrap();
        let b = loaded.search(&query, 1).unwrap();
        assert_eq!(a, b);

        let _ = std::fs::remove_file(&path);
    }

    // ── new insertion tests ──

    #[test]
    fn test_flat_insert_and_get() {
        let path = flat_path("vivid_flat_insert_get.bin");
        let mut index = FlatIndex::create(&path, 3, &[], &[]).unwrap();
        assert!(index.is_empty());

        index.insert(42, vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
        assert!(index.contains(42));
        assert!(!index.contains(99));

        let vec = index.get(42);
        assert!(vec.is_some());
        assert_eq!(vec.unwrap(), &[1.0, 2.0, 3.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_insert_multiple() {
        let path = flat_path("vivid_flat_insert_multi.bin");
        let mut index = FlatIndex::create(&path, 2, &[], &[]).unwrap();

        for id in 0..10 {
            index.insert(id, vec![id as f32, (id * 2) as f32]).unwrap();
        }
        assert_eq!(index.len(), 10);

        for id in 0..10 {
            let v = index.get(id).unwrap();
            assert_eq!(v[0], id as f32);
            assert_eq!(v[1], (id * 2) as f32);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_insert_duplicate_rejected() {
        let path = flat_path("vivid_flat_insert_dup.bin");
        let mut index = FlatIndex::create(&path, 2, &[], &[]).unwrap();

        index.insert(1, vec![0.1, 0.2]).unwrap();
        let err = index.insert(1, vec![0.3, 0.4]).unwrap_err();
        assert_eq!(err, IndexError::DuplicateId(1));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_insert_empty_vector() {
        let path = flat_path("vivid_flat_insert_empty.bin");
        let mut index = FlatIndex::create(&path, 3, &[], &[]).unwrap();

        let err = index.insert(1, vec![]).unwrap_err();
        assert_eq!(err, IndexError::EmptyVector);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_insert_dimension_mismatch() {
        let path = flat_path("vivid_flat_insert_dim.bin");
        let mut index = FlatIndex::create(&path, 3, &[], &[]).unwrap();

        let err = index.insert(1, vec![0.1, 0.2]).unwrap_err();
        assert_eq!(
            err,
            IndexError::DimensionMismatch { expected: 3, found: 2 }
        );

        let _ = std::fs::remove_file(&path);
    }

    // ── new removal tests ──

    #[test]
    fn test_flat_remove() {
        let path = flat_path("vivid_flat_remove.bin");
        let mut index = FlatIndex::create(&path, 2, &[1, 2, 3], &[vec![0.0; 2], vec![1.0; 2], vec![2.0; 2]]).unwrap();

        assert_eq!(index.len(), 3);
        index.remove(2).unwrap();
        assert_eq!(index.len(), 2);
        assert!(!index.contains(2));
        assert!(index.contains(1));
        assert!(index.contains(3));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_remove_nonexistent() {
        let path = flat_path("vivid_flat_remove_nope.bin");
        let mut index = FlatIndex::create(&path, 1, &[1], &[vec![0.0]]).unwrap();

        let err = index.remove(999).unwrap_err();
        assert_eq!(err, IndexError::IdNotFound(999));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_remove_last_vector() {
        let path = flat_path("vivid_flat_remove_last.bin");
        let mut index = FlatIndex::create(&path, 1, &[1], &[vec![0.0]]).unwrap();

        assert!(!index.is_empty());
        index.remove(1).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let _ = std::fs::remove_file(&path);
    }

    // ── new update tests ──

    #[test]
    fn test_flat_update() {
        let path = flat_path("vivid_flat_update.bin");
        let mut index = FlatIndex::create(&path, 3, &[10], &[vec![0.0; 3]]).unwrap();

        index.update(10, vec![9.0, 8.0, 7.0]).unwrap();
        assert_eq!(index.get(10).unwrap(), &[9.0, 8.0, 7.0]);
        assert_eq!(index.len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_flat_update_nonexistent() {
        let path = flat_path("vivid_flat_update_nope.bin");
        let mut index = FlatIndex::create(&path, 1, &[], &[]).unwrap();

        let err = index.update(42, vec![1.0]).unwrap_err();
        assert_eq!(err, IndexError::IdNotFound(42));

        let _ = std::fs::remove_file(&path);
    }

    // ── new get tests ──

    #[test]
    fn test_flat_get_nonexistent() {
        let path = flat_path("vivid_flat_get_nope.bin");
        let index = FlatIndex::create(&path, 2, &[1], &[vec![1.0; 2]]).unwrap();

        assert!(index.get(999).is_none());
        assert!(index.get(1).is_some());

        let _ = std::fs::remove_file(&path);
    }

    // ── new save-to-file round-trip ──

    #[test]
    fn test_flat_save_to_file_roundtrip() {
        let path = flat_path("vivid_flat_roundtrip.bin");
        let mut index = FlatIndex::create(&path, 3, &[], &[]).unwrap();

        index.insert(1, vec![1.0, 2.0, 3.0]).unwrap();
        index.insert(2, vec![4.0, 5.0, 6.0]).unwrap();
        index.save_to_file(&path).unwrap();

        let loaded = FlatIndex::open(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(1).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(loaded.get(2).unwrap(), &[4.0, 5.0, 6.0]);

        // Mutate and re-save
        let mut loaded = loaded;
        loaded.remove(1).unwrap();
        loaded.insert(3, vec![7.0, 8.0, 9.0]).unwrap();
        loaded.save_to_file(&path).unwrap();

        let reloaded = FlatIndex::open(&path).unwrap();
        assert_eq!(reloaded.len(), 2);
        assert!(reloaded.get(1).is_none());
        assert_eq!(reloaded.get(3).unwrap(), &[7.0, 8.0, 9.0]);

        let _ = std::fs::remove_file(&path);
    }
}

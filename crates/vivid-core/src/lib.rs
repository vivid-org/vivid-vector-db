//! # Vivid Core
//!
//! This module provides abstractions for vector spaces
//! and high-performance distance metrics.

#![cfg_attr(feature = "nightly", feature(portable_simd))]
#![cfg_attr(feature = "nightly", allow(missing_docs))]

use thiserror::Error;

#[cfg(feature = "nightly")]
mod simd;

/// List of errors that can occur during vector operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VectorError {
    /// Triggered when two vectors have different dimensions during metric computation.
    #[error("Vector dimensions mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// Expected dimension (first vector).
        expected: usize,
        /// Actual dimension (second vector).
        found: usize,
    },
    /// Triggered when an empty vector slice is provided.
    #[error("Vector is empty")]
    EmptyVector,
}

/// Abstraction over distance metrics in a vector space.
pub trait VectorSpace {
    /// Computes the distance between two vectors of the same dimension.
    ///
    /// # Errors
    ///
    /// Returns [`VectorError::DimensionMismatch`] if the slice lengths differ.
    /// Returns [`VectorError::EmptyVector`] if the input slices are empty.
    fn distance(v1: &[f32], v2: &[f32]) -> Result<f32, VectorError>;
}

/// Euclidean distance (L2 Space).
pub struct L2Space;

impl VectorSpace for L2Space {
    fn distance(v1: &[f32], v2: &[f32]) -> Result<f32, VectorError> {
        validate_vectors(v1, v2)?;

        #[cfg(feature = "nightly")]
        {
            Ok(simd::l2_distance_simd(v1, v2))
        }

        #[cfg(not(feature = "nightly"))]
        {
            let sum_squared: f32 = v1
                .iter()
                .zip(v2.iter())
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();

            Ok(sum_squared.sqrt())
        }
    }
}

/// Cosine distance (Cosine Space).
/// Returns a value from 0.0 (identical) to 2.0 (opposite).
pub struct CosineSpace;

impl VectorSpace for CosineSpace {
    fn distance(v1: &[f32], v2: &[f32]) -> Result<f32, VectorError> {
        validate_vectors(v1, v2)?;

        #[cfg(feature = "nightly")]
        {
            Ok(simd::cosine_distance_simd(v1, v2))
        }

        #[cfg(not(feature = "nightly"))]
        {
            let mut dot_product = 0.0;
            let mut norm_v1 = 0.0;
            let mut norm_v2 = 0.0;

            for (&x, &y) in v1.iter().zip(v2.iter()) {
                dot_product += x * y;
                norm_v1 += x * x;
                norm_v2 += y * y;
            }

            if norm_v1 == 0.0 || norm_v2 == 0.0 {
                return Ok(1.0);
            }

            let similarity = dot_product / (norm_v1.sqrt() * norm_v2.sqrt());
            Ok(1.0 - similarity)
        }
    }
}

/// Validates input slices for all metrics.
fn validate_vectors(v1: &[f32], v2: &[f32]) -> Result<(), VectorError> {
    if v1.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    if v1.len() != v2.len() {
        return Err(VectorError::DimensionMismatch {
            expected: v1.len(),
            found: v2.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [4.0, 6.0, 3.0];
        assert_eq!(L2Space::distance(&v1, &v2).unwrap(), 5.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = [1.0, 0.0];
        let v2 = [0.0, 1.0]; 
        assert_eq!(CosineSpace::distance(&v1, &v2).unwrap(), 1.0);

        let v3 = [2.0, 4.0, 5.0];
        let v4 = [2.0, 4.0, 5.0];
        assert_eq!(CosineSpace::distance(&v3, &v4).unwrap(), 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = [1.0, 2.0];
        let v2 = [1.0, 2.0, 3.0];
        let result = L2Space::distance(&v1, &v2);
        assert_eq!(
            result.unwrap_err(),
            VectorError::DimensionMismatch { expected: 2, found: 3 }
        );
    }
}

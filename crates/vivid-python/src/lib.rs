//! # Vivid Python Bindings
//!
//! Exposes the high-performance Rust HNSW index to the Python ecosystem using PyO3.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vivid_index::HnswIndex;

/// High-performance Approximate Nearest Neighbor Search (HNSW) index backed by Rust.
#[pyclass]
struct PyVividIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyVividIndex {
    /// Creates a new PyVividIndex instance from Python.
    ///
    /// Example: `index = vivid.PyVividIndex(3)`
    #[new]
    fn new(dimension: usize) -> Self {
        Self {
            inner: HnswIndex::new(dimension),
        }
    }

    /// Inserts a vector into the index.
    /// Expects an integer ID and a list of floats.
    fn insert(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner
            .insert(id, vector)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Searches for the top_k nearest neighbors of the query vector.
    /// Returns a list of dictionaries with 'id' and 'score'.
    fn search(&self, py: Python<'_>, query: Vec<f32>, top_k: usize) -> PyResult<Vec<PyObject>> {
        let hits = self.inner
            .search(&query, top_k)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut py_results = Vec::with_capacity(hits.len());
        for hit in hits {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("id", hit.id)?;
            dict.set_item("score", hit.score)?;
            py_results.push(dict.to_object(py));
        }

        Ok(py_results)
    }

    /// Saves the index to a binary file.
    fn save_to_file(&self, path: String) -> PyResult<()> {
        self.inner
            .save_to_file(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Loads the index from a binary file.
    #[staticmethod]
    fn load_from_file(path: String) -> PyResult<Self> {
        let index = HnswIndex::load_from_file(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: index })
    }

    /// Returns the total number of vectors in the index.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn vivid(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVividIndex>()?;
    Ok(())
}

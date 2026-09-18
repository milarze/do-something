//! Generic storage error type.
//!
//! Framework-level errors that are domain-agnostic.

use std::io;
use thiserror::Error;

/// Storage-related errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StorageError {
    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Config directory not found or inaccessible.
    #[error("Config directory not found")]
    NotFound,

    /// Invalid path specified.
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Record not found.
    #[error("Record not found: {0}")]
    NotFoundById(String),

    /// Lock acquisition failed.
    #[error("Lock error: {0}")]
    LockError(String),

    /// Index corruption.
    #[error("Index corruption: {0}")]
    IndexCorruption(String),
}

/// Framework-level storage result.
pub type Result<T> = std::result::Result<T, StorageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = StorageError::NotFound;
        assert_eq!(err.to_string(), "Config directory not found");

        let err = StorageError::InvalidPath("test".to_string());
        assert_eq!(err.to_string(), "Invalid path: test");
    }
}

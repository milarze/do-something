//! Storage error types.

use std::io;
use thiserror::Error;

/// Storage-related errors.
#[derive(Debug, Error)]
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

    /// Recipe not found.
    #[error("Recipe not found: {0}")]
    RecipeNotFound(String),

    /// Session not found.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Lock acquisition failed.
    #[error("Lock error: {0}")]
    LockError(String),

    /// Index corruption.
    #[error("Index corruption: {0}")]
    IndexCorruption(String),

    /// SQLite database error.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, StorageError>;

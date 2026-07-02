//! File system utilities for storage layer.
//!
//! Provides common file operations and safety checks.

use super::error::{Result, StorageError};

/// Validates that a string is safe to use as a file path component.
///
/// Prevents path traversal attacks by rejecting strings containing:
/// - Path separators (`/`, `\`)
/// - Parent directory references (`..`)
/// - Null bytes
pub fn validate_path_component(s: &str, context: &str) -> Result<()> {
    if s.is_empty() {
        return Err(StorageError::InvalidPath(format!(
            "{} cannot be empty",
            context
        )));
    }
    if s.contains('/')
        || s.contains('\\')
        || s.contains("..")
        || s.contains('\0')
    {
        return Err(StorageError::InvalidPath(format!(
            "Invalid characters in {}: {}",
            context, s
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_component() {
        assert!(validate_path_component("example.com", "domain").is_ok());
        assert!(validate_path_component("user123", "user_id").is_ok());
        assert!(validate_path_component("session-abc", "session").is_ok());
    }

    #[test]
    fn rejects_empty_string() {
        let result = validate_path_component("", "test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_slash() {
        let result = validate_path_component("foo/bar", "test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_backslash() {
        let result = validate_path_component("foo\\bar", "test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_parent_directory() {
        let result = validate_path_component("../etc/passwd", "test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_null_byte() {
        let result = validate_path_component("test\0value", "test");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }
}

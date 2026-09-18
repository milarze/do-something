//! Framework storage primitives.
//!
//! Generic, reusable storage components that contain no domain symbols.
//! These provide file I/O mechanics only - domain stores are thin wrappers
//! over these primitives.

pub mod append_only_db;
pub mod checkpoint;
pub mod config_dir;
pub mod daily_log;
pub mod error;
pub mod json_store;
pub mod session_store;

pub use append_only_db::AppendOnlyDb;
pub use checkpoint::CheckpointStore;
pub use config_dir::ConfigDir;
pub use daily_log::DailyLog;
pub use error::{Result, StorageError};
pub use json_store::JsonStore;
pub use session_store::SessionStore;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn full_storage_workflow() {
        let dir = tempdir().unwrap();
        let config = ConfigDir::from_path(dir.path().to_path_buf());
        config.init().unwrap();

        // Verify all directories exist
        assert!(config.recipes_dir().exists());
        assert!(config.signals_dir().exists());
        assert!(config.knowledge_dir().exists());
        assert!(config.state_dir().exists());
    }
}

//! Storage layer for the recipe scraping agent.
//!
//! This module provides persistence for:
//! - Recipe storage with search index
//! - Signal logging (daily JSONL files)
//! - Knowledge store (site configs, user models, patterns)
//! - Session state for resumability
//!
//! # Architecture
//!
//! Storage is abstracted through traits to allow multiple backends:
//! - [`RecipesStorage`] - Recipe CRUD operations
//! - [`SignalStorage`] - Signal logging
//! - [`KnowledgeStorage`] - Knowledge persistence
//! - [`SessionStorage`] - Session state
//! - [`Storage`] - Combined interface
//!
//! # Backends
//!
//! For recipes, two backends are provided:
//! - [`FileRecipesDb`] - Uses JSONL files (default)
//! - [`SqliteRecipesDb`] - Uses SQLite database
//!
//! For other storage, file-based implementations are used:
//! - [`FileSignalLog`] - Daily JSONL files
//! - [`FileKnowledgeStore`] - JSON files
//! - [`FileSessionStore`] - JSON files
//!
//! # Directory Structure
//!
//! ```text
//! ~/.do-something/
//! ├── config.json                    # Runtime config
//! ├── knowledge/
//! │   ├── site_configs/
//! │   ├── user_models/
//! │   └── patterns/
//! ├── recipes/
//! │   ├── recipes.jsonl
//! │   └── index.json
//! ├── signals/
//! │   └── YYYY-MM-DD.jsonl
//! └── state/
//!     └── sessions/
//! ```

pub mod config_dir;
pub mod error;
pub mod file_storage;
pub mod knowledge_store;
pub mod recipes_db;
pub mod session_store;
pub mod signal_log;
pub mod sqlite_recipes;
pub mod traits;

// Re-export file-based implementations
pub use config_dir::ConfigDir;
pub use error::{Result, StorageError};
pub use file_storage::FileStorage;
pub use knowledge_store::FileKnowledgeStore;
pub use recipes_db::FileRecipesDb;
pub use session_store::FileSessionStore;
pub use signal_log::FileSignalLog;

// Re-export SQLite recipes implementation
pub use sqlite_recipes::SqliteRecipesDb;

// Re-export traits
pub use traits::{KnowledgeStorage, RecipesStorage, SessionStorage, SignalStorage, Storage};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn full_storage_workflow() {
        let dir = tempdir().unwrap();
        let config = ConfigDir::from_path(dir.path().to_path_buf());
        config.init().unwrap();

        // Open all storage components
        let recipes = FileRecipesDb::open(config.recipes_dir()).unwrap();
        let _signals = FileSignalLog::open(config.signals_dir()).unwrap();
        let _knowledge = FileKnowledgeStore::open(config.knowledge_dir()).unwrap();
        let sessions = FileSessionStore::open(config.sessions_dir()).unwrap();

        // Verify all directories exist
        assert!(config.recipes_dir().exists());
        assert!(config.signals_dir().exists());
        assert!(config.knowledge_dir().exists());
        assert!(config.sessions_dir().exists());

        // Verify we can use each component
        use traits::{RecipesStorage, SessionStorage};
        assert_eq!(recipes.count().unwrap(), 0);
        assert_eq!(sessions.count().unwrap(), 0);
    }

    #[test]
    fn storage_trait_implementation() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::open_at(dir.path().to_path_buf()).unwrap();

        // Can use through trait methods
        use traits::{RecipesStorage, SessionStorage};
        assert_eq!(storage.recipes().count().unwrap(), 0);
        assert_eq!(storage.sessions().count().unwrap(), 0);
    }

    #[test]
    fn sqlite_recipes_works() {
        let sqlite_db = SqliteRecipesDb::open_in_memory().unwrap();
        
        use traits::RecipesStorage;
        assert_eq!(sqlite_db.count().unwrap(), 0);
    }
}

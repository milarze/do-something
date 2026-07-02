//! Combined file-based storage backend.
//!
//! Provides a unified storage interface with file-based implementations
//! for all storage components.

use std::path::PathBuf;
use std::sync::Arc;

use super::config_dir::ConfigDir;
use super::error::Result;
use super::knowledge_store::FileKnowledgeStore;
use super::recipes_db::FileRecipesDb;
use super::session_store::FileSessionStore;
use super::signal_log::FileSignalLog;
use super::traits::Storage;

/// File-based storage backend.
#[derive(Debug)]
pub struct FileStorage {
    config: ConfigDir,
    recipes: Arc<FileRecipesDb>,
    signals: Arc<FileSignalLog>,
    knowledge: Arc<FileKnowledgeStore>,
    sessions: Arc<FileSessionStore>,
}

impl FileStorage {
    /// Open storage using the default config directory.
    pub fn open() -> Result<Self> {
        let config = ConfigDir::resolve()?;
        Self::from_config(config)
    }

    /// Open storage at a specific path.
    pub fn open_at(path: PathBuf) -> Result<Self> {
        let config = ConfigDir::from_path(path);
        Self::from_config(config)
    }

    /// Create storage from a ConfigDir.
    pub fn from_config(config: ConfigDir) -> Result<Self> {
        config.init()?;
        
        let recipes = Arc::new(FileRecipesDb::open(config.recipes_dir())?);
        let signals = Arc::new(FileSignalLog::open(config.signals_dir())?);
        let knowledge = Arc::new(FileKnowledgeStore::open(config.knowledge_dir())?);
        let sessions = Arc::new(FileSessionStore::open(config.sessions_dir())?);

        Ok(Self {
            config,
            recipes,
            signals,
            knowledge,
            sessions,
        })
    }

    /// Get the config directory.
    pub fn config_dir(&self) -> &ConfigDir {
        &self.config
    }
}

impl Storage for FileStorage {
    type Recipes = FileRecipesDb;
    type Signals = FileSignalLog;
    type Knowledge = FileKnowledgeStore;
    type Sessions = FileSessionStore;

    fn recipes(&self) -> &Self::Recipes {
        &self.recipes
    }

    fn signals(&self) -> &Self::Signals {
        &self.signals
    }

    fn knowledge(&self) -> &Self::Knowledge {
        &self.knowledge
    }

    fn sessions(&self) -> &Self::Sessions {
        &self.sessions
    }
}

/// Extension trait for file-based storage with direct access to implementations.
impl FileStorage {
    /// Get direct access to the file-based recipes storage.
    pub fn file_recipes(&self) -> &FileRecipesDb {
        &self.recipes
    }

    /// Get direct access to the file-based signal storage.
    pub fn file_signals(&self) -> &FileSignalLog {
        &self.signals
    }

    /// Get direct access to the file-based knowledge storage.
    pub fn file_knowledge(&self) -> &FileKnowledgeStore {
        &self.knowledge
    }

    /// Get direct access to the file-based session storage.
    pub fn file_sessions(&self) -> &FileSessionStore {
        &self.sessions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::storage::traits::{RecipesStorage, SessionStorage};

    #[test]
    fn open_creates_directory_structure() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::open_at(dir.path().to_path_buf()).unwrap();

        assert!(storage.config.recipes_dir().exists());
        assert!(storage.config.signals_dir().exists());
        assert!(storage.config.knowledge_dir().exists());
        assert!(storage.config.sessions_dir().exists());
    }

    #[test]
    fn storage_trait_methods_work() {
        let dir = tempdir().unwrap();
        let storage = FileStorage::open_at(dir.path().to_path_buf()).unwrap();

        // Can use through trait methods
        assert_eq!(storage.recipes().count().unwrap(), 0);
        assert_eq!(storage.sessions().count().unwrap(), 0);
    }
}

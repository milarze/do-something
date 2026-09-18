//! Recipe-domain directory layout.
//!
//! The framework's [`ConfigDir`] is intentionally domain-agnostic: it only
//! knows about generic agent infrastructure (state, sessions, checkpoint).
//! Recipe-specific directories live here, in the domain layer, and are built
//! on top of the framework root path.

use std::fs;
use std::path::PathBuf;

use crate::framework::storage::{ConfigDir, Result};

/// Recipe-domain directory layout on top of a framework [`ConfigDir`].
///
/// Owns a `ConfigDir` so that [`RecipePaths::init`] can create both the
/// generic framework directories and the recipe-specific ones.
#[derive(Debug, Clone)]
pub struct RecipePaths {
    config: ConfigDir,
}

impl RecipePaths {
    /// Wrap a framework [`ConfigDir`].
    pub fn new(config: ConfigDir) -> Self {
        Self { config }
    }

    /// Borrow the underlying framework config.
    pub fn config(&self) -> &ConfigDir {
        &self.config
    }

    /// Recipes directory (append-only recipe records).
    pub fn recipes_dir(&self) -> PathBuf {
        self.config.path().join("recipes")
    }

    /// Signals directory (daily signal logs).
    pub fn signals_dir(&self) -> PathBuf {
        self.config.path().join("signals")
    }

    /// Knowledge root directory.
    pub fn knowledge_dir(&self) -> PathBuf {
        self.config.path().join("knowledge")
    }

    /// Site configs directory (keyed JSON store).
    pub fn site_configs_dir(&self) -> PathBuf {
        self.knowledge_dir().join("site_configs")
    }

    /// User models directory (keyed JSON store).
    pub fn user_models_dir(&self) -> PathBuf {
        self.knowledge_dir().join("user_models")
    }

    /// Patterns directory (single-object JSON store).
    pub fn patterns_dir(&self) -> PathBuf {
        self.knowledge_dir().join("patterns")
    }

    /// Create the framework directories plus all recipe-domain directories.
    pub fn init(&self) -> Result<()> {
        // Framework-level generic dirs.
        self.config.init()?;
        // Recipe-domain dirs.
        fs::create_dir_all(self.recipes_dir())?;
        fs::create_dir_all(self.signals_dir())?;
        fs::create_dir_all(self.knowledge_dir())?;
        fs::create_dir_all(self.site_configs_dir())?;
        fs::create_dir_all(self.user_models_dir())?;
        fs::create_dir_all(self.patterns_dir())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn init_creates_all_directories() {
        let dir = tempdir().unwrap();
        let paths = RecipePaths::new(ConfigDir::from_path(dir.path().to_path_buf()));

        paths.init().unwrap();

        // Framework-generic.
        assert!(paths.config().state_dir().exists());
        assert!(paths.config().sessions_dir().exists());
        // Recipe-domain.
        assert!(paths.recipes_dir().exists());
        assert!(paths.signals_dir().exists());
        assert!(paths.knowledge_dir().exists());
        assert!(paths.site_configs_dir().exists());
        assert!(paths.user_models_dir().exists());
        assert!(paths.patterns_dir().exists());
    }

    #[test]
    fn paths_are_nested_correctly() {
        let dir = tempdir().unwrap();
        let paths = RecipePaths::new(ConfigDir::from_path(dir.path().to_path_buf()));

        assert_eq!(paths.recipes_dir(), dir.path().join("recipes"));
        assert_eq!(paths.signals_dir(), dir.path().join("signals"));
        assert_eq!(paths.knowledge_dir(), dir.path().join("knowledge"));
        assert_eq!(
            paths.site_configs_dir(),
            dir.path().join("knowledge/site_configs")
        );
        assert_eq!(
            paths.user_models_dir(),
            dir.path().join("knowledge/user_models")
        );
        assert_eq!(paths.patterns_dir(), dir.path().join("knowledge/patterns"));
    }

    #[test]
    fn init_is_idempotent() {
        let dir = tempdir().unwrap();
        let paths = RecipePaths::new(ConfigDir::from_path(dir.path().to_path_buf()));

        paths.init().unwrap();
        // Second init must not error on existing directories.
        paths.init().unwrap();
    }
}

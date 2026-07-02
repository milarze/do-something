//! Config directory management.
//!
//! Manages the `~/.do-something` directory structure and provides
//! paths to all subdirectories.

use std::env;
use std::fs;
use std::path::PathBuf;

use super::error::{Result, StorageError};

/// Manages the config directory path resolution and initialization.
#[derive(Debug, Clone)]
pub struct ConfigDir {
    path: PathBuf,
}

impl ConfigDir {
    /// Resolve config directory: `$DO_SOMETHING_CONFIG` or `~/.do-something`.
    pub fn resolve() -> Result<Self> {
        let path = if let Ok(custom) = env::var("DO_SOMETHING_CONFIG") {
            PathBuf::from(custom)
        } else {
            dirs::home_dir()
                .ok_or(StorageError::NotFound)?
                .join(".do-something")
        };

        Ok(Self { path })
    }

    /// Create a ConfigDir from a specific path (for testing).
    pub fn from_path(path: PathBuf) -> Self {
        Self { path }
    }

    /// Ensure directory structure exists.
    pub fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.path)?;
        fs::create_dir_all(self.knowledge_dir())?;
        fs::create_dir_all(self.knowledge_dir().join("site_configs"))?;
        fs::create_dir_all(self.knowledge_dir().join("user_models"))?;
        fs::create_dir_all(self.knowledge_dir().join("patterns"))?;
        fs::create_dir_all(self.recipes_dir())?;
        fs::create_dir_all(self.signals_dir())?;
        fs::create_dir_all(self.state_dir())?;
        fs::create_dir_all(self.state_dir().join("sessions"))?;
        Ok(())
    }

    /// Get the root config directory path.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Get knowledge directory path.
    pub fn knowledge_dir(&self) -> PathBuf {
        self.path.join("knowledge")
    }

    /// Get recipes directory path.
    pub fn recipes_dir(&self) -> PathBuf {
        self.path.join("recipes")
    }

    /// Get signals directory path.
    pub fn signals_dir(&self) -> PathBuf {
        self.path.join("signals")
    }

    /// Get state directory path.
    pub fn state_dir(&self) -> PathBuf {
        self.path.join("state")
    }

    /// Get config file path.
    pub fn config_file(&self) -> PathBuf {
        self.path.join("config.json")
    }

    /// Get site configs directory path.
    pub fn site_configs_dir(&self) -> PathBuf {
        self.knowledge_dir().join("site_configs")
    }

    /// Get user models directory path.
    pub fn user_models_dir(&self) -> PathBuf {
        self.knowledge_dir().join("user_models")
    }

    /// Get patterns directory path.
    pub fn patterns_dir(&self) -> PathBuf {
        self.knowledge_dir().join("patterns")
    }

    /// Get sessions directory path.
    pub fn sessions_dir(&self) -> PathBuf {
        self.state_dir().join("sessions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn resolve_uses_env_variable() {
        let dir = tempdir().unwrap();
        // SAFETY: Test sets environment variable in single-threaded context
        unsafe {
            env::set_var("DO_SOMETHING_CONFIG", dir.path());
        }

        let config = ConfigDir::resolve().unwrap();
        assert_eq!(config.path(), dir.path());

        // SAFETY: Test removes environment variable in single-threaded context
        unsafe {
            env::remove_var("DO_SOMETHING_CONFIG");
        }
    }

    #[test]
    fn init_creates_directory_structure() {
        let dir = tempdir().unwrap();
        let config = ConfigDir::from_path(dir.path().to_path_buf());

        config.init().unwrap();

        assert!(dir.path().exists());
        assert!(config.knowledge_dir().exists());
        assert!(config.recipes_dir().exists());
        assert!(config.signals_dir().exists());
        assert!(config.state_dir().exists());
        assert!(config.site_configs_dir().exists());
        assert!(config.user_models_dir().exists());
        assert!(config.patterns_dir().exists());
        assert!(config.sessions_dir().exists());
    }

    #[test]
    fn subdirectory_paths_are_correct() {
        let dir = tempdir().unwrap();
        let config = ConfigDir::from_path(dir.path().to_path_buf());

        assert_eq!(config.knowledge_dir(), dir.path().join("knowledge"));
        assert_eq!(config.recipes_dir(), dir.path().join("recipes"));
        assert_eq!(config.signals_dir(), dir.path().join("signals"));
        assert_eq!(config.state_dir(), dir.path().join("state"));
        assert_eq!(config.config_file(), dir.path().join("config.json"));
    }
}

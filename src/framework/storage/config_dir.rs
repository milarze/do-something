//! Config directory management.
//!
//! Manages the root config directory and provides paths to generic,
//! domain-agnostic subdirectories (state, sessions, checkpoint). Domain
//! layers are responsible for their own directory layout on top of the
//! root path exposed here.

use std::env;
use std::fs;
use std::path::PathBuf;

use super::error::{Result, StorageError};

/// Manages the config directory path resolution and initialization.
///
/// This is framework-level: it knows only about generic agent infrastructure
/// directories. Domain-specific layouts live in the domain layer.
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

    /// Ensure the generic directory structure exists. Domain layers create
    /// their own directories via their path helpers.
    pub fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.path)?;
        fs::create_dir_all(self.state_dir())?;
        fs::create_dir_all(self.sessions_dir())?;
        Ok(())
    }

    /// Get the root config directory path.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Get state directory path.
    pub fn state_dir(&self) -> PathBuf {
        self.path.join("state")
    }

    /// Get config file path.
    pub fn config_file(&self) -> PathBuf {
        self.path.join("config.json")
    }

    /// Get sessions directory path.
    pub fn sessions_dir(&self) -> PathBuf {
        self.state_dir().join("sessions")
    }

    /// Get checkpoint file path.
    pub fn checkpoint_file(&self) -> PathBuf {
        self.state_dir().join("compression_checkpoint.json")
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
    fn init_creates_generic_directory_structure() {
        let dir = tempdir().unwrap();
        let config = ConfigDir::from_path(dir.path().to_path_buf());

        config.init().unwrap();

        assert!(dir.path().exists());
        assert!(config.state_dir().exists());
        assert!(config.sessions_dir().exists());
    }

    #[test]
    fn generic_paths_are_correct() {
        let dir = tempdir().unwrap();
        let config = ConfigDir::from_path(dir.path().to_path_buf());

        assert_eq!(config.state_dir(), dir.path().join("state"));
        assert_eq!(config.config_file(), dir.path().join("config.json"));
        assert_eq!(config.sessions_dir(), dir.path().join("state/sessions"));
        assert_eq!(
            config.checkpoint_file(),
            dir.path().join("state/compression_checkpoint.json")
        );
    }
}

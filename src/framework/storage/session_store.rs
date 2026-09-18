//! Generic session state persistence.
//!
//! Parameterized by session type. Used to save/resume agent sessions.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::marker::PhantomData;
use std::path::PathBuf;

use serde::{Serialize, de::DeserializeOwned};

use super::error::{Result, StorageError};

/// Validates that a string is safe to use as a file path component.
fn validate_path_component(s: &str, context: &str) -> Result<()> {
    if s.is_empty() {
        return Err(StorageError::InvalidPath(format!(
            "{} cannot be empty",
            context
        )));
    }
    if s.contains('/') || s.contains('\\') || s.contains("..") || s.contains('\0') {
        return Err(StorageError::InvalidPath(format!(
            "Invalid characters in {}: {}",
            context, s
        )));
    }
    Ok(())
}

/// Generic session state persistence.
///
/// Sessions are stored as JSON files keyed by session ID.
pub struct SessionStore<S: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<S>,
}

impl<S: Serialize + DeserializeOwned> SessionStore<S> {
    /// Open the session store at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            _marker: PhantomData,
        })
    }

    /// Get the path for a session file.
    fn session_path(&self, id: &str) -> Result<PathBuf> {
        validate_path_component(id, "session ID")?;
        Ok(self.dir.join(format!("{}.json", id)))
    }

    /// Save a session.
    pub fn save(&self, session: &S) -> Result<()>
    where
        S: HasSessionId,
    {
        let id = session.session_id();
        let path = self.session_path(&id)?;
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, session)?;
        Ok(())
    }

    /// Load a session by ID.
    pub fn load(&self, id: &str) -> Result<Option<S>> {
        let path = self.session_path(id)?;
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let session = serde_json::from_reader(reader)?;
        Ok(Some(session))
    }

    /// List all session IDs, sorted by modification time (newest first).
    pub fn list(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(stem) = path.file_stem()
            {
                let name = stem.to_string_lossy();
                ids.push(name.to_string());
            }
        }

        // Sort by modification time, newest first
        ids.sort_by(|a, b| {
            let path_a = self.dir.join(format!("{}.json", a));
            let path_b = self.dir.join(format!("{}.json", b));
            let time_a = path_a.metadata().and_then(|m| m.modified()).ok();
            let time_b = path_b.metadata().and_then(|m| m.modified()).ok();
            time_b.cmp(&time_a)
        });

        Ok(ids)
    }

    /// Delete a session.
    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.session_path(id)?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Check if a session exists.
    pub fn exists(&self, id: &str) -> Result<bool> {
        let path = self.session_path(id)?;
        Ok(path.exists())
    }

    /// Get the most recent session.
    pub fn latest(&self) -> Result<Option<S>> {
        let ids = self.list()?;
        if let Some(latest_id) = ids.first() {
            return self.load(latest_id);
        }
        Ok(None)
    }

    /// Count total sessions.
    pub fn count(&self) -> Result<usize> {
        Ok(self.list()?.len())
    }

    /// Delete all sessions.
    pub fn clear(&self) -> Result<u64> {
        let ids = self.list()?;
        let count = ids.len() as u64;
        for id in ids {
            self.delete(&id)?;
        }
        Ok(count)
    }
}

/// Trait for types that have a session ID.
///
/// Types stored in SessionStore must implement this trait.
pub trait HasSessionId {
    fn session_id(&self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestSession {
        id: String,
        data: String,
    }

    impl HasSessionId for TestSession {
        fn session_id(&self) -> String {
            self.id.clone()
        }
    }

    #[test]
    fn save_and_load() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        let session = TestSession {
            id: "test-session".to_string(),
            data: "test data".to_string(),
        };

        store.save(&session).unwrap();
        let loaded = store.load("test-session").unwrap().unwrap();

        assert_eq!(loaded, session);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        let loaded = store.load("nonexistent").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn list_sessions() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&TestSession {
                id: "session-a".to_string(),
                data: "a".to_string(),
            })
            .unwrap();
        store
            .save(&TestSession {
                id: "session-b".to_string(),
                data: "b".to_string(),
            })
            .unwrap();

        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&"session-a".to_string()));
        assert!(list.contains(&"session-b".to_string()));
    }

    #[test]
    fn delete_session() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&TestSession {
                id: "to-delete".to_string(),
                data: "test".to_string(),
            })
            .unwrap();
        assert!(store.exists("to-delete").unwrap());

        store.delete("to-delete").unwrap();
        assert!(!store.exists("to-delete").unwrap());
    }

    #[test]
    fn latest_returns_most_recent() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&TestSession {
                id: "old".to_string(),
                data: "old".to_string(),
            })
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        store
            .save(&TestSession {
                id: "new".to_string(),
                data: "new".to_string(),
            })
            .unwrap();

        let latest = store.latest().unwrap().unwrap();
        assert_eq!(latest.id, "new");
    }

    #[test]
    fn clear_removes_all_sessions() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&TestSession {
                id: "s1".to_string(),
                data: "a".to_string(),
            })
            .unwrap();
        store
            .save(&TestSession {
                id: "s2".to_string(),
                data: "b".to_string(),
            })
            .unwrap();
        store
            .save(&TestSession {
                id: "s3".to_string(),
                data: "c".to_string(),
            })
            .unwrap();

        let count = store.clear().unwrap();
        assert_eq!(count, 3);
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn rejects_path_traversal_in_session_id() {
        let dir = tempdir().unwrap();
        let store: SessionStore<TestSession> =
            SessionStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.load("../etc/passwd");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }
}

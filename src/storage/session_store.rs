//! File-based session state persistence for resumability.
//!
//! Stores session state to disk so that long-running scraping
//! sessions can be resumed after interruption.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use crate::models::agent_state::{SessionId, SessionState};

use super::error::Result;
use super::file_utils::validate_path_component;
use super::traits::SessionStorage;

/// File-based persistable session state for resumability.
#[derive(Debug)]
pub struct FileSessionStore {
    dir: PathBuf,
}

impl FileSessionStore {
    /// Open the session store at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Get the path for a session file.
    ///
    /// Returns an error if the session ID contains path traversal characters.
    fn session_path(&self, id: &SessionId) -> Result<PathBuf> {
        validate_path_component(&id.0, "session ID")?;
        Ok(self.dir.join(format!("{}.json", id)))
    }
}

impl SessionStorage for FileSessionStore {
    fn save(&self, session: &SessionState) -> Result<()> {
        let path = self.session_path(&session.id)?;
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, session)?;
        writer.flush()?;
        Ok(())
    }

    fn load(&self, id: &SessionId) -> Result<Option<SessionState>> {
        let path = self.session_path(id)?;
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let session = serde_json::from_reader(reader)?;
        Ok(Some(session))
    }

    fn list(&self) -> Result<Vec<SessionId>> {
        let mut ids = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(stem) = path.file_stem()
            {
                let name = stem.to_string_lossy();
                ids.push(SessionId::new(name.to_string()));
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

    fn delete(&self, id: &SessionId) -> Result<()> {
        let path = self.session_path(id)?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    fn exists(&self, id: &SessionId) -> Result<bool> {
        let path = self.session_path(id)?;
        Ok(path.exists())
    }

    fn latest(&self) -> Result<Option<SessionState>> {
        let sessions = self.list()?;
        if let Some(latest_id) = sessions.first() {
            return self.load(latest_id);
        }
        Ok(None)
    }

    fn count(&self) -> Result<usize> {
        Ok(self.list()?.len())
    }

    fn clear(&self) -> Result<u64> {
        let ids = self.list()?;
        let count = ids.len() as u64;
        for id in ids {
            self.delete(&id)?;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageError;
    use tempfile::tempdir;

    #[test]
    fn save_and_load_session() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let session = SessionState::new(SessionId::new("test-session"), "test-profile");
        store.save(&session).unwrap();

        let loaded = store.load(&SessionId::new("test-session")).unwrap().unwrap();
        assert_eq!(loaded.id.0, "test-session");
        assert_eq!(loaded.profile_name, "test-profile");
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let loaded = store.load(&SessionId::new("nonexistent")).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn list_sessions() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&SessionState::new(SessionId::new("session-a"), "p1"))
            .unwrap();
        store
            .save(&SessionState::new(SessionId::new("session-b"), "p2"))
            .unwrap();

        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn delete_session() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let id = SessionId::new("to-delete");
        store
            .save(&SessionState::new(id.clone(), "profile"))
            .unwrap();
        assert!(store.exists(&id).unwrap());

        store.delete(&id).unwrap();
        assert!(!store.exists(&id).unwrap());
    }

    #[test]
    fn latest_returns_most_recent() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&SessionState::new(SessionId::new("old"), "p1"))
            .unwrap();
        // Small delay to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));
        store
            .save(&SessionState::new(SessionId::new("new"), "p2"))
            .unwrap();

        let latest = store.latest().unwrap().unwrap();
        assert_eq!(latest.id.0, "new");
    }

    #[test]
    fn clear_removes_all_sessions() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save(&SessionState::new(SessionId::new("s1"), "p"))
            .unwrap();
        store
            .save(&SessionState::new(SessionId::new("s2"), "p"))
            .unwrap();
        store
            .save(&SessionState::new(SessionId::new("s3"), "p"))
            .unwrap();

        let count = store.clear().unwrap();
        assert_eq!(count, 3);
        assert_eq!(store.count().unwrap(), 0);
    }

    // Path traversal security tests
    #[test]
    fn rejects_path_traversal_in_session_id() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.load(&SessionId::new("../etc/passwd"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_slash_in_session_id() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.exists(&SessionId::new("foo/bar"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_empty_session_id() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.load(&SessionId::new(""));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_null_byte_in_session_id() {
        let dir = tempdir().unwrap();
        let store = FileSessionStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.delete(&SessionId::new("sess\0ion"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }
}

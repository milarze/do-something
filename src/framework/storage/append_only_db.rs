//! Generic append-only database with index.
//!
//! Records are appended to a JSONL file. An index maps IDs to file offsets
//! for efficient retrieval without scanning the entire file.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Serialize, de::DeserializeOwned};

use super::error::{Result, StorageError};

/// Generic append-only database with index.
///
/// Appends records to a JSONL file and maintains an index for O(1) lookups.
pub struct AppendOnlyDb<T: Serialize + DeserializeOwned> {
    dir: PathBuf,
    /// Serializes write operations (record append + index update) so the
    /// recorded file offset always matches the actual line position.
    /// Without this, concurrent inserters can each capture the same offset
    /// before either writes, corrupting the ID→offset index.
    write_lock: Mutex<()>,
    _marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> AppendOnlyDb<T> {
    /// Open the database at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            write_lock: Mutex::new(()),
            _marker: PhantomData,
        })
    }

    /// Get the path to the records file.
    fn records_file(&self) -> PathBuf {
        self.dir.join("records.jsonl")
    }

    /// Get the path to the index file.
    fn index_file(&self) -> PathBuf {
        self.dir.join("index.json")
    }

    /// Append a record and return its generated ID.
    pub fn insert(&self, record: &T) -> Result<String>
    where
        T: HasId,
    {
        // Serialize the offset-capture + append + index-update sequence so
        // that the offset recorded in the index points at this record's line.
        let _guard = self
            .write_lock
            .lock()
            .map_err(|e| StorageError::LockError(format!("lock poisoned: {}", e)))?;

        let id = T::generate_id();

        // Open records file for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.records_file())?;
        let mut writer = BufWriter::new(file);

        // Get current position as offset
        let offset = writer.stream_position()?;

        // Write the record
        let mut record = record.clone();
        record.set_id(id.clone());
        let json = serde_json::to_string(&record)?;
        writeln!(writer, "{}", json)?;
        writer.flush()?;

        // Update index
        self.update_index(&id, offset)?;

        Ok(id)
    }

    /// Get a record by ID.
    pub fn get(&self, id: &str) -> Result<Option<T>> {
        let index = self.load_index()?;
        let offset = match index.get(id) {
            Some(offset) => *offset,
            None => return Ok(None),
        };

        let file = File::open(self.records_file())?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(offset))?;

        let mut line = String::new();
        reader.read_line(&mut line)?;

        if line.is_empty() {
            return Ok(None);
        }

        let record: T = serde_json::from_str(&line)?;
        Ok(Some(record))
    }

    /// Count total records.
    pub fn count(&self) -> Result<u64> {
        let path = self.records_file();
        if !path.exists() {
            return Ok(0);
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut count = 0u64;
        for line in reader.lines() {
            match line {
                Ok(_) => count += 1,
                Err(e) => tracing::warn!("error reading line in {}: {e}", path.display()),
            }
        }
        Ok(count)
    }

    /// Iterate all records.
    pub fn all(&self) -> Result<Vec<T>> {
        let path = self.records_file();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<T>(&line) {
                Ok(record) => records.push(record),
                Err(e) => tracing::warn!("skipping malformed record in {}: {e}", path.display()),
            }
        }

        Ok(records)
    }

    /// Rebuild the index from scratch.
    pub fn rebuild_index(&self) -> Result<()>
    where
        T: HasId,
    {
        // Hold the write lock so no concurrent insert can append a record
        // (and update the index) while we are rebuilding it.
        let _guard = self
            .write_lock
            .lock()
            .map_err(|e| StorageError::LockError(format!("lock poisoned: {}", e)))?;

        let path = self.records_file();
        if !path.exists() {
            fs::write(self.index_file(), "{}")?;
            return Ok(());
        }

        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let mut index: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

        loop {
            let mut line = String::new();
            let start_pos = reader.stream_position()?;
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }

            if let Ok(record) = serde_json::from_str::<T>(&line)
                && let Some(id) = record.get_id()
            {
                index.insert(id, start_pos);
            }
        }

        // Write index
        let json = serde_json::to_string(&index)?;
        fs::write(self.index_file(), json)?;

        Ok(())
    }

    /// Load the index from disk.
    fn load_index(&self) -> Result<std::collections::HashMap<String, u64>> {
        let path = self.index_file();
        if !path.exists() {
            return Ok(std::collections::HashMap::new());
        }

        let content = fs::read_to_string(path)?;
        let index = serde_json::from_str(&content).unwrap_or_default();
        Ok(index)
    }

    /// Update the index with a new entry.
    fn update_index(&self, id: &str, offset: u64) -> Result<()> {
        let mut index = self.load_index()?;
        index.insert(id.to_string(), offset);

        let json = serde_json::to_string(&index)?;
        fs::write(self.index_file(), json)?;

        Ok(())
    }
}

/// Trait for types that have an ID.
///
/// Types stored in AppendOnlyDb must implement this trait.
pub trait HasId: Clone {
    fn generate_id() -> String;
    fn get_id(&self) -> Option<String>;
    fn set_id(&mut self, id: String);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestRecord {
        id: Option<String>,
        value: String,
    }

    impl HasId for TestRecord {
        fn generate_id() -> String {
            format!("rec_{}", uuid::Uuid::new_v4().simple())
        }

        fn get_id(&self) -> Option<String> {
            self.id.clone()
        }

        fn set_id(&mut self, id: String) {
            self.id = Some(id);
        }
    }

    #[test]
    fn insert_and_get() {
        let dir = tempdir().unwrap();
        let db: AppendOnlyDb<TestRecord> = AppendOnlyDb::open(dir.path().to_path_buf()).unwrap();

        let record = TestRecord {
            id: None,
            value: "test value".to_string(),
        };

        let id = db.insert(&record).unwrap();
        let retrieved = db.get(&id).unwrap().unwrap();

        assert_eq!(retrieved.value, "test value");
        assert!(retrieved.id.is_some());
    }

    #[test]
    fn get_missing_returns_none() {
        let dir = tempdir().unwrap();
        let db: AppendOnlyDb<TestRecord> = AppendOnlyDb::open(dir.path().to_path_buf()).unwrap();

        let result = db.get("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn count_returns_correct_count() {
        let dir = tempdir().unwrap();
        let db: AppendOnlyDb<TestRecord> = AppendOnlyDb::open(dir.path().to_path_buf()).unwrap();

        assert_eq!(db.count().unwrap(), 0);

        for i in 0..5 {
            db.insert(&TestRecord {
                id: None,
                value: format!("value {}", i),
            })
            .unwrap();
        }

        assert_eq!(db.count().unwrap(), 5);
    }

    #[test]
    fn all_returns_all_records() {
        let dir = tempdir().unwrap();
        let db: AppendOnlyDb<TestRecord> = AppendOnlyDb::open(dir.path().to_path_buf()).unwrap();

        for i in 0..3 {
            db.insert(&TestRecord {
                id: None,
                value: format!("value {}", i),
            })
            .unwrap();
        }

        let all = db.all().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn rebuild_index_works() {
        let dir = tempdir().unwrap();
        let db: AppendOnlyDb<TestRecord> = AppendOnlyDb::open(dir.path().to_path_buf()).unwrap();

        let id1 = db
            .insert(&TestRecord {
                id: None,
                value: "first".to_string(),
            })
            .unwrap();
        let id2 = db
            .insert(&TestRecord {
                id: None,
                value: "second".to_string(),
            })
            .unwrap();

        // Delete index
        fs::remove_file(db.index_file()).unwrap();

        // Verify get fails
        assert!(db.get(&id1).unwrap().is_none());

        // Rebuild
        db.rebuild_index().unwrap();

        // Now get works
        let retrieved = db.get(&id1).unwrap().unwrap();
        assert_eq!(retrieved.value, "first");

        let retrieved = db.get(&id2).unwrap().unwrap();
        assert_eq!(retrieved.value, "second");
    }

    #[test]
    fn concurrent_inserts_are_all_retrievable() {
        // Without the internal write lock, concurrent inserters can each
        // capture the same stream offset before either writes, so the index
        // points at the wrong line and later `get`s return the wrong record.
        let dir = tempdir().unwrap();
        let db = Arc::new(AppendOnlyDb::<TestRecord>::open(dir.path().to_path_buf()).unwrap());

        const N: usize = 32;
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let db = db.clone();
            handles.push(std::thread::spawn(move || {
                db.insert(&TestRecord {
                    id: None,
                    value: format!("v{i}"),
                })
                .unwrap()
            }));
        }

        let mut ids = Vec::with_capacity(N);
        for h in handles {
            ids.push(h.join().unwrap());
        }

        // Every record must be present and retrievable by its recorded offset.
        assert_eq!(db.count().unwrap(), N as u64);
        assert_eq!(db.all().unwrap().len(), N);
        for id in &ids {
            let record = db.get(id).unwrap().unwrap();
            assert!(record.value.starts_with('v'));
        }
    }

    #[test]
    fn all_skips_malformed_lines() {
        let dir = tempdir().unwrap();
        let db: AppendOnlyDb<TestRecord> = AppendOnlyDb::open(dir.path().to_path_buf()).unwrap();

        db.insert(&TestRecord {
            id: None,
            value: "good".to_string(),
        })
        .unwrap();

        // Corrupt the records file with a malformed line.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(db.records_file())
            .unwrap();
        writeln!(file, "{{not valid json}}").unwrap();

        let all = db.all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].value, "good");
    }
}

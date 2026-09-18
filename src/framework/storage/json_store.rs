//! Generic JSON file store for single objects or keyed maps.
//!
//! Used for knowledge storage: site configs, user models, patterns.

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

/// Generic JSON file store.
///
/// Can store either a single object (e.g., patterns) or
/// a keyed map of objects (e.g., site configs by domain).
pub struct JsonStore<T: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> JsonStore<T> {
    /// Open the store at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            _marker: PhantomData,
        })
    }

    /// Get the path for a key.
    fn key_path(&self, key: &str) -> Result<PathBuf> {
        validate_path_component(key, "key")?;
        Ok(self.dir.join(format!("{}.json", key)))
    }

    /// Load a single object from the default file.
    ///
    /// The default file is named after the type, e.g., `patterns.json`.
    pub fn load(&self, default_filename: &str) -> Result<T>
    where
        T: Default,
    {
        validate_path_component(default_filename, "filename")?;
        let path = self.dir.join(format!("{}.json", default_filename));
        if !path.exists() {
            return Ok(T::default());
        }
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let value = serde_json::from_reader(reader)?;
        Ok(value)
    }

    /// Save a single object to the default file.
    pub fn save(&self, default_filename: &str, value: &T) -> Result<()> {
        validate_path_component(default_filename, "filename")?;
        let path = self.dir.join(format!("{}.json", default_filename));
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        Ok(())
    }

    /// Get a keyed object by name.
    pub fn get(&self, key: &str) -> Result<Option<T>> {
        let path = self.key_path(key)?;
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let value = serde_json::from_reader(reader)?;
        Ok(Some(value))
    }

    /// Save a keyed object by name.
    pub fn put(&self, key: &str, value: &T) -> Result<()> {
        let path = self.key_path(key)?;
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        Ok(())
    }

    /// Delete a keyed object.
    pub fn delete(&self, key: &str) -> Result<()> {
        let path = self.key_path(key)?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    /// List all keys.
    pub fn list(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(stem) = path.file_stem()
            {
                keys.push(stem.to_string_lossy().to_string());
            }
        }

        keys.sort();
        Ok(keys)
    }

    /// Check if a key exists.
    pub fn exists(&self, key: &str) -> Result<bool> {
        let path = self.key_path(key)?;
        Ok(path.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestConfig {
        name: String,
        value: u32,
    }

    impl Default for TestConfig {
        fn default() -> Self {
            Self {
                name: "default".to_string(),
                value: 0,
            }
        }
    }

    #[test]
    fn single_object_roundtrip() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        let config = TestConfig {
            name: "test".to_string(),
            value: 42,
        };

        store.save("config", &config).unwrap();
        let loaded = store.load("config").unwrap();

        assert_eq!(loaded, config);
    }

    #[test]
    fn load_returns_default_when_missing() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        let loaded = store.load("nonexistent").unwrap();
        assert_eq!(loaded.name, "default");
        assert_eq!(loaded.value, 0);
    }

    #[test]
    fn keyed_object_roundtrip() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        let config1 = TestConfig {
            name: "first".to_string(),
            value: 1,
        };
        let config2 = TestConfig {
            name: "second".to_string(),
            value: 2,
        };

        store.put("key1", &config1).unwrap();
        store.put("key2", &config2).unwrap();

        let loaded1 = store.get("key1").unwrap().unwrap();
        let loaded2 = store.get("key2").unwrap().unwrap();

        assert_eq!(loaded1, config1);
        assert_eq!(loaded2, config2);
    }

    #[test]
    fn get_returns_none_for_missing_key() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn list_returns_all_keys() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        store
            .put(
                "alpha",
                &TestConfig {
                    name: "a".to_string(),
                    value: 1,
                },
            )
            .unwrap();
        store
            .put(
                "beta",
                &TestConfig {
                    name: "b".to_string(),
                    value: 2,
                },
            )
            .unwrap();
        store
            .put(
                "gamma",
                &TestConfig {
                    name: "g".to_string(),
                    value: 3,
                },
            )
            .unwrap();

        let keys = store.list().unwrap();
        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn delete_removes_key() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        store
            .put(
                "to-delete",
                &TestConfig {
                    name: "test".to_string(),
                    value: 1,
                },
            )
            .unwrap();
        assert!(store.exists("to-delete").unwrap());

        store.delete("to-delete").unwrap();
        assert!(!store.exists("to-delete").unwrap());
    }

    #[test]
    fn rejects_path_traversal_in_key() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get("../etc/passwd");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_slash_in_key() {
        let dir = tempdir().unwrap();
        let store: JsonStore<TestConfig> = JsonStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get("foo/bar");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }
}

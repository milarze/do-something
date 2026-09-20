//! Compression checkpoint store.
//!
//! Persists the last compression run state for recovery and scheduling.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::error::Result;

/// Compression checkpoint store.
///
/// Persists the last compression run state to disk.
pub struct CheckpointStore {
    path: PathBuf,
}

impl CheckpointStore {
    /// Open the checkpoint store at the given path.
    pub fn open(path: PathBuf) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self { path })
    }

    /// Load the checkpoint from disk.
    ///
    /// Returns a default checkpoint if the file doesn't exist.
    pub fn load(&self) -> Result<Checkpoint> {
        if !self.path.exists() {
            return Ok(Checkpoint::default());
        }
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }

    /// Save the checkpoint to disk.
    pub fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let file = File::create(&self.path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, checkpoint)?;
        Ok(())
    }

    /// Update the last run timestamp and save.
    pub fn update_last_run(&self, last_run: DateTime<Utc>) -> Result<()> {
        let mut checkpoint = self.load()?;
        checkpoint.last_run = last_run;
        self.save(&checkpoint)
    }

    /// Update the last domain and save.
    pub fn update_domain(&self, domain: Option<String>) -> Result<()> {
        let mut checkpoint = self.load()?;
        checkpoint.last_domain = domain;
        self.save(&checkpoint)
    }
}

/// Compression checkpoint data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// When the last compression ran.
    #[serde(default = "Utc::now")]
    pub last_run: DateTime<Utc>,

    /// Domain that was last compressed.
    #[serde(default)]
    pub last_domain: Option<String>,

    /// Statistics from the last compression run.
    #[serde(default)]
    pub last_stats: Option<CompressionStats>,
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self {
            last_run: Utc::now(),
            last_domain: None,
            last_stats: None,
        }
    }
}

/// Statistics from a compression run.
#[must_use]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Number of signals processed.
    pub signals_processed: u64,

    /// Number of patterns extracted.
    pub patterns_extracted: u64,

    /// Number of configs updated.
    pub configs_updated: u64,

    /// Number of signals pruned.
    pub signals_pruned: u64,

    /// Time taken in milliseconds.
    pub time_ms: u64,

    /// When compression completed.
    #[serde(default = "Utc::now")]
    pub completed_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn load_returns_default_when_missing() {
        let dir = tempdir().unwrap();
        let store = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();

        let checkpoint = store.load().unwrap();
        assert!(checkpoint.last_domain.is_none());
        assert!(checkpoint.last_stats.is_none());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempdir().unwrap();
        let store = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();

        let checkpoint = Checkpoint {
            last_run: Utc::now(),
            last_domain: Some("example.com".to_string()),
            last_stats: Some(CompressionStats {
                signals_processed: 100,
                patterns_extracted: 5,
                configs_updated: 2,
                signals_pruned: 50,
                time_ms: 250,
                completed_at: Utc::now(),
            }),
        };

        store.save(&checkpoint).unwrap();
        let loaded = store.load().unwrap();

        assert_eq!(loaded.last_domain, Some("example.com".to_string()));
        assert_eq!(loaded.last_stats.as_ref().unwrap().signals_processed, 100);
    }

    #[test]
    fn update_last_run() {
        let dir = tempdir().unwrap();
        let store = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();

        let new_time = Utc::now();
        store.update_last_run(new_time).unwrap();

        let loaded = store.load().unwrap();
        // Compare timestamps (they may differ slightly due to serialization)
        assert!((loaded.last_run - new_time).num_seconds().abs() < 1);
    }

    #[test]
    fn update_domain() {
        let dir = tempdir().unwrap();
        let store = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();

        store.update_domain(Some("test.com".to_string())).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.last_domain, Some("test.com".to_string()));
    }
}

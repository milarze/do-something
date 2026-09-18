//! Generic append-only daily JSONL log.
//!
//! Parameterized by record type. Creates date-based JSONL files
//! for audit trails and later compression.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::path::PathBuf;

use chrono::{Duration, NaiveDate, Utc};
use serde::{Serialize, de::DeserializeOwned};

use super::error::Result;

/// Generic append-only daily JSONL log.
///
/// Creates one file per day, named `YYYY-MM-DD.jsonl`.
pub struct DailyLog<T: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> DailyLog<T> {
    /// Open the daily log at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            _marker: PhantomData,
        })
    }

    /// Get the path to a date's log file.
    fn log_file_for_date(&self, date: NaiveDate) -> PathBuf {
        let filename = format!("{}.jsonl", date.format("%Y-%m-%d"));
        self.dir.join(filename)
    }

    /// Get today's log file path.
    fn today_file(&self) -> PathBuf {
        self.log_file_for_date(Utc::now().date_naive())
    }

    /// Append a record to today's log.
    pub fn append(&self, record: &T) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.today_file())?;
        let mut writer = BufWriter::new(file);
        let json = serde_json::to_string(record)?;
        writeln!(writer, "{}", json)?;
        writer.flush()?;
        Ok(())
    }

    /// Read records from a date range (inclusive).
    pub fn read_range(&self, from: NaiveDate, to: NaiveDate) -> Result<Vec<T>> {
        let mut records = Vec::new();
        let mut current = from;

        while current <= to {
            let file_path = self.log_file_for_date(current);
            if file_path.exists() {
                let file = File::open(file_path)?;
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    let line = line?;
                    if let Ok(record) = serde_json::from_str::<T>(&line) {
                        records.push(record);
                    }
                }
            }
            current += Duration::days(1);
        }

        Ok(records)
    }

    /// Read records for a specific date.
    pub fn read_date(&self, date: NaiveDate) -> Result<Vec<T>> {
        let file_path = self.log_file_for_date(date);
        if !file_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if let Ok(record) = serde_json::from_str::<T>(&line) {
                records.push(record);
            }
        }

        Ok(records)
    }

    /// Count records matching a domain key in the last N days.
    /// Uses a predicate function to check if a record matches.
    pub fn count_matching<P>(&self, days: u32, predicate: P) -> Result<u64>
    where
        P: Fn(&T) -> bool,
    {
        let from = Utc::now().date_naive() - Duration::days(days as i64);
        let to = Utc::now().date_naive();

        let records = self.read_range(from, to)?;
        let count = records.iter().filter(|r| predicate(r)).count();
        Ok(count as u64)
    }

    /// Prune logs older than the specified number of days.
    pub fn prune(&self, older_than_days: u32) -> Result<u64> {
        let cutoff = Utc::now().date_naive() - Duration::days(older_than_days as i64);
        let mut pruned = 0u64;

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "jsonl").unwrap_or(false)
                && let Some(filename) = path.file_stem()
            {
                let filename_str = filename.to_string_lossy();
                if let Ok(date) = NaiveDate::parse_from_str(&filename_str, "%Y-%m-%d")
                    && date < cutoff
                {
                    let count = self.count_records_in_file(&path)?;
                    fs::remove_file(path)?;
                    pruned += count;
                }
            }
        }

        Ok(pruned)
    }

    /// Count records in a single file.
    fn count_records_in_file(&self, path: &std::path::Path) -> Result<u64> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().filter(|l| l.is_ok()).count() as u64)
    }

    /// List all available log dates.
    pub fn available_dates(&self) -> Result<Vec<NaiveDate>> {
        let mut dates = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "jsonl").unwrap_or(false)
                && let Some(filename) = path.file_stem()
            {
                let filename_str = filename.to_string_lossy();
                if let Ok(date) = NaiveDate::parse_from_str(&filename_str, "%Y-%m-%d") {
                    dates.push(date);
                }
            }
        }

        dates.sort();
        Ok(dates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestRecord {
        id: u32,
        message: String,
    }

    #[test]
    fn append_creates_daily_file() {
        let dir = tempdir().unwrap();
        let log: DailyLog<TestRecord> = DailyLog::open(dir.path().to_path_buf()).unwrap();

        log.append(&TestRecord {
            id: 1,
            message: "test".to_string(),
        })
        .unwrap();

        let today_path = log.today_file();
        assert!(today_path.exists());
    }

    #[test]
    fn read_range_returns_records() {
        let dir = tempdir().unwrap();
        let log: DailyLog<TestRecord> = DailyLog::open(dir.path().to_path_buf()).unwrap();

        log.append(&TestRecord {
            id: 1,
            message: "first".to_string(),
        })
        .unwrap();
        log.append(&TestRecord {
            id: 2,
            message: "second".to_string(),
        })
        .unwrap();

        let today = Utc::now().date_naive();
        let records = log.read_range(today, today).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].message, "first");
        assert_eq!(records[1].message, "second");
    }

    #[test]
    fn count_matching_filters_correctly() {
        let dir = tempdir().unwrap();
        let log: DailyLog<TestRecord> = DailyLog::open(dir.path().to_path_buf()).unwrap();

        for i in 0..5 {
            log.append(&TestRecord {
                id: i,
                message: "test".to_string(),
            })
            .unwrap();
        }
        for i in 5..8 {
            log.append(&TestRecord {
                id: i,
                message: "other".to_string(),
            })
            .unwrap();
        }

        let count = log.count_matching(1, |r| r.message == "test").unwrap();
        assert_eq!(count, 5);

        let count = log.count_matching(1, |r| r.message == "other").unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn prune_removes_old_files() {
        let dir = tempdir().unwrap();
        let log: DailyLog<TestRecord> = DailyLog::open(dir.path().to_path_buf()).unwrap();

        // Create a file for an old date
        let old_date = Utc::now().date_naive() - Duration::days(10);
        let old_file = log.log_file_for_date(old_date);
        let mut file = File::create(old_file).unwrap();
        writeln!(
            file,
            "{}",
            serde_json::to_string(&TestRecord {
                id: 0,
                message: "old".to_string()
            })
            .unwrap()
        )
        .unwrap();

        // Create today's file
        log.append(&TestRecord {
            id: 1,
            message: "today".to_string(),
        })
        .unwrap();

        let pruned = log.prune(5).unwrap();
        assert_eq!(pruned, 1);

        assert!(log.today_file().exists());
        assert!(!log.log_file_for_date(old_date).exists());
    }

    #[test]
    fn available_dates_lists_all_dates() {
        let dir = tempdir().unwrap();
        let log: DailyLog<TestRecord> = DailyLog::open(dir.path().to_path_buf()).unwrap();

        log.append(&TestRecord {
            id: 1,
            message: "today".to_string(),
        })
        .unwrap();

        let dates = log.available_dates().unwrap();
        assert_eq!(dates.len(), 1);
        assert_eq!(dates[0], Utc::now().date_naive());
    }
}

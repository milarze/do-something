//! File-based daily JSONL signal logging.
//!
//! Signals are appended to date-based JSONL files for audit trails
//! and later compression into knowledge.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use chrono::{Duration, NaiveDate, Utc};

use crate::models::signal::Signal;

use super::error::Result;
use super::file_utils::validate_path_component;
use super::traits::SignalStorage;

/// File-based daily JSONL files for signal logging.
#[derive(Debug)]
pub struct FileSignalLog {
    dir: PathBuf,
}

impl FileSignalLog {
    /// Open the signal log at the given directory.
    ///
    /// The directory path should be a safe, application-controlled path
    /// (e.g. from [`ConfigDir`](super::ConfigDir)).
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Get the path to a date's log file.
    ///
    /// Date filenames are validated by the `NaiveDate` type, ensuring
    /// they cannot contain path traversal characters.
    fn log_file_for_date(&self, date: NaiveDate) -> PathBuf {
        let filename = format!("{}.jsonl", date.format("%Y-%m-%d"));
        // Defensive validation even though NaiveDate guarantees safe format
        let _ = validate_path_component(&filename, "date filename");
        self.dir.join(filename)
    }

    /// Get today's log file path.
    fn today_file(&self) -> PathBuf {
        self.log_file_for_date(Utc::now().date_naive())
    }

    /// Count signals in a single file.
    fn count_signals_in_file(&self, path: &std::path::Path) -> Result<u64> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let count = reader.lines().filter(|l| l.is_ok()).count();
        Ok(count as u64)
    }
}

impl SignalStorage for FileSignalLog {
    fn append(&self, signal: &Signal) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.today_file())?;
        let mut writer = BufWriter::new(file);
        let json = serde_json::to_string(signal)?;
        writeln!(writer, "{}", json)?;
        writer.flush()?; // Explicit flush for durability
        Ok(())
    }

    fn read_range(&self, from: NaiveDate, to: NaiveDate) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();
        let mut current = from;

        while current <= to {
            let file_path = self.log_file_for_date(current);
            if file_path.exists() {
                let file = File::open(file_path)?;
                let reader = BufReader::new(file);
                for line in reader.lines() {
                    let line = line?;
                    if let Ok(signal) = serde_json::from_str::<Signal>(&line) {
                        signals.push(signal);
                    }
                }
            }
            current += Duration::days(1);
        }

        Ok(signals)
    }

    fn count_for_domain(&self, domain: &str, days: u32) -> Result<u64> {
        let from = Utc::now().date_naive() - Duration::days(days as i64);
        let to = Utc::now().date_naive();

        let signals = self.read_range(from, to)?;
        let count = signals
            .iter()
            .filter(|s| s.domain.as_deref() == Some(domain))
            .count();

        Ok(count as u64)
    }

    fn prune(&self, older_than_days: u32) -> Result<u64> {
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
                    let count = self.count_signals_in_file(&path)?;
                    fs::remove_file(path)?;
                    pruned += count;
                }
            }
        }

        Ok(pruned)
    }

    fn read_date(&self, date: NaiveDate) -> Result<Vec<Signal>> {
        let file_path = self.log_file_for_date(date);
        if !file_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut signals = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if let Ok(signal) = serde_json::from_str::<Signal>(&line) {
                signals.push(signal);
            }
        }

        Ok(signals)
    }

    fn available_dates(&self) -> Result<Vec<NaiveDate>> {
        let mut dates = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "jsonl").unwrap_or(false)
                && let Some(filename) = path.file_stem()
            {
                let filename_str = filename.to_string_lossy();
                if let Ok(naive) = NaiveDate::parse_from_str(&filename_str, "%Y-%m-%d") {
                    dates.push(naive);
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
    use crate::models::signal::SignalType;
    use tempfile::tempdir;

    #[test]
    fn append_creates_daily_file() {
        let dir = tempdir().unwrap();
        let log = FileSignalLog::open(dir.path().to_path_buf()).unwrap();

        let signal = Signal::new(SignalType::ParseSuccess {
            method: crate::models::ParseMethod::SchemaOrg,
            time_ms: 350,
        });

        log.append(&signal).unwrap();

        let today_path = log.today_file();
        assert!(today_path.exists());
    }

    #[test]
    fn read_range_returns_signals() {
        let dir = tempdir().unwrap();
        let log = FileSignalLog::open(dir.path().to_path_buf()).unwrap();

        let signal1 = Signal::new(SignalType::ParseSuccess {
            method: crate::models::ParseMethod::SchemaOrg,
            time_ms: 100,
        })
        .with_domain("example.com");

        let signal2 = Signal::new(SignalType::ParseSuccess {
            method: crate::models::ParseMethod::Selectors,
            time_ms: 200,
        })
        .with_domain("other.com");

        log.append(&signal1).unwrap();
        log.append(&signal2).unwrap();

        let today = Utc::now().date_naive();
        let signals = log.read_range(today, today).unwrap();

        assert_eq!(signals.len(), 2);
    }

    #[test]
    fn count_for_domain_filters_correctly() {
        let dir = tempdir().unwrap();
        let log = FileSignalLog::open(dir.path().to_path_buf()).unwrap();

        for _ in 0..5 {
            log.append(
                &Signal::new(SignalType::ParseSuccess {
                    method: crate::models::ParseMethod::SchemaOrg,
                    time_ms: 100,
                })
                .with_domain("example.com"),
            )
            .unwrap();
        }

        for _ in 0..3 {
            log.append(
                &Signal::new(SignalType::ParseSuccess {
                    method: crate::models::ParseMethod::SchemaOrg,
                    time_ms: 100,
                })
                .with_domain("other.com"),
            )
            .unwrap();
        }

        let count = log.count_for_domain("example.com", 1).unwrap();
        assert_eq!(count, 5);

        let count = log.count_for_domain("other.com", 1).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn prune_removes_old_files() {
        let dir = tempdir().unwrap();
        let log = FileSignalLog::open(dir.path().to_path_buf()).unwrap();

        // Create a file for an old date
        let old_date = Utc::now().date_naive() - Duration::days(10);
        let old_file = log.log_file_for_date(old_date);
        let mut file = File::create(old_file).unwrap();
        writeln!(
            file,
            "{}",
            serde_json::to_string(&Signal::new(SignalType::ParseSuccess {
                method: crate::models::ParseMethod::SchemaOrg,
                time_ms: 100
            }))
            .unwrap()
        )
        .unwrap();

        // Create today's file
        log.append(&Signal::new(SignalType::ParseSuccess {
            method: crate::models::ParseMethod::SchemaOrg,
            time_ms: 100,
        }))
        .unwrap();

        let pruned = log.prune(5).unwrap();
        assert_eq!(pruned, 1);

        assert!(log.today_file().exists());
        assert!(!log.log_file_for_date(old_date).exists());
    }
}

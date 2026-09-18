//! Recipe signal log built over framework DailyLog.
//!
//! Provides recipe-specific signal logging.

use std::path::PathBuf;

use chrono::NaiveDate;

use crate::framework::storage::DailyLog;

use super::super::models::signal::RecipeSignal;
use super::traits::SignalStorage;

/// Daily JSONL signal log for recipes.
pub struct RecipeSignalLog {
    inner: DailyLog<RecipeSignal>,
}

impl RecipeSignalLog {
    /// Open the signal log at the given directory.
    pub fn open(dir: PathBuf) -> crate::framework::storage::Result<Self> {
        let inner = DailyLog::open(dir)?;
        Ok(Self { inner })
    }
}

impl SignalStorage for RecipeSignalLog {
    fn append(&self, signal: &RecipeSignal) -> crate::framework::storage::Result<()> {
        self.inner.append(signal)
    }

    fn read_range(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> crate::framework::storage::Result<Vec<RecipeSignal>> {
        self.inner.read_range(from, to)
    }

    fn count_for_domain(&self, domain: &str, days: u32) -> crate::framework::storage::Result<u64> {
        self.inner
            .count_matching(days, |s| s.domain.as_deref() == Some(domain))
    }

    fn prune(&self, older_than_days: u32) -> crate::framework::storage::Result<u64> {
        self.inner.prune(older_than_days)
    }

    fn read_date(&self, date: NaiveDate) -> crate::framework::storage::Result<Vec<RecipeSignal>> {
        self.inner.read_date(date)
    }

    fn available_dates(&self) -> crate::framework::storage::Result<Vec<NaiveDate>> {
        self.inner.available_dates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::recipe::models::ParseMethod;
    use crate::domain::recipe::models::signal::RecipeSignalType;
    use chrono::Utc;
    use tempfile::tempdir;

    #[test]
    fn append_creates_daily_file() {
        let dir = tempdir().unwrap();
        let log = RecipeSignalLog::open(dir.path().to_path_buf()).unwrap();

        let signal = RecipeSignal::new(RecipeSignalType::ParseSuccess {
            method: ParseMethod::SchemaOrg,
            time_ms: 350,
        });

        log.append(&signal).unwrap();

        // Verify file exists
        let today = Utc::now().date_naive();
        let expected_file = dir
            .path()
            .join(format!("{}.jsonl", today.format("%Y-%m-%d")));
        assert!(expected_file.exists());
    }

    #[test]
    fn read_range_returns_signals() {
        let dir = tempdir().unwrap();
        let log = RecipeSignalLog::open(dir.path().to_path_buf()).unwrap();

        let signal1 = RecipeSignal::new(RecipeSignalType::ParseSuccess {
            method: ParseMethod::SchemaOrg,
            time_ms: 100,
        })
        .with_domain("example.com");

        let signal2 = RecipeSignal::new(RecipeSignalType::ParseSuccess {
            method: ParseMethod::Selectors,
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
        let log = RecipeSignalLog::open(dir.path().to_path_buf()).unwrap();

        for _ in 0..5 {
            log.append(
                &RecipeSignal::new(RecipeSignalType::ParseSuccess {
                    method: ParseMethod::SchemaOrg,
                    time_ms: 100,
                })
                .with_domain("example.com"),
            )
            .unwrap();
        }

        for _ in 0..3 {
            log.append(
                &RecipeSignal::new(RecipeSignalType::ParseSuccess {
                    method: ParseMethod::SchemaOrg,
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
}

//! Compression scheduling: when should the runner trigger?
//!
//! [`CompressionRunner::should_run`] is split into its own file for clarity,
//! but is an inherent method on [`CompressionRunner`].

use chrono::Utc;

use crate::framework::storage::StorageError;

use super::domain::LearningDomain;
use super::runner::CompressionRunner;

impl<D: LearningDomain> CompressionRunner<D> {
    /// Whether compression should run for `domain` now.
    ///
    /// Returns `true` when either:
    /// - the schedule interval has elapsed since the last run, or
    /// - enough signals have accumulated for this domain.
    ///
    /// These correspond to the time-based and signal-count triggers in the
    /// compression pipeline specification.
    pub fn should_run(&self, domain: &str) -> Result<bool, StorageError> {
        // Condition 1: schedule interval exceeded.
        let checkpoint = self.checkpoint.load()?;
        let elapsed = Utc::now().signed_duration_since(checkpoint.last_run);
        if elapsed > chrono::Duration::days(self.config.schedule_interval_days.get() as i64) {
            return Ok(true);
        }

        // Condition 2: signal count threshold met.
        let count = self
            .signal_log
            .count_matching(self.config.lookback_days.get(), |s| {
                D::signal_domain(s) == Some(domain)
            })?;
        if count >= self.config.signal_count_threshold {
            return Ok(true);
        }

        Ok(false)
    }
}

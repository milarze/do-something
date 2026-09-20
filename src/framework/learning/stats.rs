//! Compression configuration and result types.
//!
//! [`CompressionStats`] is defined in the storage layer
//! ([`crate::framework::storage::CompressionStats`]) because the checkpoint
//! store persists it. It is re-exported here for the learning layer's
//! convenience; the learning layer depends on storage, not the reverse.

pub use crate::framework::storage::CompressionStats;

/// Configuration for the compression pipeline.
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    /// Minimum signals required before pattern extraction.
    pub min_sample_size: u64,

    /// Confidence threshold for persisting a pattern to knowledge.
    pub confidence_threshold: f64,

    /// Days of signals to process per run.
    pub lookback_days: u32,

    /// Retention period for processed signals (pruned after).
    pub retention_days: u32,

    /// Minimum success rate to recommend a method.
    pub min_success_rate: f64,

    /// Maximum days between scheduled runs.
    pub schedule_interval_days: u32,

    /// Signal count that triggers an early run.
    pub signal_count_threshold: u64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 10,
            confidence_threshold: 0.7,
            lookback_days: 7,
            retention_days: 30,
            min_success_rate: 0.8,
            schedule_interval_days: 7,
            signal_count_threshold: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_thresholds() {
        let c = CompressionConfig::default();
        assert_eq!(c.min_sample_size, 10);
        assert_eq!(c.confidence_threshold, 0.7);
        assert_eq!(c.lookback_days, 7);
        assert_eq!(c.retention_days, 30);
        assert_eq!(c.min_success_rate, 0.8);
        assert_eq!(c.schedule_interval_days, 7);
        assert_eq!(c.signal_count_threshold, 100);
    }

    #[test]
    fn config_is_clone() {
        let c = CompressionConfig::default();
        let _c2 = c.clone();
    }
}

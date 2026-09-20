//! Compression configuration and result types.
//!
//! [`CompressionStats`] is defined in the storage layer
//! ([`crate::framework::storage::CompressionStats`]) because the checkpoint
//! store persists it. It is re-exported here for the learning layer's
//! convenience; the learning layer depends on storage, not the reverse.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use crate::framework::storage::CompressionStats;

/// Errors from invalid configuration values.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("schedule_interval_days must be 1-365, got {value}")]
    InvalidScheduleInterval { value: u32 },

    #[error("retention_days must be 1-365, got {value}")]
    InvalidRetentionDays { value: u32 },

    #[error("lookback_days must be 1-365, got {value}")]
    InvalidLookbackDays { value: u32 },

    #[error("probability must be 0.0-1.0, got {value}")]
    InvalidProbability { value: f64 },
}

/// Schedule interval in days. Must be 1-365.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleIntervalDays(u32);

impl ScheduleIntervalDays {
    /// Create a new schedule interval. Returns error if outside 1-365.
    pub fn new(value: u32) -> Result<Self, ConfigError> {
        if (1..=365).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ConfigError::InvalidScheduleInterval { value })
        }
    }

    /// Get the inner value.
    pub fn get(&self) -> u32 {
        self.0
    }
}

impl Default for ScheduleIntervalDays {
    fn default() -> Self {
        Self(7)
    }
}

impl Serialize for ScheduleIntervalDays {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for ScheduleIntervalDays {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl From<ScheduleIntervalDays> for u32 {
    fn from(value: ScheduleIntervalDays) -> u32 {
        value.0
    }
}

/// Retention period in days. Must be 1-365.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionDays(u32);

impl RetentionDays {
    /// Create a new retention period. Returns error if outside 1-365.
    pub fn new(value: u32) -> Result<Self, ConfigError> {
        if (1..=365).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ConfigError::InvalidRetentionDays { value })
        }
    }

    /// Get the inner value.
    pub fn get(&self) -> u32 {
        self.0
    }
}

impl Default for RetentionDays {
    fn default() -> Self {
        Self(30)
    }
}

impl Serialize for RetentionDays {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for RetentionDays {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl From<RetentionDays> for u32 {
    fn from(value: RetentionDays) -> u32 {
        value.0
    }
}

/// Lookback period in days. Must be 1-365.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookbackDays(u32);

impl LookbackDays {
    /// Create a new lookback period. Returns error if outside 1-365.
    pub fn new(value: u32) -> Result<Self, ConfigError> {
        if (1..=365).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ConfigError::InvalidLookbackDays { value })
        }
    }

    /// Get the inner value.
    pub fn get(&self) -> u32 {
        self.0
    }
}

impl Default for LookbackDays {
    fn default() -> Self {
        Self(7)
    }
}

impl Serialize for LookbackDays {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for LookbackDays {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl From<LookbackDays> for u32 {
    fn from(value: LookbackDays) -> u32 {
        value.0
    }
}

/// A probability or rate value in [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Create a new probability. Returns error if outside 0.0-1.0.
    pub fn new(value: f64) -> Result<Self, ConfigError> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ConfigError::InvalidProbability { value })
        }
    }

    /// Get the inner value.
    pub fn get(&self) -> f64 {
        self.0
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self(0.7)
    }
}

impl Serialize for Probability {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl From<Probability> for f64 {
    fn from(value: Probability) -> f64 {
        value.0
    }
}

/// Configuration for the compression pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Minimum signals required before pattern extraction.
    pub min_sample_size: u64,

    /// Confidence threshold for persisting a pattern to knowledge.
    pub confidence_threshold: Probability,

    /// Days of signals to process per run.
    pub lookback_days: LookbackDays,

    /// Retention period for processed signals (pruned after).
    pub retention_days: RetentionDays,

    /// Minimum success rate to recommend a method.
    pub min_success_rate: Probability,

    /// Maximum days between scheduled runs.
    pub schedule_interval_days: ScheduleIntervalDays,

    /// Signal count that triggers an early run.
    pub signal_count_threshold: u64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 10,
            confidence_threshold: Probability::default(),
            lookback_days: LookbackDays::default(),
            retention_days: RetentionDays::default(),
            min_success_rate: Probability::new(0.8).unwrap(),
            schedule_interval_days: ScheduleIntervalDays::default(),
            signal_count_threshold: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ScheduleIntervalDays ---

    #[test]
    fn schedule_interval_valid_range() {
        assert!(ScheduleIntervalDays::new(1).is_ok());
        assert!(ScheduleIntervalDays::new(365).is_ok());
        assert!(ScheduleIntervalDays::new(0).is_err());
        assert!(ScheduleIntervalDays::new(366).is_err());
    }

    #[test]
    fn schedule_interval_serialization_roundtrip() {
        let interval = ScheduleIntervalDays::new(30).unwrap();
        let json = serde_json::to_string(&interval).unwrap();
        let parsed: ScheduleIntervalDays = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.get(), 30);
    }

    #[test]
    fn schedule_interval_deserialization_rejects_invalid() {
        let result: Result<ScheduleIntervalDays, _> = serde_json::from_str("500");
        assert!(result.is_err());
    }

    // --- RetentionDays ---

    #[test]
    fn retention_days_valid_range() {
        assert!(RetentionDays::new(1).is_ok());
        assert!(RetentionDays::new(365).is_ok());
        assert!(RetentionDays::new(0).is_err());
        assert!(RetentionDays::new(366).is_err());
    }

    // --- LookbackDays ---

    #[test]
    fn lookback_days_valid_range() {
        assert!(LookbackDays::new(1).is_ok());
        assert!(LookbackDays::new(365).is_ok());
        assert!(LookbackDays::new(0).is_err());
        assert!(LookbackDays::new(366).is_err());
    }

    // --- Probability ---

    #[test]
    fn probability_valid_range() {
        assert!(Probability::new(0.0).is_ok());
        assert!(Probability::new(1.0).is_ok());
        assert!(Probability::new(0.5).is_ok());
        assert!(Probability::new(-0.1).is_err());
        assert!(Probability::new(1.1).is_err());
    }

    #[test]
    fn probability_serialization_roundtrip() {
        let prob = Probability::new(0.85).unwrap();
        let json = serde_json::to_string(&prob).unwrap();
        let parsed: Probability = serde_json::from_str(&json).unwrap();
        assert!((parsed.get() - 0.85).abs() < 1e-9);
    }

    #[test]
    fn probability_deserialization_rejects_invalid() {
        let result: Result<Probability, _> = serde_json::from_str("1.5");
        assert!(result.is_err());
    }

    // --- CompressionConfig ---

    #[test]
    fn default_config_has_sensible_thresholds() {
        let c = CompressionConfig::default();
        assert_eq!(c.min_sample_size, 10);
        assert!((c.confidence_threshold.get() - 0.7).abs() < 1e-9);
        assert_eq!(c.lookback_days.get(), 7);
        assert_eq!(c.retention_days.get(), 30);
        assert!((c.min_success_rate.get() - 0.8).abs() < 1e-9);
        assert_eq!(c.schedule_interval_days.get(), 7);
        assert_eq!(c.signal_count_threshold, 100);
    }

    #[test]
    fn config_is_clone() {
        let c = CompressionConfig::default();
        let _c2 = c.clone();
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = CompressionConfig {
            min_sample_size: 20,
            confidence_threshold: Probability::new(0.9).unwrap(),
            lookback_days: LookbackDays::new(14).unwrap(),
            retention_days: RetentionDays::new(60).unwrap(),
            min_success_rate: Probability::new(0.75).unwrap(),
            schedule_interval_days: ScheduleIntervalDays::new(3).unwrap(),
            signal_count_threshold: 50,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: CompressionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.min_sample_size, 20);
        assert_eq!(parsed.schedule_interval_days.get(), 3);
        assert_eq!(parsed.retention_days.get(), 60);
    }

    #[test]
    fn config_deserialization_rejects_invalid_schedule_interval() {
        let json = r#"{
            "min_sample_size": 10,
            "confidence_threshold": 0.7,
            "lookback_days": 7,
            "retention_days": 30,
            "min_success_rate": 0.8,
            "schedule_interval_days": 500,
            "signal_count_threshold": 100
        }"#;

        let result: Result<CompressionConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn config_deserialization_rejects_invalid_probability() {
        let json = r#"{
            "min_sample_size": 10,
            "confidence_threshold": 1.5,
            "lookback_days": 7,
            "retention_days": 30,
            "min_success_rate": 0.8,
            "schedule_interval_days": 7,
            "signal_count_threshold": 100
        }"#;

        let result: Result<CompressionConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}

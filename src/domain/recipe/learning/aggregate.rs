//! Recipe signal aggregation.
//!
//! Groups parse signals by method, counts successes/failures, and tracks
//! common errors. Produces [`RecipeAggregatedStats`] for pattern extraction.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::domain::recipe::models::recipe::ParseMethod;
use crate::domain::recipe::models::signal::{RecipeSignal, RecipeSignalType};
use crate::framework::learning::domain::DomainStats;

/// Per-method statistics accumulated during aggregation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MethodStats {
    pub successes: u64,
    pub failures: u64,
    /// Running average parse time (ms) for successful parses.
    pub avg_time_ms: f64,
    /// Error message -> occurrence count.
    pub errors: HashMap<String, u64>,
}

impl MethodStats {
    /// Total samples for this method.
    pub fn total(&self) -> u64 {
        self.successes + self.failures
    }

    /// Success rate in `[0, 1]`.
    pub fn success_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            self.successes as f64 / total as f64
        }
    }

    fn record_success(&mut self, time_ms: u64) {
        self.successes += 1;
        // Incremental running average: avoids storing every sample.
        let n = self.successes as f64;
        self.avg_time_ms += (time_ms as f64 - self.avg_time_ms) / n;
    }

    fn record_failure(&mut self, error: &str) {
        self.failures += 1;
        *self.errors.entry(error.to_string()).or_default() += 1;
    }
}

/// A frequent error observed during aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorFrequency {
    pub error: String,
    pub count: u64,
    /// Percentage of total errors (0-100).
    pub percentage: f64,
}

/// Aggregated statistics for recipe signals (single domain).
#[derive(Debug, Clone)]
pub struct RecipeAggregatedStats {
    pub domain: String,
    pub by_method: HashMap<ParseMethod, MethodStats>,
    pub total_successes: u64,
    pub total_failures: u64,
    pub common_errors: Vec<ErrorFrequency>,
    pub time_period: (DateTime<Utc>, DateTime<Utc>),
}

impl Default for RecipeAggregatedStats {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            domain: String::new(),
            by_method: HashMap::new(),
            total_successes: 0,
            total_failures: 0,
            common_errors: Vec::new(),
            time_period: (now, now),
        }
    }
}

impl RecipeAggregatedStats {
    /// Overall success rate in `[0, 1]`.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            0.0
        } else {
            self.total_successes as f64 / total as f64
        }
    }

    /// Overall failure rate in `[0, 1]`.
    pub fn failure_rate(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            0.0
        } else {
            self.total_failures as f64 / total as f64
        }
    }

    /// The method with the highest success rate (among tried methods).
    pub fn best_method(&self) -> Option<(ParseMethod, f64)> {
        self.by_method
            .iter()
            .filter(|(_, s)| s.total() > 0)
            .max_by(|a, b| {
                a.1.success_rate()
                    .partial_cmp(&b.1.success_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(m, s)| (*m, s.success_rate()))
    }
}

impl DomainStats for RecipeAggregatedStats {
    fn total_count(&self) -> u64 {
        self.total_successes + self.total_failures
    }
}

/// Aggregate recipe signals into stats.
///
/// Signals are grouped by parse method; successes, failures, parse times,
/// and errors are tallied. Common errors are sorted by descending frequency.
/// The `domain` field is taken from the first signal carrying one.
pub fn aggregate_recipe_signals(signals: &[RecipeSignal]) -> RecipeAggregatedStats {
    if signals.is_empty() {
        return RecipeAggregatedStats::default();
    }

    let domain = signals
        .iter()
        .find_map(|s| s.domain.clone())
        .unwrap_or_default();

    let mut by_method: HashMap<ParseMethod, MethodStats> = HashMap::new();
    let mut total_successes = 0u64;
    let mut total_failures = 0u64;
    let mut min_time = signals[0].timestamp;
    let mut max_time = signals[0].timestamp;

    for signal in signals {
        if signal.timestamp < min_time {
            min_time = signal.timestamp;
        }
        if signal.timestamp > max_time {
            max_time = signal.timestamp;
        }

        match &signal.signal_type {
            RecipeSignalType::ParseSuccess { method, time_ms } => {
                by_method
                    .entry(*method)
                    .or_default()
                    .record_success(*time_ms);
                total_successes += 1;
            }
            RecipeSignalType::ParseFailure { method, error, .. } => {
                by_method.entry(*method).or_default().record_failure(error);
                total_failures += 1;
            }
            _ => {}
        }
    }

    let common_errors = build_common_errors(&by_method);

    RecipeAggregatedStats {
        domain,
        by_method,
        total_successes,
        total_failures,
        common_errors,
        time_period: (min_time, max_time),
    }
}

/// Build the sorted common-errors list from per-method stats.
fn build_common_errors(by_method: &HashMap<ParseMethod, MethodStats>) -> Vec<ErrorFrequency> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    for stats in by_method.values() {
        for (err, &count) in &stats.errors {
            *counts.entry(err.clone()).or_default() += count;
        }
    }
    let total: u64 = counts.values().sum();
    let mut freqs: Vec<ErrorFrequency> = counts
        .into_iter()
        .map(|(error, count)| ErrorFrequency {
            percentage: if total == 0 {
                0.0
            } else {
                count as f64 / total as f64 * 100.0
            },
            error,
            count,
        })
        .collect();
    freqs.sort_by_key(|f| std::cmp::Reverse(f.count));
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn success(domain: &str, method: ParseMethod, time_ms: u64) -> RecipeSignal {
        RecipeSignal::new(RecipeSignalType::ParseSuccess { method, time_ms }).with_domain(domain)
    }

    fn failure(domain: &str, method: ParseMethod, error: &str) -> RecipeSignal {
        RecipeSignal::new(RecipeSignalType::ParseFailure {
            method,
            error: error.to_string(),
            attempted_methods: vec![],
        })
        .with_domain(domain)
    }

    #[test]
    fn aggregate_empty_returns_default() {
        let stats = aggregate_recipe_signals(&[]);
        assert_eq!(stats.total_count(), 0);
        assert_eq!(stats.domain, "");
        assert!(stats.by_method.is_empty());
        assert!(stats.common_errors.is_empty());
    }

    #[test]
    fn aggregate_counts_successes_and_failures() {
        let signals = vec![
            success("a.com", ParseMethod::SchemaOrg, 100),
            success("a.com", ParseMethod::SchemaOrg, 200),
            failure("a.com", ParseMethod::SchemaOrg, "timeout"),
        ];
        let stats = aggregate_recipe_signals(&signals);

        assert_eq!(stats.total_successes, 2);
        assert_eq!(stats.total_failures, 1);
        assert_eq!(stats.total_count(), 3);
        assert_eq!(stats.domain, "a.com");

        let method = &stats.by_method[&ParseMethod::SchemaOrg];
        assert_eq!(method.successes, 2);
        assert_eq!(method.failures, 1);
        assert!((method.success_rate() - (2.0 / 3.0)).abs() < 1e-9);
        // avg of 100 and 200 = 150
        assert!((method.avg_time_ms - 150.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_ignores_non_parse_signals() {
        use crate::domain::recipe::models::signal::Sentiment;
        let mut s = RecipeSignal::new(RecipeSignalType::ExplicitFeedback {
            feedback: "nice".to_string(),
            url: None,
            recipe_id: None,
            sentiment: Sentiment::Positive,
        })
        .with_domain("a.com");
        s.timestamp = Utc::now();
        let stats = aggregate_recipe_signals(&[s]);
        assert_eq!(stats.total_count(), 0);
        assert!(stats.by_method.is_empty());
    }

    #[test]
    fn aggregate_groups_errors_across_methods() {
        let signals = vec![
            failure("a.com", ParseMethod::SchemaOrg, "timeout"),
            failure("a.com", ParseMethod::Selectors, "timeout"),
            failure("a.com", ParseMethod::SchemaOrg, "no_ingredients"),
        ];
        let stats = aggregate_recipe_signals(&signals);

        assert_eq!(stats.common_errors.len(), 2);
        // Sorted by descending count: timeout (2) first.
        assert_eq!(stats.common_errors[0].error, "timeout");
        assert_eq!(stats.common_errors[0].count, 2);
        assert!(
            (stats.common_errors[0].percentage - (2.0 / 3.0 * 100.0)).abs() < 1e-6,
            "got {}",
            stats.common_errors[0].percentage
        );
        assert_eq!(stats.common_errors[1].error, "no_ingredients");
        assert_eq!(stats.common_errors[1].count, 1);
    }

    #[test]
    fn aggregate_records_time_period() {
        let mut signals = vec![
            success("a.com", ParseMethod::SchemaOrg, 100),
            success("a.com", ParseMethod::SchemaOrg, 200),
        ];
        signals[0].timestamp = Utc::now() - chrono::Duration::hours(2);
        signals[1].timestamp = Utc::now();
        let stats = aggregate_recipe_signals(&signals);

        assert_eq!(stats.time_period.0, signals[0].timestamp);
        assert_eq!(stats.time_period.1, signals[1].timestamp);
    }

    #[test]
    fn best_method_returns_highest_success_rate() {
        let signals = vec![
            success("a.com", ParseMethod::SchemaOrg, 100),
            success("a.com", ParseMethod::SchemaOrg, 100),
            failure("a.com", ParseMethod::Selectors, "err"),
        ];
        let stats = aggregate_recipe_signals(&signals);

        let (method, rate) = stats.best_method().unwrap();
        assert_eq!(method, ParseMethod::SchemaOrg);
        assert!((rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn best_method_none_when_no_signals() {
        let stats = RecipeAggregatedStats::default();
        assert!(stats.best_method().is_none());
    }

    #[test]
    fn method_stats_running_average() {
        let mut ms = MethodStats::default();
        ms.record_success(100);
        assert!((ms.avg_time_ms - 100.0).abs() < 1e-9);
        ms.record_success(300);
        // avg(100, 300) = 200
        assert!((ms.avg_time_ms - 200.0).abs() < 1e-9);
        ms.record_success(200);
        // avg(100, 300, 200) = 200
        assert!((ms.avg_time_ms - 200.0).abs() < 1e-9);
    }

    #[test]
    fn success_and_failure_rates() {
        let stats = RecipeAggregatedStats {
            domain: "a.com".into(),
            by_method: HashMap::new(),
            total_successes: 3,
            total_failures: 1,
            common_errors: vec![],
            time_period: (Utc::now(), Utc::now()),
        };
        assert!((stats.success_rate() - 0.75).abs() < 1e-9);
        assert!((stats.failure_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn rates_are_zero_with_no_data() {
        let stats = RecipeAggregatedStats::default();
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.failure_rate(), 0.0);
    }
}

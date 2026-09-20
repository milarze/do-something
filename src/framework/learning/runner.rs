//! Generic compression loop orchestration.
//!
//! The runner drives the four-stage loop: aggregate (domain) → extract
//! patterns (domain) → update knowledge (domain) → prune old signals
//! (framework). It owns checkpointing and stats reporting; scheduling lives
//! in [`super::scheduler`].
//!
//! The framework reads signals from the log itself (a storage concern) and
//! hands them to the domain's pure [`LearningDomain::aggregate`]. This keeps
//! the domain stages free of I/O and trivially unit-testable.

use std::sync::Arc;

use chrono::{Duration, Utc};

use crate::framework::storage::{Checkpoint, CheckpointStore, CompressionStats, DailyLog};

use super::domain::{DomainStats, LearningDomain};
use super::stats::CompressionConfig;

/// Generic compression runner parameterized by a [`LearningDomain`].
pub struct CompressionRunner<D: LearningDomain> {
    pub(super) domain: Arc<D>,
    pub(super) signal_log: Arc<DailyLog<D::Signal>>,
    pub(super) config: CompressionConfig,
    pub(super) checkpoint: CheckpointStore,
}

impl<D: LearningDomain> CompressionRunner<D> {
    /// Create a new runner.
    pub fn new(
        domain: Arc<D>,
        signal_log: Arc<DailyLog<D::Signal>>,
        config: CompressionConfig,
        checkpoint: CheckpointStore,
    ) -> Self {
        Self {
            domain,
            signal_log,
            config,
            checkpoint,
        }
    }

    /// Run the full compression pipeline.
    ///
    /// Reads signals for `domain` (or all domains if `None`) over the last
    /// `days`, then runs aggregate → extract → update → prune, and persists
    /// the checkpoint.
    pub fn run(&self, domain: Option<&str>, days: u32) -> anyhow::Result<CompressionStats> {
        let start = std::time::Instant::now();

        // Read signals (framework responsibility).
        let signals = self.read_signals(domain, days)?;

        // Stage 1: aggregate (domain).
        let stats = self.domain.aggregate(&signals);

        // Stage 2: extract patterns (domain).
        let patterns = self.domain.extract_patterns(&stats);

        // Stage 3: update knowledge (domain).
        let configs_updated = self.domain.update_knowledge(&patterns)?;

        // Stage 4: prune old signals (framework).
        let pruned = self.signal_log.prune(self.config.retention_days)?;

        let result = CompressionStats {
            signals_processed: stats.total_count(),
            patterns_extracted: patterns.len() as u64,
            configs_updated,
            signals_pruned: pruned,
            time_ms: start.elapsed().as_millis() as u64,
            completed_at: Utc::now(),
        };

        // Persist checkpoint.
        self.checkpoint.save(&Checkpoint {
            last_run: result.completed_at,
            last_domain: domain.map(str::to_string),
            last_stats: Some(result.clone()),
        })?;

        Ok(result)
    }

    /// Read signals from the log, optionally filtered by domain key.
    fn read_signals(&self, domain: Option<&str>, days: u32) -> anyhow::Result<Vec<D::Signal>> {
        let to = Utc::now().date_naive();
        let from = to - Duration::days(days as i64);
        let signals = self.signal_log.read_range(from, to)?;
        match domain {
            Some(d) => Ok(signals
                .into_iter()
                .filter(|s| D::signal_domain(s) == Some(d))
                .collect()),
            None => Ok(signals),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;
    use thiserror::Error;

    // --- Toy learning domain (proves the loop is reusable without recipes) ---

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ToySignal {
        domain: Option<String>,
        success: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    enum ToyPattern {
        Good,
        Bad,
    }

    #[derive(Default)]
    struct ToyStats {
        count: u64,
        successes: u64,
    }
    impl DomainStats for ToyStats {
        fn total_count(&self) -> u64 {
            self.count
        }
    }

    #[derive(Debug, Error)]
    #[error("toy failure")]
    struct ToyError;

    struct ToyLearningDomain {
        threshold: u64,
        knowledge_updates: std::sync::Mutex<u64>,
    }

    impl LearningDomain for ToyLearningDomain {
        type Signal = ToySignal;
        type Pattern = ToyPattern;
        type Stats = ToyStats;
        type Error = ToyError;

        fn signal_domain(s: &ToySignal) -> Option<&str> {
            s.domain.as_deref()
        }

        fn aggregate(&self, signals: &[ToySignal]) -> ToyStats {
            ToyStats {
                count: signals.len() as u64,
                successes: signals.iter().filter(|s| s.success).count() as u64,
            }
        }

        fn extract_patterns(&self, stats: &ToyStats) -> Vec<ToyPattern> {
            if stats.successes >= self.threshold {
                vec![ToyPattern::Good]
            } else if stats.count > 0 && stats.successes == 0 {
                vec![ToyPattern::Bad]
            } else {
                vec![]
            }
        }

        fn update_knowledge(&self, patterns: &[ToyPattern]) -> Result<u64, ToyError> {
            let mut updates = self.knowledge_updates.lock().unwrap();
            *updates += patterns.len() as u64;
            Ok(*updates)
        }
    }

    // --- Helpers ---

    fn make_runner(
        dir: &std::path::Path,
        threshold: u64,
        config: CompressionConfig,
    ) -> CompressionRunner<ToyLearningDomain> {
        let domain = Arc::new(ToyLearningDomain {
            threshold,
            knowledge_updates: std::sync::Mutex::new(0),
        });
        let signal_log = Arc::new(DailyLog::<ToySignal>::open(dir.join("signals")).unwrap());
        let checkpoint = CheckpointStore::open(dir.join("checkpoint.json")).unwrap();
        CompressionRunner::new(domain, signal_log, config, checkpoint)
    }

    fn default_runner(dir: &std::path::Path) -> CompressionRunner<ToyLearningDomain> {
        make_runner(
            dir,
            3,
            CompressionConfig {
                min_sample_size: 1,
                ..Default::default()
            },
        )
    }

    fn append_signal(runner: &CompressionRunner<ToyLearningDomain>, domain: &str, success: bool) {
        runner
            .signal_log
            .append(&ToySignal {
                domain: Some(domain.to_string()),
                success,
            })
            .unwrap();
    }

    // --- Loop tests ---

    #[test]
    fn run_end_to_end_with_toy_domain() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        for _ in 0..5 {
            append_signal(&runner, "toy.com", true);
        }

        let stats = runner.run(Some("toy.com"), 1).unwrap();
        assert_eq!(stats.signals_processed, 5);
        assert_eq!(stats.patterns_extracted, 1);
        assert_eq!(stats.configs_updated, 1);
        assert_eq!(stats.signals_pruned, 0);
    }

    #[test]
    fn run_with_no_signals_returns_zero_stats() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        let stats = runner.run(Some("toy.com"), 1).unwrap();
        assert_eq!(stats.signals_processed, 0);
        assert_eq!(stats.patterns_extracted, 0);
        assert_eq!(stats.configs_updated, 0);
    }

    #[test]
    fn run_filters_by_domain() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        for _ in 0..4 {
            append_signal(&runner, "a.com", true);
        }
        for _ in 0..2 {
            append_signal(&runner, "b.com", true);
        }

        let stats = runner.run(Some("a.com"), 1).unwrap();
        assert_eq!(stats.signals_processed, 4);
    }

    #[test]
    fn run_with_none_domain_processes_all() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        for _ in 0..3 {
            append_signal(&runner, "a.com", true);
        }
        for _ in 0..2 {
            append_signal(&runner, "b.com", true);
        }

        let stats = runner.run(None, 1).unwrap();
        assert_eq!(stats.signals_processed, 5);
    }

    #[test]
    fn run_persists_checkpoint() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        append_signal(&runner, "toy.com", true);
        runner.run(Some("toy.com"), 1).unwrap();

        let cp = runner.checkpoint.load().unwrap();
        assert_eq!(cp.last_domain.as_deref(), Some("toy.com"));
        assert!(cp.last_stats.is_some());
        assert_eq!(cp.last_stats.as_ref().unwrap().signals_processed, 1);
    }

    #[test]
    fn run_prunes_old_signals_when_past_retention() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        // Create an old signal file (11 days ago).
        let old_date = Utc::now().date_naive() - Duration::days(11);
        let old_path = dir
            .path()
            .join("signals")
            .join(format!("{}.jsonl", old_date.format("%Y-%m-%d")));
        std::fs::write(
            &old_path,
            serde_json::to_string(&ToySignal {
                domain: Some("old.com".to_string()),
                success: true,
            })
            .unwrap(),
        )
        .unwrap();

        // retention_days = 30 (default) -> 11-day-old file is NOT pruned.
        let stats = runner.run(Some("toy.com"), 1).unwrap();
        assert_eq!(stats.signals_pruned, 0);
        assert!(old_path.exists());

        // Now use a config that prunes anything older than 5 days.
        let runner2 = make_runner(
            dir.path(),
            3,
            CompressionConfig {
                min_sample_size: 1,
                retention_days: 5,
                ..Default::default()
            },
        );
        let stats2 = runner2.run(Some("toy.com"), 1).unwrap();
        assert_eq!(stats2.signals_pruned, 1);
        assert!(!old_path.exists());
    }

    #[test]
    fn run_propagates_domain_error() {
        use crate::framework::learning::domain::LearningDomain;

        struct FailingDomain;
        #[derive(Debug, Error)]
        #[error("boom")]
        struct Boom;
        impl LearningDomain for FailingDomain {
            type Signal = ToySignal;
            type Pattern = ToyPattern;
            type Stats = ToyStats;
            type Error = Boom;
            fn signal_domain(s: &ToySignal) -> Option<&str> {
                s.domain.as_deref()
            }
            fn aggregate(&self, signals: &[ToySignal]) -> ToyStats {
                ToyStats {
                    count: signals.len() as u64,
                    successes: 0,
                }
            }
            fn extract_patterns(&self, _: &ToyStats) -> Vec<ToyPattern> {
                vec![]
            }
            fn update_knowledge(&self, _: &[ToyPattern]) -> Result<u64, Boom> {
                Err(Boom)
            }
        }

        let dir = tempdir().unwrap();
        let domain = Arc::new(FailingDomain);
        let signal_log = Arc::new(DailyLog::<ToySignal>::open(dir.path().join("signals")).unwrap());
        let checkpoint = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();
        let runner = CompressionRunner::new(
            domain,
            signal_log.clone(),
            CompressionConfig {
                min_sample_size: 1,
                ..Default::default()
            },
            checkpoint,
        );
        signal_log
            .append(&ToySignal {
                domain: Some("x.com".into()),
                success: true,
            })
            .unwrap();

        let err = runner.run(Some("x.com"), 1).unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    // --- Scheduler tests (should_run lives in scheduler.rs) ---

    #[test]
    fn should_run_false_when_no_signals_and_recent_checkpoint() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        // Fresh checkpoint defaults to now -> time condition false; no signals.
        assert!(!runner.should_run("toy.com").unwrap());
    }

    #[test]
    fn should_run_true_when_signal_threshold_met() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        for _ in 0..101 {
            append_signal(&runner, "toy.com", true);
        }

        assert!(runner.should_run("toy.com").unwrap());
    }

    #[test]
    fn should_run_false_for_domain_with_no_signals() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        for _ in 0..101 {
            append_signal(&runner, "toy.com", true);
        }

        // Different domain -> below threshold, and checkpoint is recent.
        assert!(!runner.should_run("other.com").unwrap());
    }

    #[test]
    fn should_run_true_when_schedule_interval_exceeded() {
        let dir = tempdir().unwrap();
        let runner = default_runner(dir.path());

        runner
            .checkpoint
            .update_last_run(Utc::now() - Duration::days(8))
            .unwrap();

        assert!(runner.should_run("toy.com").unwrap());
    }
}

//! Recipe learning domain: implements the framework's [`LearningDomain`].
//!
//! Owns signal aggregation, pattern extraction (including `infer_action`),
//! and knowledge updates over a [`RecipeKnowledgeStore`]. The three domain
//! stages live in [`super::aggregate`], [`super::extract`], and
//! [`super::update`]; this module wires them to the trait.

use std::sync::Arc;

use crate::domain::recipe::learning::aggregate::{RecipeAggregatedStats, aggregate_recipe_signals};
use crate::domain::recipe::learning::extract::RecipePattern;
use crate::domain::recipe::models::signal::RecipeSignal;
use crate::domain::recipe::storage::RecipeKnowledgeStore;
use crate::framework::learning::domain::LearningDomain;
use crate::framework::learning::stats::CompressionConfig;
use crate::framework::storage::StorageError;

/// Recipe-domain implementation of [`LearningDomain`].
pub struct RecipeLearningDomain {
    pub(super) knowledge: Arc<RecipeKnowledgeStore>,
    pub(super) config: CompressionConfig,
}

impl RecipeLearningDomain {
    /// Create a new recipe learning domain.
    pub fn new(knowledge: Arc<RecipeKnowledgeStore>, config: CompressionConfig) -> Self {
        Self { knowledge, config }
    }

    /// Borrow the underlying knowledge store (used by the binary and tests).
    pub fn knowledge(&self) -> &RecipeKnowledgeStore {
        &self.knowledge
    }
}

impl LearningDomain for RecipeLearningDomain {
    type Signal = RecipeSignal;
    type Pattern = RecipePattern;
    type Stats = RecipeAggregatedStats;
    type Error = StorageError;

    fn signal_domain(signal: &RecipeSignal) -> Option<&str> {
        signal.domain.as_deref()
    }

    fn aggregate(&self, signals: &[RecipeSignal]) -> RecipeAggregatedStats {
        aggregate_recipe_signals(signals)
    }

    fn extract_patterns(&self, stats: &RecipeAggregatedStats) -> Vec<RecipePattern> {
        self.extract_recipe_patterns(stats)
    }

    fn update_knowledge(&self, patterns: &[RecipePattern]) -> Result<u64, StorageError> {
        self.update_recipe_knowledge(patterns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::learning::Probability;
    use crate::domain::recipe::models::recipe::ParseMethod;
    use crate::domain::recipe::models::signal::{RecipeSignal, RecipeSignalType};
    use crate::domain::recipe::storage::{KnowledgeStorage, RecipeKnowledgeStore};
    use crate::framework::learning::runner::CompressionRunner;
    use crate::framework::storage::{CheckpointStore, DailyLog};
    use tempfile::tempdir;

    #[test]
    fn recipe_domain_implements_learning_loop() {
        let dir = tempdir().unwrap();
        let knowledge = Arc::new(RecipeKnowledgeStore::open(dir.path().join("knowledge")).unwrap());
        let config = CompressionConfig {
            min_sample_size: 5,
            confidence_threshold: Probability::new(0.5).unwrap(),
            min_success_rate: Probability::new(0.7).unwrap(),
            ..Default::default()
        };
        let domain = Arc::new(RecipeLearningDomain::new(knowledge.clone(), config.clone()));
        let signal_log =
            Arc::new(DailyLog::<RecipeSignal>::open(dir.path().join("signals")).unwrap());
        let checkpoint = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();
        let runner = CompressionRunner::new(domain, signal_log.clone(), config, checkpoint);

        // Append enough successful signals for "example.com".
        // 30 samples -> sample_confidence = 1.0, wilson(30,30) ~ 0.89 -> confidence ~ 0.89 >= 0.5
        for _ in 0..30 {
            signal_log
                .append(
                    &RecipeSignal::new(RecipeSignalType::ParseSuccess {
                        method: ParseMethod::SchemaOrg,
                        time_ms: 100,
                    })
                    .with_domain("example.com"),
                )
                .unwrap();
        }

        let stats = runner.run(Some("example.com"), 1).unwrap();
        assert_eq!(stats.signals_processed, 30);
        assert_eq!(stats.patterns_extracted, 1);
        assert_eq!(stats.configs_updated, 1);

        // Knowledge store should now have a site config for example.com.
        let config = knowledge.get_site_config("example.com").unwrap().unwrap();
        assert_eq!(config.preferred_method, ParseMethod::SchemaOrg);

        // Patterns should be stored.
        let patterns = knowledge.get_patterns().unwrap();
        assert_eq!(patterns.success_patterns.len(), 1);
        assert!(patterns.version >= 1);
    }

    #[test]
    fn recipe_domain_handles_mixed_signals() {
        let dir = tempdir().unwrap();
        let knowledge = Arc::new(RecipeKnowledgeStore::open(dir.path().join("knowledge")).unwrap());
        let config = CompressionConfig {
            min_sample_size: 5,
            confidence_threshold: Probability::new(0.1).unwrap(),
            min_success_rate: Probability::new(0.5).unwrap(),
            ..Default::default()
        };
        let domain = Arc::new(RecipeLearningDomain::new(knowledge.clone(), config.clone()));
        let signal_log =
            Arc::new(DailyLog::<RecipeSignal>::open(dir.path().join("signals")).unwrap());
        let checkpoint = CheckpointStore::open(dir.path().join("checkpoint.json")).unwrap();
        let runner = CompressionRunner::new(domain, signal_log.clone(), config, checkpoint);

        // 6 successes, 8 failures (failure_rate 0.57 > 0.5) with a frequent error.
        for _ in 0..6 {
            signal_log
                .append(
                    &RecipeSignal::new(RecipeSignalType::ParseSuccess {
                        method: ParseMethod::SchemaOrg,
                        time_ms: 100,
                    })
                    .with_domain("mixed.com"),
                )
                .unwrap();
        }
        for _ in 0..8 {
            signal_log
                .append(
                    &RecipeSignal::new(RecipeSignalType::ParseFailure {
                        method: ParseMethod::SchemaOrg,
                        error: "timeout".to_string(),
                        attempted_methods: vec![],
                    })
                    .with_domain("mixed.com"),
                )
                .unwrap();
        }

        let stats = runner.run(Some("mixed.com"), 1).unwrap();
        assert_eq!(stats.signals_processed, 14);
        // success_rate 6/14 = 0.43 < 0.5 -> no success pattern.
        // failure_rate 0.57 > 0.5, timeout count 8 > 5 -> 1 anti-pattern.
        assert_eq!(stats.patterns_extracted, 1);

        let patterns = knowledge.get_patterns().unwrap();
        assert_eq!(patterns.anti_patterns.len(), 1);
        assert_eq!(patterns.anti_patterns[0].description, "timeout");
    }
}

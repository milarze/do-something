//! Pattern extraction from aggregated recipe stats.
//!
//! Produces [`RecipePattern`]s: success patterns (reliable methods) and
//! anti-patterns (frequent failures with corrective actions via
//! [`RecipeLearningDomain::infer_action`]). Confidence is derived from the
//! framework's [`wilson_score`] and [`sample_confidence`] utilities.

use crate::domain::recipe::models::patterns::{AntiPattern, AntiPatternAction, SuccessPattern};
use crate::framework::learning::confidence::{sample_confidence, wilson_score};
use crate::framework::learning::domain::DomainStats;

use super::aggregate::RecipeAggregatedStats;
use super::domain::RecipeLearningDomain;

/// A pattern extracted from recipe signal analysis.
///
/// Either a success pattern (what works) or an anti-pattern (what to avoid).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RecipePattern {
    /// A method that reliably succeeds.
    Success(SuccessPattern),
    /// A failure mode with a corrective action.
    Anti(AntiPattern),
}

impl RecipeLearningDomain {
    /// Stage 2: extract patterns from aggregated stats.
    ///
    /// Returns success patterns for methods whose success rate clears
    /// `min_success_rate`, and anti-patterns for errors that are both
    /// frequent (`> 5` occurrences) and occur in a high-failure context
    /// (`failure_rate > 0.5`).
    pub(crate) fn extract_recipe_patterns(
        &self,
        stats: &RecipeAggregatedStats,
    ) -> Vec<RecipePattern> {
        let mut patterns = Vec::new();

        // Skip if not enough data.
        if stats.total_count() < self.config.min_sample_size {
            return patterns;
        }

        // Success pattern: best method above the success-rate threshold.
        if let Some((method, success_rate)) = stats.best_method()
            && success_rate >= self.config.min_success_rate.get()
        {
            let confidence = self.confidence_of(stats.total_successes, stats.total_count());
            patterns.push(RecipePattern::Success(SuccessPattern {
                description: format!("{:?} parsing works well for {}", method, stats.domain),
                sites: vec![stats.domain.clone()],
                method,
                success_rate,
                sample_size: stats.total_count(),
                confidence,
            }));
        }

        // Anti-patterns: frequent errors with high failure rates.
        if stats.failure_rate() > 0.5 {
            for err_freq in &stats.common_errors {
                if err_freq.count > 5 {
                    let confidence = self.confidence_of(err_freq.count, stats.total_count());
                    patterns.push(RecipePattern::Anti(AntiPattern {
                        description: err_freq.error.clone(),
                        sites: vec![stats.domain.clone()],
                        failure_rate: stats.failure_rate(),
                        action: self.infer_action(&err_freq.error),
                        sample_size: err_freq.count,
                        confidence,
                    }));
                }
            }
        }

        patterns
    }

    /// Overall confidence in a proportion: sample-size factor × Wilson lower bound.
    fn confidence_of(&self, positive: u64, total: u64) -> f64 {
        if total == 0 {
            return 0.0;
        }
        sample_confidence(total) * wilson_score(positive, total)
    }

    /// Map a parse error to a corrective action. Recipe logic; lives in the domain.
    ///
    /// Matching is case-insensitive for robustness.
    fn infer_action(&self, error: &str) -> AntiPatternAction {
        let error = error.to_lowercase();
        if error.contains("timeout") {
            AntiPatternAction::SlowDown
        } else if error.contains("no_ingredients") || error.contains("no ingredients") {
            AntiPatternAction::TryAlternativeMethod
        } else if error.contains("/video") {
            AntiPatternAction::SkipUrl
        } else if error.contains("javascript") {
            AntiPatternAction::UseHeadlessBrowser
        } else {
            AntiPatternAction::LogWarning
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::learning::Probability;
    use crate::domain::recipe::learning::aggregate::{
        ErrorFrequency, MethodStats, RecipeAggregatedStats,
    };
    use crate::domain::recipe::models::recipe::ParseMethod;
    use crate::domain::recipe::storage::RecipeKnowledgeStore;
    use crate::framework::learning::stats::CompressionConfig;
    use chrono::Utc;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_domain(config: CompressionConfig) -> (RecipeLearningDomain, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let knowledge = Arc::new(RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap());
        let domain = RecipeLearningDomain::new(knowledge, config);
        (domain, dir)
    }

    /// Build stats with a single method and the given totals + errors.
    fn make_stats(
        domain: &str,
        successes: u64,
        failures: u64,
        errors: Vec<(&str, u64)>,
    ) -> RecipeAggregatedStats {
        let total_errors: u64 = errors.iter().map(|(_, c)| c).sum();
        let common_errors: Vec<ErrorFrequency> = errors
            .into_iter()
            .map(|(e, c)| ErrorFrequency {
                error: e.to_string(),
                count: c,
                percentage: if total_errors == 0 {
                    0.0
                } else {
                    c as f64 / total_errors as f64 * 100.0
                },
            })
            .collect();

        let mut by_method = HashMap::new();
        by_method.insert(
            ParseMethod::SchemaOrg,
            MethodStats {
                successes,
                failures,
                avg_time_ms: 100.0,
                errors: common_errors
                    .iter()
                    .map(|e| (e.error.clone(), e.count))
                    .collect(),
            },
        );

        RecipeAggregatedStats {
            domain: domain.to_string(),
            by_method,
            total_successes: successes,
            total_failures: failures,
            common_errors,
            time_period: (Utc::now(), Utc::now()),
        }
    }

    fn default_config() -> CompressionConfig {
        CompressionConfig {
            min_sample_size: 10,
            confidence_threshold: Probability::new(0.5).unwrap(),
            min_success_rate: Probability::new(0.8).unwrap(),
            ..Default::default()
        }
    }

    #[test]
    fn extract_below_min_sample_returns_empty() {
        let (domain, _dir) = make_domain(default_config());
        let stats = make_stats("a.com", 5, 0, vec![]); // total 5 < 10
        let patterns = domain.extract_recipe_patterns(&stats);
        assert!(patterns.is_empty());
    }

    #[test]
    fn extract_success_pattern_when_high_success_rate() {
        let (domain, _dir) = make_domain(default_config());
        let stats = make_stats("a.com", 12, 0, vec![]); // rate 1.0 >= 0.8
        let patterns = domain.extract_recipe_patterns(&stats);

        assert_eq!(patterns.len(), 1);
        match &patterns[0] {
            RecipePattern::Success(sp) => {
                assert_eq!(sp.method, ParseMethod::SchemaOrg);
                assert!((sp.success_rate - 1.0).abs() < 1e-9);
                assert_eq!(sp.sample_size, 12);
                assert!(sp.confidence > 0.0);
                assert!(sp.sites.contains(&"a.com".to_string()));
            }
            _ => panic!("expected success pattern"),
        }
    }

    #[test]
    fn extract_no_success_pattern_when_low_success_rate() {
        let (domain, _dir) = make_domain(default_config());
        // 8 successes, 4 failures -> rate 0.67 < 0.8; failure_rate 0.33 <= 0.5
        let stats = make_stats("a.com", 8, 4, vec![]);
        let patterns = domain.extract_recipe_patterns(&stats);
        assert!(patterns.is_empty());
    }

    #[test]
    fn extract_anti_pattern_for_frequent_errors() {
        let (domain, _dir) = make_domain(default_config());
        // 3 successes, 9 failures -> failure_rate 0.75 > 0.5; error count 7 > 5
        let stats = make_stats("a.com", 3, 9, vec![("timeout", 7)]);
        let patterns = domain.extract_recipe_patterns(&stats);

        let anti = patterns.iter().find_map(|p| match p {
            RecipePattern::Anti(ap) => Some(ap),
            _ => None,
        });
        let anti = anti.expect("expected an anti-pattern");
        assert_eq!(anti.description, "timeout");
        assert_eq!(anti.action, AntiPatternAction::SlowDown);
        assert_eq!(anti.sample_size, 7);
        assert!(anti.confidence > 0.0);
    }

    #[test]
    fn infer_action_mappings_via_anti_patterns() {
        let (domain, _dir) = make_domain(default_config());
        let cases = [
            ("timeout", AntiPatternAction::SlowDown),
            (
                "no_ingredients_found",
                AntiPatternAction::TryAlternativeMethod,
            ),
            ("url contains /video", AntiPatternAction::SkipUrl),
            ("needs javascript", AntiPatternAction::UseHeadlessBrowser),
            ("unknown error", AntiPatternAction::LogWarning),
        ];

        for (error, expected_action) in cases {
            let stats = make_stats("a.com", 3, 9, vec![(error, 7)]);
            let patterns = domain.extract_recipe_patterns(&stats);
            let anti = patterns
                .iter()
                .find_map(|p| match p {
                    RecipePattern::Anti(ap) => Some(ap),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("expected anti-pattern for '{error}'"));
            assert_eq!(anti.action, expected_action, "action for '{error}'");
        }
    }

    #[test]
    fn infer_action_is_case_insensitive() {
        let (domain, _dir) = make_domain(default_config());
        let stats = make_stats("a.com", 3, 9, vec![("TIMEOUT", 7)]);
        let patterns = domain.extract_recipe_patterns(&stats);
        let anti = patterns
            .iter()
            .find_map(|p| match p {
                RecipePattern::Anti(ap) => Some(ap),
                _ => None,
            })
            .expect("expected anti-pattern");
        assert_eq!(anti.action, AntiPatternAction::SlowDown);
    }

    #[test]
    fn extract_skips_anti_pattern_when_failure_rate_low() {
        let (domain, _dir) = make_domain(default_config());
        // failure_rate 0.25 <= 0.5 even with frequent error
        let stats = make_stats("a.com", 9, 3, vec![("timeout", 7)]);
        let patterns = domain.extract_recipe_patterns(&stats);
        assert!(!patterns.iter().any(|p| matches!(p, RecipePattern::Anti(_))));
    }

    #[test]
    fn extract_skips_anti_pattern_when_error_count_low() {
        let (domain, _dir) = make_domain(default_config());
        // failure_rate 0.75 > 0.5 but error count 3 <= 5
        let stats = make_stats("a.com", 3, 9, vec![("timeout", 3)]);
        let patterns = domain.extract_recipe_patterns(&stats);
        assert!(!patterns.iter().any(|p| matches!(p, RecipePattern::Anti(_))));
    }

    #[test]
    fn extract_emits_both_success_and_anti_when_applicable() {
        // Best method succeeds (SchemaOrg 12/12) and another method fails a lot.
        let (domain, _dir) = make_domain(default_config());
        let mut stats = make_stats("a.com", 12, 8, vec![("timeout", 7)]);
        // Add a failing method so failure_rate reflects failures, but keep
        // SchemaOrg as the clear best. totals: 12 success, 8 failure.
        stats.total_failures = 8;
        // failure_rate = 8/20 = 0.4 <= 0.5 -> no anti. Adjust to force anti.
        stats.total_successes = 4;
        stats.total_failures = 16;
        // Rebuild best method stats: SchemaOrg still perfect with 4 successes.
        stats
            .by_method
            .get_mut(&ParseMethod::SchemaOrg)
            .unwrap()
            .successes = 4;
        stats
            .by_method
            .get_mut(&ParseMethod::SchemaOrg)
            .unwrap()
            .failures = 0;
        // failure_rate = 16/20 = 0.8 > 0.5, error count 7 > 5 -> anti.
        let patterns = domain.extract_recipe_patterns(&stats);
        assert!(
            patterns
                .iter()
                .any(|p| matches!(p, RecipePattern::Success(_)))
        );
        assert!(patterns.iter().any(|p| matches!(p, RecipePattern::Anti(_))));
    }
}

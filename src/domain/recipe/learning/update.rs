//! Knowledge store updates from extracted patterns.
//!
//! Updates site configs for high-confidence patterns and merges new
//! patterns into the stored [`RecipePatterns`] without duplicates.

use chrono::Utc;

use crate::domain::recipe::models::patterns::AntiPatternAction;
use crate::domain::recipe::models::site_config::SiteConfig;
use crate::domain::recipe::storage::KnowledgeStorage;
use crate::framework::storage::StorageError;

use super::domain::RecipeLearningDomain;
use super::extract::RecipePattern;

impl RecipeLearningDomain {
    /// Stage 3: update knowledge from extracted patterns.
    ///
    /// For each high-confidence pattern:
    /// - **Success**: sets the site's `preferred_method` and records the
    ///   success count.
    /// - **Anti** (SkipUrl): adds the description to the site's
    ///   `skip_patterns` so future scrapes avoid it.
    ///
    /// All patterns are merged into the stored [`RecipePatterns`] without
    /// duplicates (matched by description). Returns the number of site
    /// configs updated.
    pub(crate) fn update_recipe_knowledge(
        &self,
        patterns: &[RecipePattern],
    ) -> Result<u64, StorageError> {
        let mut configs_updated = 0u64;
        let mut existing = self.knowledge.get_patterns()?;

        for pattern in patterns {
            match pattern {
                RecipePattern::Success(sp) => {
                    for site in &sp.sites {
                        if sp.confidence >= self.config.confidence_threshold.get() {
                            let mut config = self
                                .knowledge
                                .get_site_config(site)?
                                .unwrap_or_else(|| SiteConfig::new(site));
                            config.preferred_method = sp.method;
                            config.stats.success_count =
                                (sp.success_rate * sp.sample_size as f64) as u64;
                            config.version += 1;
                            config.updated_at = Utc::now();
                            self.knowledge.save_site_config(&config)?;
                            configs_updated += 1;
                        }
                    }
                    if !existing
                        .success_patterns
                        .iter()
                        .any(|p| p.description == sp.description)
                    {
                        existing.success_patterns.push(sp.clone());
                    }
                }
                RecipePattern::Anti(ap) => {
                    for site in &ap.sites {
                        if ap.confidence >= self.config.confidence_threshold.get() {
                            let mut config = self
                                .knowledge
                                .get_site_config(site)?
                                .unwrap_or_else(|| SiteConfig::new(site));
                            if ap.action == AntiPatternAction::SkipUrl
                                && !config.skip_patterns.contains(&ap.description)
                            {
                                config.skip_patterns.push(ap.description.clone());
                            }
                            config.stats.failure_count =
                                config.stats.failure_count.max(ap.sample_size);
                            config.version += 1;
                            config.updated_at = Utc::now();
                            self.knowledge.save_site_config(&config)?;
                            configs_updated += 1;
                        }
                    }
                    if !existing
                        .anti_patterns
                        .iter()
                        .any(|p| p.description == ap.description)
                    {
                        existing.anti_patterns.push(ap.clone());
                    }
                }
            }
        }

        existing.version += 1;
        existing.computed_at = Utc::now();
        self.knowledge.save_patterns(&existing)?;

        Ok(configs_updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::learning::Probability;
    use crate::domain::recipe::models::patterns::{AntiPattern, SuccessPattern};
    use crate::domain::recipe::models::recipe::ParseMethod;
    use crate::domain::recipe::storage::RecipeKnowledgeStore;
    use crate::framework::learning::stats::CompressionConfig;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_domain(conf: f64) -> (RecipeLearningDomain, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let knowledge = Arc::new(RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap());
        let domain = RecipeLearningDomain::new(
            knowledge,
            CompressionConfig {
                confidence_threshold: Probability::new(conf).unwrap(),
                ..Default::default()
            },
        );
        (domain, dir)
    }

    fn success_pattern(site: &str, method: ParseMethod, confidence: f64) -> RecipePattern {
        RecipePattern::Success(SuccessPattern {
            description: format!("{:?} works for {}", method, site),
            sites: vec![site.to_string()],
            method,
            success_rate: 0.9,
            sample_size: 20,
            confidence,
        })
    }

    fn anti_pattern(
        site: &str,
        error: &str,
        action: AntiPatternAction,
        confidence: f64,
    ) -> RecipePattern {
        RecipePattern::Anti(AntiPattern {
            description: error.to_string(),
            sites: vec![site.to_string()],
            failure_rate: 0.8,
            action,
            sample_size: 10,
            confidence,
        })
    }

    #[test]
    fn updates_site_config_preferred_method() {
        let (domain, _dir) = make_domain(0.5);
        let patterns = vec![success_pattern("a.com", ParseMethod::SchemaOrg, 0.8)];

        let updated = domain.update_recipe_knowledge(&patterns).unwrap();
        assert_eq!(updated, 1);

        let config = domain.knowledge.get_site_config("a.com").unwrap().unwrap();
        assert_eq!(config.preferred_method, ParseMethod::SchemaOrg);
        assert_eq!(config.stats.success_count, 18); // 0.9 * 20 = 18
        assert!(config.version >= 1);
    }

    #[test]
    fn creates_config_when_missing() {
        let (domain, _dir) = make_domain(0.5);
        let patterns = vec![success_pattern("new.com", ParseMethod::Selectors, 0.8)];

        domain.update_recipe_knowledge(&patterns).unwrap();

        let config = domain
            .knowledge
            .get_site_config("new.com")
            .unwrap()
            .unwrap();
        assert_eq!(config.domain, "new.com");
        assert_eq!(config.preferred_method, ParseMethod::Selectors);
    }

    #[test]
    fn skips_low_confidence_success_pattern() {
        let (domain, _dir) = make_domain(0.9);
        let patterns = vec![success_pattern("a.com", ParseMethod::SchemaOrg, 0.5)];

        let updated = domain.update_recipe_knowledge(&patterns).unwrap();
        assert_eq!(updated, 0);
        assert!(domain.knowledge.get_site_config("a.com").unwrap().is_none());
    }

    #[test]
    fn merges_success_patterns_without_duplicates() {
        let (domain, _dir) = make_domain(0.9);
        let p = success_pattern("a.com", ParseMethod::SchemaOrg, 0.95);

        domain
            .update_recipe_knowledge(std::slice::from_ref(&p))
            .unwrap();
        domain.update_recipe_knowledge(&[p]).unwrap(); // same description

        let patterns = domain.knowledge.get_patterns().unwrap();
        assert_eq!(patterns.success_patterns.len(), 1);
    }

    #[test]
    fn merges_distinct_success_patterns() {
        let (domain, _dir) = make_domain(0.9);
        let p1 = success_pattern("a.com", ParseMethod::SchemaOrg, 0.95);
        let p2 = RecipePattern::Success(SuccessPattern {
            description: "Selectors works for a.com".to_string(),
            sites: vec!["a.com".to_string()],
            method: ParseMethod::Selectors,
            success_rate: 0.85,
            sample_size: 15,
            confidence: 0.95,
        });

        domain.update_recipe_knowledge(&[p1, p2]).unwrap();

        let patterns = domain.knowledge.get_patterns().unwrap();
        assert_eq!(patterns.success_patterns.len(), 2);
    }

    #[test]
    fn adds_skip_pattern_for_skip_url_action() {
        let (domain, _dir) = make_domain(0.5);
        let patterns = vec![anti_pattern(
            "a.com",
            "url contains /video",
            AntiPatternAction::SkipUrl,
            0.8,
        )];

        domain.update_recipe_knowledge(&patterns).unwrap();

        let config = domain.knowledge.get_site_config("a.com").unwrap().unwrap();
        assert!(
            config
                .skip_patterns
                .contains(&"url contains /video".to_string())
        );
    }

    #[test]
    fn does_not_add_skip_pattern_for_other_actions() {
        let (domain, _dir) = make_domain(0.5);
        let patterns = vec![anti_pattern(
            "a.com",
            "timeout",
            AntiPatternAction::SlowDown,
            0.8,
        )];

        domain.update_recipe_knowledge(&patterns).unwrap();

        let config = domain.knowledge.get_site_config("a.com").unwrap().unwrap();
        assert!(config.skip_patterns.is_empty());
        // But failure_count should be updated.
        assert_eq!(config.stats.failure_count, 10);
    }

    #[test]
    fn does_not_duplicate_skip_pattern() {
        let (domain, _dir) = make_domain(0.5);
        let p = anti_pattern("a.com", "/video", AntiPatternAction::SkipUrl, 0.8);

        domain
            .update_recipe_knowledge(std::slice::from_ref(&p))
            .unwrap();
        domain.update_recipe_knowledge(&[p]).unwrap();

        let config = domain.knowledge.get_site_config("a.com").unwrap().unwrap();
        assert_eq!(
            config
                .skip_patterns
                .iter()
                .filter(|s| *s == "/video")
                .count(),
            1
        );
    }

    #[test]
    fn merges_anti_patterns_without_duplicates() {
        let (domain, _dir) = make_domain(0.5);
        let p = anti_pattern("a.com", "timeout", AntiPatternAction::SlowDown, 0.8);

        domain
            .update_recipe_knowledge(std::slice::from_ref(&p))
            .unwrap();
        domain.update_recipe_knowledge(&[p]).unwrap();

        let patterns = domain.knowledge.get_patterns().unwrap();
        assert_eq!(patterns.anti_patterns.len(), 1);
    }

    #[test]
    fn returns_total_configs_updated() {
        let (domain, _dir) = make_domain(0.5);
        let patterns = vec![
            success_pattern("a.com", ParseMethod::SchemaOrg, 0.8),
            success_pattern("b.com", ParseMethod::Selectors, 0.8),
        ];

        let updated = domain.update_recipe_knowledge(&patterns).unwrap();
        assert_eq!(updated, 2);
    }

    #[test]
    fn empty_patterns_updates_nothing_but_increments_version() {
        let (domain, _dir) = make_domain(0.5);
        let updated = domain.update_recipe_knowledge(&[]).unwrap();
        assert_eq!(updated, 0);

        let patterns = domain.knowledge.get_patterns().unwrap();
        assert_eq!(patterns.version, 1);
    }

    #[test]
    fn increments_pattern_version_on_each_update() {
        let (domain, _dir) = make_domain(0.5);

        domain
            .update_recipe_knowledge(&[success_pattern("a.com", ParseMethod::SchemaOrg, 0.8)])
            .unwrap();
        let v1 = domain.knowledge.get_patterns().unwrap().version;

        domain
            .update_recipe_knowledge(&[success_pattern("a.com", ParseMethod::Selectors, 0.8)])
            .unwrap();
        let v2 = domain.knowledge.get_patterns().unwrap().version;

        assert_eq!(v2, v1 + 1);
    }
}

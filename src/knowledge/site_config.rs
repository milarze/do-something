//! Site configuration management with defaults.
//!
//! Provides site-specific parsing configurations with
//! validation. Default configurations for known recipe sites
//! are provided by `crate::domain::recipes::site_defaults`.

use crate::domain::recipes::site_defaults::SiteDefaults;
use crate::models::knowledge::{Selectors, SiteConfig, SiteStats};
use crate::models::ParseMethod;
use crate::storage::StorageError;

/// Manager for site-specific parsing configurations.
pub struct SiteConfigManager;

impl SiteConfigManager {
    /// Create a default configuration for a domain.
    ///
    /// Returns a known default if the domain is recognized,
    /// otherwise returns a generic default configuration.
    pub fn default_for_domain(domain: &str) -> SiteConfig {
        // Check if we have a known default from domain knowledge
        if let Some(config) = SiteDefaults::get(domain) {
            return config;
        }

        // Generic default for unknown sites
        SiteConfig {
            domain: domain.to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors::default(),
            rate_limit_ms: 1000,
            requires_js: false,
            headers: std::collections::HashMap::new(),
            skip_patterns: Vec::new(),
            stats: SiteStats::default(),
            updated_at: chrono::Utc::now(),
            version: 0,
        }
    }

    /// Merge configuration with a fallback.
    ///
    /// Values present in config are always kept (explicitly set).
    /// Missing/empty values (headers, skip_patterns, selectors) use fallback.
    ///
    /// Note: preferred_method and rate_limit_ms are always taken from config
    /// since there's no way to distinguish "not set" from "set to default".
    pub fn merge_with_fallback(config: &SiteConfig, fallback: &SiteConfig) -> SiteConfig {
        SiteConfig {
            domain: config.domain.clone(),
            // Always use config's method - it was explicitly set
            preferred_method: config.preferred_method,
            selectors: merge_selectors(&config.selectors, &fallback.selectors),
            // Always use config's rate limit - it was explicitly set
            rate_limit_ms: config.rate_limit_ms,
            // JS requirement is OR'd - if either needs it, use it
            requires_js: config.requires_js || fallback.requires_js,
            headers: if config.headers.is_empty() {
                fallback.headers.clone()
            } else {
                config.headers.clone()
            },
            skip_patterns: if config.skip_patterns.is_empty() {
                fallback.skip_patterns.clone()
            } else {
                config.skip_patterns.clone()
            },
            stats: config.stats.clone(),
            updated_at: chrono::Utc::now(),
            version: config.version,
        }
    }

    /// Validate a site configuration.
    pub fn validate(config: &SiteConfig) -> Result<(), StorageError> {
        // Validate domain
        if config.domain.is_empty() {
            return Err(StorageError::InvalidPath("domain cannot be empty".into()));
        }

        // Domain should not contain path separators
        if config.domain.contains('/') || config.domain.contains('\\') {
            return Err(StorageError::InvalidPath(
                format!("invalid domain '{}': contains path separator", config.domain)
            ));
        }

        // Validate rate limit
        if config.rate_limit_ms == 0 {
            return Err(StorageError::InvalidPath(
                "rate_limit_ms must be greater than 0".into()
            ));
        }

        // Validate selectors (if present)
        validate_selectors(&config.selectors)?;

        Ok(())
    }

    /// Get selector for a field, falling back to default.
    pub fn get_selector<'a>(config: &'a SiteConfig, field: &str) -> Option<&'a str> {
        match field {
            "title" => config.selectors.title.as_deref(),
            "ingredients" => config.selectors.ingredients.as_deref(),
            "instructions" => config.selectors.instructions.as_deref(),
            "prep_time" => config.selectors.prep_time.as_deref(),
            "cook_time" => config.selectors.cook_time.as_deref(),
            "total_time" => config.selectors.total_time.as_deref(),
            "servings" | "yield" => config.selectors.servings.as_deref(),
            "author" => config.selectors.author.as_deref(),
            "image" => config.selectors.image.as_deref(),
            "description" => config.selectors.description.as_deref(),
            _ => None,
        }
    }
}

/// Merge two selectors, preferring self over fallback.
fn merge_selectors(self_sel: &Selectors, fallback: &Selectors) -> Selectors {
    Selectors {
        title: self_sel.title.as_ref().or(fallback.title.as_ref()).cloned(),
        ingredients: self_sel.ingredients.as_ref().or(fallback.ingredients.as_ref()).cloned(),
        instructions: self_sel.instructions.as_ref().or(fallback.instructions.as_ref()).cloned(),
        prep_time: self_sel.prep_time.as_ref().or(fallback.prep_time.as_ref()).cloned(),
        cook_time: self_sel.cook_time.as_ref().or(fallback.cook_time.as_ref()).cloned(),
        total_time: self_sel.total_time.as_ref().or(fallback.total_time.as_ref()).cloned(),
        servings: self_sel.servings.as_ref().or(fallback.servings.as_ref()).cloned(),
        author: self_sel.author.as_ref().or(fallback.author.as_ref()).cloned(),
        image: self_sel.image.as_ref().or(fallback.image.as_ref()).cloned(),
        description: self_sel.description.as_ref().or(fallback.description.as_ref()).cloned(),
    }
}

/// Validate CSS selectors.
fn validate_selectors(selectors: &Selectors) -> Result<(), StorageError> {
    // Basic validation: selectors should not be empty strings if present
    for (name, value) in [
        ("title", &selectors.title),
        ("ingredients", &selectors.ingredients),
        ("instructions", &selectors.instructions),
    ] {
        if let Some(sel) = value
            && sel.trim().is_empty()
        {
            return Err(StorageError::InvalidPath(
                format!("selector '{}' cannot be empty string", name)
            ));
        }
    }

    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_for_known_domain() {
        let config = SiteConfigManager::default_for_domain("allrecipes.com");
        assert_eq!(config.domain, "allrecipes.com");
        assert!(config.selectors.title.is_some());
    }

    #[test]
    fn default_for_unknown_domain() {
        let config = SiteConfigManager::default_for_domain("unknown-site.org");
        assert_eq!(config.domain, "unknown-site.org");
        assert_eq!(config.preferred_method, ParseMethod::SchemaOrg);
    }

    #[test]
    fn merge_keeps_self_values() {
        let mut self_config = SiteConfig::new("test.com");
        self_config.rate_limit_ms = 3000;

        let fallback = SiteConfig::new("fallback.com");

        let merged = SiteConfigManager::merge_with_fallback(&self_config, &fallback);
        
        assert_eq!(merged.rate_limit_ms, 3000); // Kept self value
        assert_eq!(merged.domain, "test.com");   // Kept self domain
    }

    #[test]
    fn merge_uses_fallback_for_empty() {
        let self_config = SiteConfig::new("test.com");
        
        let mut fallback = SiteConfig::new("fallback.com");
        fallback.requires_js = true;
        fallback.skip_patterns = vec!["/video/".to_string()];

        let merged = SiteConfigManager::merge_with_fallback(&self_config, &fallback);
        
        // Rate limit is always from config (explicitly set)
        assert_eq!(merged.rate_limit_ms, 1000); // Config's default
        // JS is OR'd
        assert!(merged.requires_js);
        // Empty skip_patterns uses fallback
        assert_eq!(merged.skip_patterns, vec!["/video/"]);
    }

    #[test]
    fn validate_accepts_valid_config() {
        let config = SiteConfig::new("valid.com");
        assert!(SiteConfigManager::validate(&config).is_ok());
    }

    #[test]
    fn validate_rejects_empty_domain() {
        let config = SiteConfig::new("");
        assert!(SiteConfigManager::validate(&config).is_err());
    }

    #[test]
    fn validate_rejects_domain_with_slash() {
        let config = SiteConfig::new("example.com/path");
        assert!(SiteConfigManager::validate(&config).is_err());
    }

    #[test]
    fn validate_rejects_zero_rate_limit() {
        let mut config = SiteConfig::new("test.com");
        config.rate_limit_ms = 0;
        assert!(SiteConfigManager::validate(&config).is_err());
    }

    #[test]
    fn get_selector_returns_correct_field() {
        let mut config = SiteConfig::new("test.com");
        config.selectors.title = Some("h1.title".to_string());
        config.selectors.ingredients = Some("ul.ingredients li".to_string());

        assert_eq!(
            SiteConfigManager::get_selector(&config, "title"),
            Some("h1.title")
        );
        assert_eq!(
            SiteConfigManager::get_selector(&config, "ingredients"),
            Some("ul.ingredients li")
        );
        assert_eq!(
            SiteConfigManager::get_selector(&config, "nonexistent"),
            None
        );
    }

    #[test]
    fn defaults_module_provides_known_sites() {
        let all_defaults = SiteDefaults::all();
        assert!(!all_defaults.is_empty());

        // Should have configs for major recipe sites
        let domains: Vec<&str> = all_defaults.iter().map(|c| c.domain.as_str()).collect();
        assert!(domains.contains(&"allrecipes.com"));
        assert!(domains.contains(&"foodnetwork.com"));
    }

    #[test]
    fn defaults_get_by_domain() {
        let config = SiteDefaults::get("allrecipes.com").unwrap();
        assert_eq!(config.domain, "allrecipes.com");

        let config = SiteDefaults::get("unknown.com");
        assert!(config.is_none());
    }
}

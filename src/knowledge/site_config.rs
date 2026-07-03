//! Site configuration management with defaults.
//!
//! Provides site-specific parsing configurations with
//! validation and default configurations for known recipe sites.

use crate::models::knowledge::{Selectors, SiteConfig, SiteStats};
use crate::models::ParseMethod;
use crate::storage::StorageError;

/// Manager for site-specific parsing configurations.
pub struct SiteConfigManager;

impl SiteConfigManager {
    /// Create a default configuration for a domain.
    pub fn default_for_domain(domain: &str) -> SiteConfig {
        // Check if we have a known default
        if let Some(config) = defaults::get(domain) {
            return config;
        }

        // Generic default
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
    /// Values present in self are kept; missing values use fallback.
    pub fn merge_with_fallback(config: &SiteConfig, fallback: &SiteConfig) -> SiteConfig {
        SiteConfig {
            domain: config.domain.clone(),
            preferred_method: if config.preferred_method == ParseMethod::SchemaOrg {
                fallback.preferred_method
            } else {
                config.preferred_method
            },
            selectors: merge_selectors(&config.selectors, &fallback.selectors),
            rate_limit_ms: if config.rate_limit_ms == 1000 {
                fallback.rate_limit_ms
            } else {
                config.rate_limit_ms
            },
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

/// Default configurations for known recipe sites.
pub mod defaults {
    use crate::models::knowledge::{Selectors, SiteConfig};
    use crate::models::ParseMethod;
    use std::collections::HashMap;

    /// Get default config for a known domain, if available.
    pub fn get(domain: &str) -> Option<SiteConfig> {
        match domain {
            "allrecipes.com" | "www.allrecipes.com" => Some(allrecipes()),
            "foodnetwork.com" | "www.foodnetwork.com" => Some(foodnetwork()),
            "tasty.co" | "www.tasty.co" => Some(tasty()),
            "bettycrocker.com" | "www.bettycrocker.com" => Some(bettycrocker()),
            "pillsbury.com" | "www.pillsbury.com" => Some(pillsbury()),
            "bonappetit.com" | "www.bonappetit.com" => Some(bonappetit()),
            "seriouseats.com" | "www.seriouseats.com" => Some(seriouseats()),
            _ => None,
        }
    }

    /// All known site defaults.
    pub fn all() -> Vec<SiteConfig> {
        vec![
            allrecipes(),
            foodnetwork(),
            tasty(),
            bettycrocker(),
            pillsbury(),
            bonappetit(),
            seriouseats(),
        ]
    }

    fn allrecipes() -> SiteConfig {
        let mut headers = HashMap::new();
        headers.insert("User-Agent".to_string(), "Mozilla/5.0".to_string());

        SiteConfig {
            domain: "allrecipes.com".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.article-heading".to_string()),
                ingredients: Some("ul.mntl-structured-ingredients__list li".to_string()),
                instructions: Some("ol.mntl-sc-block-group--OL li".to_string()),
                prep_time: Some("div.mntl-recipe-block--time".to_string()),
                author: Some("a.mntl-attributed-author__link".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 1500,
            requires_js: false,
            headers,
            skip_patterns: vec![
                "/gallery/".to_string(),
                "/video/".to_string(),
            ],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    fn foodnetwork() -> SiteConfig {
        SiteConfig {
            domain: "foodnetwork.com".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.o-RecipeTitle".to_string()),
                ingredients: Some("div.o-Ingredients__m-Body li".to_string()),
                instructions: Some("div.o-Method__m-Body li".to_string()),
                author: Some("span.o-Attribution__a-Name".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 2000,
            requires_js: false,
            headers: HashMap::new(),
            skip_patterns: vec![
                "/videos/".to_string(),
                "/shows/".to_string(),
            ],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    fn tasty() -> SiteConfig {
        SiteConfig {
            domain: "tasty.co".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.recipe-name".to_string()),
                ingredients: Some("ul.ingredient-list li".to_string()),
                instructions: Some("ol.prep-steps li".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 1500,
            requires_js: true, // Tasty often needs JS for full content
            headers: HashMap::new(),
            skip_patterns: vec![
                "/video/".to_string(),
                "/article/".to_string(),
            ],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    fn bettycrocker() -> SiteConfig {
        SiteConfig {
            domain: "bettycrocker.com".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.recipe-title".to_string()),
                ingredients: Some("div.ingredients ul li".to_string()),
                instructions: Some("div.directions ol li".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 1500,
            requires_js: false,
            headers: HashMap::new(),
            skip_patterns: vec!["/videos/".to_string()],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    fn pillsbury() -> SiteConfig {
        SiteConfig {
            domain: "pillsbury.com".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.recipe-title".to_string()),
                ingredients: Some("div.ingredients-section li".to_string()),
                instructions: Some("div.directions-section li".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 1500,
            requires_js: false,
            headers: HashMap::new(),
            skip_patterns: vec!["/videos/".to_string()],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    fn bonappetit() -> SiteConfig {
        SiteConfig {
            domain: "bonappetit.com".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.Hed".to_string()),
                ingredients: Some("div.ingredients__group li".to_string()),
                instructions: Some("div.directions ol li".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 2500,
            requires_js: true,
            headers: HashMap::new(),
            skip_patterns: vec![
                "/video/".to_string(),
                "/gallery/".to_string(),
            ],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    fn seriouseats() -> SiteConfig {
        SiteConfig {
            domain: "seriouseats.com".to_string(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors {
                title: Some("h1.heading__title".to_string()),
                ingredients: Some("div.recipe-ingredients li".to_string()),
                instructions: Some("div.recipe-instructions ol li".to_string()),
                author: Some("a.author-name".to_string()),
                ..Default::default()
            },
            rate_limit_ms: 2000,
            requires_js: false,
            headers: HashMap::new(),
            skip_patterns: vec!["/videos/".to_string()],
            stats: Default::default(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }
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
    fn merge_uses_fallback_for_missing() {
        let self_config = SiteConfig::new("test.com");
        
        let mut fallback = SiteConfig::new("fallback.com");
        fallback.rate_limit_ms = 5000;
        fallback.requires_js = true;

        let merged = SiteConfigManager::merge_with_fallback(&self_config, &fallback);
        
        assert_eq!(merged.rate_limit_ms, 5000); // Used fallback
        assert!(merged.requires_js);              // Used fallback
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
        let all_defaults = defaults::all();
        assert!(!all_defaults.is_empty());

        // Should have configs for major recipe sites
        let domains: Vec<&str> = all_defaults.iter().map(|c| c.domain.as_str()).collect();
        assert!(domains.contains(&"allrecipes.com"));
        assert!(domains.contains(&"foodnetwork.com"));
    }

    #[test]
    fn defaults_get_by_domain() {
        let config = defaults::get("allrecipes.com").unwrap();
        assert_eq!(config.domain, "allrecipes.com");

        let config = defaults::get("unknown.com");
        assert!(config.is_none());
    }
}

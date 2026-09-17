//! Default configurations for known recipe sites.
//!
//! Contains site-specific knowledge about how to scrape recipes from
//! popular recipe websites. This includes CSS selectors, rate limits,
//! JavaScript requirements, and URL patterns to skip.

use crate::models::knowledge::{Selectors, SiteConfig};
use crate::models::ParseMethod;
use std::collections::HashMap;

/// Default User-Agent for requests to recipe sites.
///
/// Using a realistic browser User-Agent helps avoid basic bot detection.
/// This is a standard Chrome on macOS User-Agent string.
///
/// Note: For production use, consider:
/// - Rotating User-Agents
/// - Using the `user-agent-from-env` feature to allow configuration
/// - Respecting robots.txt and rate limits regardless of User-Agent
const DEFAULT_USER_AGENT: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
     (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/// Provider for site-specific default configurations.
///
/// Knows about popular recipe websites and their scraping requirements.
/// This knowledge is hard-coded (not learned) and represents domain expertise.
pub struct SiteDefaults;

impl SiteDefaults {
    /// Get default config for a known domain, if available.
    ///
    /// Returns `None` for unknown domains, indicating that a generic
    /// default should be used instead.
    pub fn get(domain: &str) -> Option<SiteConfig> {
        match domain {
            "allrecipes.com" | "www.allrecipes.com" => Some(Self::allrecipes()),
            "foodnetwork.com" | "www.foodnetwork.com" => Some(Self::foodnetwork()),
            "tasty.co" | "www.tasty.co" => Some(Self::tasty()),
            "bettycrocker.com" | "www.bettycrocker.com" => Some(Self::bettycrocker()),
            "pillsbury.com" | "www.pillsbury.com" => Some(Self::pillsbury()),
            "bonappetit.com" | "www.bonappetit.com" => Some(Self::bonappetit()),
            "seriouseats.com" | "www.seriouseats.com" => Some(Self::seriouseats()),
            _ => None,
        }
    }

    /// All known site defaults.
    ///
    /// Returns configurations for all recipe sites that have hard-coded
    /// knowledge in this module.
    pub fn all() -> Vec<SiteConfig> {
        vec![
            Self::allrecipes(),
            Self::foodnetwork(),
            Self::tasty(),
            Self::bettycrocker(),
            Self::pillsbury(),
            Self::bonappetit(),
            Self::seriouseats(),
        ]
    }

    fn allrecipes() -> SiteConfig {
        let mut headers = HashMap::new();
        headers.insert("User-Agent".to_string(), DEFAULT_USER_AGENT.to_string());

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
            skip_patterns: vec!["/gallery/".to_string(), "/video/".to_string()],
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
            skip_patterns: vec!["/videos/".to_string(), "/shows/".to_string()],
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
            skip_patterns: vec!["/video/".to_string(), "/article/".to_string()],
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
            skip_patterns: vec!["/video/".to_string(), "/gallery/".to_string()],
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
    fn get_returns_known_site() {
        let config = SiteDefaults::get("allrecipes.com").unwrap();
        assert_eq!(config.domain, "allrecipes.com");
        assert!(config.selectors.title.is_some());
    }

    #[test]
    fn get_returns_www_variant() {
        let config = SiteDefaults::get("www.allrecipes.com").unwrap();
        assert_eq!(config.domain, "allrecipes.com");
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let config = SiteDefaults::get("unknown-site.org");
        assert!(config.is_none());
    }

    #[test]
    fn all_returns_all_known_sites() {
        let all_defaults = SiteDefaults::all();
        assert!(!all_defaults.is_empty());

        let domains: Vec<&str> = all_defaults.iter().map(|c| c.domain.as_str()).collect();
        assert!(domains.contains(&"allrecipes.com"));
        assert!(domains.contains(&"foodnetwork.com"));
        assert!(domains.contains(&"tasty.co"));
        assert!(domains.contains(&"bonappetit.com"));
        assert!(domains.contains(&"seriouseats.com"));
    }

    #[test]
    fn allrecipes_has_correct_selectors() {
        let config = SiteDefaults::get("allrecipes.com").unwrap();
        assert_eq!(config.selectors.title, Some("h1.article-heading".to_string()));
        assert!(config.selectors.ingredients.is_some());
        assert!(config.selectors.instructions.is_some());
    }

    #[test]
    fn tasty_requires_js() {
        let config = SiteDefaults::get("tasty.co").unwrap();
        assert!(config.requires_js);
    }

    #[test]
    fn allrecipes_has_skip_patterns() {
        let config = SiteDefaults::get("allrecipes.com").unwrap();
        assert!(!config.skip_patterns.is_empty());
        assert!(config.skip_patterns.contains(&"/gallery/".to_string()));
    }

    #[test]
    fn rate_limits_are_reasonable() {
        let all_defaults = SiteDefaults::all();
        for config in all_defaults {
            assert!(
                config.rate_limit_ms >= 1000,
                "Rate limit for {} too aggressive: {}ms",
                config.domain,
                config.rate_limit_ms
            );
            assert!(
                config.rate_limit_ms <= 5000,
                "Rate limit for {} too slow: {}ms",
                config.domain,
                config.rate_limit_ms
            );
        }
    }
}

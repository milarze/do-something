//! Site configuration model.
//!
//! Configuration for parsing recipes from a specific site.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::recipe::ParseMethod;

/// Configuration for parsing recipes from a specific site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteConfig {
    /// Domain this config applies to (e.g., "allrecipes.com").
    pub domain: String,

    /// Preferred parsing method.
    #[serde(default)]
    pub preferred_method: ParseMethod,

    /// CSS selectors for each field (fallback when Schema.org unavailable).
    #[serde(default)]
    pub selectors: Selectors,

    /// Rate limiting: minimum milliseconds between requests to this domain.
    #[serde(default = "default_rate_limit")]
    pub rate_limit_ms: u64,

    /// Whether this site requires JavaScript rendering.
    #[serde(default)]
    pub requires_js: bool,

    /// Custom headers to send with requests.
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Known URL patterns to skip (e.g., video pages).
    #[serde(default)]
    pub skip_patterns: Vec<String>,

    /// Statistics about parsing success/failure.
    #[serde(default)]
    pub stats: SiteStats,

    /// When this config was last updated.
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,

    /// Version number for tracking config changes.
    #[serde(default)]
    pub version: u32,
}

fn default_rate_limit() -> u64 {
    1000
}

impl SiteConfig {
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            preferred_method: ParseMethod::SchemaOrg,
            selectors: Selectors::default(),
            rate_limit_ms: default_rate_limit(),
            requires_js: false,
            headers: HashMap::new(),
            skip_patterns: Vec::new(),
            stats: SiteStats::default(),
            updated_at: Utc::now(),
            version: 0,
        }
    }
}

/// CSS selectors for extracting recipe fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Selectors {
    #[serde(default)]
    pub title: Option<String>,

    #[serde(default)]
    pub ingredients: Option<String>,

    #[serde(default)]
    pub instructions: Option<String>,

    #[serde(default)]
    pub prep_time: Option<String>,

    #[serde(default)]
    pub cook_time: Option<String>,

    #[serde(default)]
    pub total_time: Option<String>,

    #[serde(default, rename = "yield")]
    pub servings: Option<String>,

    #[serde(default)]
    pub author: Option<String>,

    #[serde(default)]
    pub image: Option<String>,

    #[serde(default)]
    pub description: Option<String>,
}

/// Statistics for a site's parsing success rate.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SiteStats {
    /// Total successful parses.
    pub success_count: u64,

    /// Total failed parses.
    pub failure_count: u64,

    /// Average parse time in milliseconds.
    #[serde(default)]
    pub avg_time_ms: f64,

    /// When stats were last updated.
    #[serde(default = "Utc::now")]
    pub last_updated: DateTime<Utc>,
}

impl SiteStats {
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_config_creation() {
        let config = SiteConfig::new("example.com");
        assert_eq!(config.domain, "example.com");
        assert_eq!(config.preferred_method, ParseMethod::SchemaOrg);
        assert_eq!(config.rate_limit_ms, 1000);
    }

    #[test]
    fn site_stats_success_rate() {
        let stats = SiteStats {
            success_count: 80,
            failure_count: 20,
            avg_time_ms: 350.0,
            last_updated: Utc::now(),
        };
        assert!((stats.success_rate() - 0.8).abs() < 0.01);

        let empty_stats = SiteStats::default();
        assert_eq!(empty_stats.success_rate(), 0.0);
    }
}

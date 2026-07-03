//! Knowledge store models for self-improvement.
//!
//! The knowledge store contains:
//! - Site-specific parsing configurations
//! - User preference models
//! - Discovered patterns (success and anti-patterns)

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ParseMethod;

// ============================================================================
// Site Configuration
// ============================================================================

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



// ============================================================================
// User Model
// ============================================================================

/// Learned preferences for a user.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserModel {
    /// User identifier (or "default" for single-user mode).
    #[serde(default = "default_user_id")]
    pub user_id: String,

    /// Maximum preferred prep time in minutes.
    #[serde(default)]
    pub max_prep_time_minutes: Option<u32>,

    /// Maximum preferred cook time in minutes.
    #[serde(default)]
    pub max_cook_time_minutes: Option<u32>,

    /// Maximum preferred total time in minutes.
    #[serde(default)]
    pub max_total_time_minutes: Option<u32>,

    /// Whether user prefers recipes with exact quantities.
    #[serde(default)]
    pub require_quantities: bool,

    /// Maximum preferred number of ingredients.
    #[serde(default)]
    pub max_ingredients: Option<u32>,

    /// Preferred difficulty level.
    #[serde(default)]
    pub preferred_difficulty: Option<crate::models::recipe::Difficulty>,

    /// Dietary restrictions.
    #[serde(default)]
    pub dietary_restrictions: Vec<String>,

    /// Number of data points used to infer preferences.
    #[serde(default)]
    pub sample_size: u32,

    /// When this model was last updated.
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,

    /// Confidence level (0.0 - 1.0).
    #[serde(default)]
    pub confidence: f64,
}

fn default_user_id() -> String {
    "default".to_string()
}

impl UserModel {
    pub fn default_user() -> Self {
        Self {
            user_id: default_user_id(),
            max_prep_time_minutes: None,
            max_cook_time_minutes: None,
            max_total_time_minutes: None,
            require_quantities: false,
            max_ingredients: None,
            preferred_difficulty: None,
            dietary_restrictions: Vec::new(),
            sample_size: 0,
            updated_at: Utc::now(),
            confidence: 0.0,
        }
    }
}

// ============================================================================
// Patterns
// ============================================================================

/// Discovered patterns from signal analysis.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Patterns {
    /// Patterns indicating what works well.
    #[serde(default)]
    pub success_patterns: Vec<SuccessPattern>,

    /// Patterns indicating what to avoid.
    #[serde(default)]
    pub anti_patterns: Vec<AntiPattern>,

    /// When patterns were last computed.
    #[serde(default = "Utc::now")]
    pub computed_at: DateTime<Utc>,

    /// Version of the pattern set.
    #[serde(default)]
    pub version: u32,
}

/// A pattern that leads to successful scrapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessPattern {
    /// Description of the pattern.
    pub description: String,

    /// Sites where this pattern applies.
    #[serde(default)]
    pub sites: Vec<String>,

    /// Success rate when this pattern applies.
    pub success_rate: f64,

    /// Number of samples supporting this pattern.
    pub sample_size: u64,

    /// Confidence level (0.0 - 1.0).
    pub confidence: f64,
}

/// A pattern that leads to failures or bad results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiPattern {
    /// Description of what to avoid.
    pub description: String,

    /// Sites where this anti-pattern was observed.
    #[serde(default)]
    pub sites: Vec<String>,

    /// Failure rate when this pattern is present.
    pub failure_rate: f64,

    /// The action to take when this pattern is detected.
    pub action: AntiPatternAction,

    /// Number of samples supporting this anti-pattern.
    pub sample_size: u64,

    /// Confidence level (0.0 - 1.0).
    pub confidence: f64,
}

/// Action to take when an anti-pattern is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AntiPatternAction {
    /// Skip this URL entirely.
    SkipUrl,
    /// Use alternative parsing method.
    TryAlternativeMethod,
    /// Increase rate limit.
    SlowDown,
    /// Require JS rendering.
    UseHeadlessBrowser,
    /// Log warning but proceed.
    LogWarning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_config_serialization() {
        let config = SiteConfig::new("example.com");
        let json = serde_json::to_string(&config).unwrap();
        let parsed: SiteConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, "example.com");
        assert_eq!(parsed.rate_limit_ms, 1000);
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

    #[test]
    fn user_model_serialization() {
        let model = UserModel {
            user_id: "test-user".to_string(),
            max_prep_time_minutes: Some(30),
            max_cook_time_minutes: None,
            max_total_time_minutes: Some(60),
            require_quantities: true,
            max_ingredients: Some(10),
            preferred_difficulty: Some(crate::models::recipe::Difficulty::Easy),
            dietary_restrictions: vec!["vegetarian".to_string()],
            sample_size: 15,
            updated_at: Utc::now(),
            confidence: 0.75,
        };

        let json = serde_json::to_string(&model).unwrap();
        let parsed: UserModel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.user_id, "test-user");
        assert_eq!(parsed.max_prep_time_minutes, Some(30));
    }

    #[test]
    fn patterns_roundtrip() {
        let patterns = Patterns {
            success_patterns: vec![SuccessPattern {
                description: "Schema.org parsing works well".to_string(),
                sites: vec!["allrecipes.com".to_string()],
                success_rate: 0.92,
                sample_size: 100,
                confidence: 0.88,
            }],
            anti_patterns: vec![AntiPattern {
                description: "URL ends with /video".to_string(),
                sites: vec!["tasty.co".to_string()],
                failure_rate: 0.95,
                action: AntiPatternAction::SkipUrl,
                sample_size: 20,
                confidence: 0.9,
            }],
            computed_at: Utc::now(),
            version: 1,
        };

        let json = serde_json::to_string(&patterns).unwrap();
        let parsed: Patterns = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.success_patterns.len(), 1);
        assert_eq!(parsed.anti_patterns.len(), 1);
    }
}

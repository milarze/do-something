//! Discovered patterns from signal analysis.
//!
//! Patterns capture what works and what doesn't for recipe scraping.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::recipe::ParseMethod;

/// Discovered patterns from signal analysis.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecipePatterns {
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

    /// Parse method that works.
    #[serde(default)]
    pub method: ParseMethod,

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
    fn recipe_patterns_default() {
        let patterns = RecipePatterns::default();
        assert!(patterns.success_patterns.is_empty());
        assert!(patterns.anti_patterns.is_empty());
        assert_eq!(patterns.version, 0);
    }

    #[test]
    fn success_pattern_serialization() {
        let pattern = SuccessPattern {
            description: "Schema.org parsing works well".to_string(),
            sites: vec!["allrecipes.com".to_string()],
            method: ParseMethod::SchemaOrg,
            success_rate: 0.92,
            sample_size: 100,
            confidence: 0.88,
        };

        let json = serde_json::to_string(&pattern).unwrap();
        let parsed: SuccessPattern = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.description, "Schema.org parsing works well");
        assert_eq!(parsed.method, ParseMethod::SchemaOrg);
    }

    #[test]
    fn anti_pattern_serialization() {
        let pattern = AntiPattern {
            description: "URL ends with /video".to_string(),
            sites: vec!["tasty.co".to_string()],
            failure_rate: 0.95,
            action: AntiPatternAction::SkipUrl,
            sample_size: 20,
            confidence: 0.9,
        };

        let json = serde_json::to_string(&pattern).unwrap();
        let parsed: AntiPattern = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.action, AntiPatternAction::SkipUrl);
        assert!(json.contains("skip_url"));
    }
}

//! Signal logging models for feedback loops.
//!
//! Signals are discrete events that capture:
//! - Parse success/failure
//! - User actions (kept, deleted, modified recipes)
//! - Explicit feedback
//! - Performance metrics

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::ParseMethod;

/// Unique identifier for a signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SignalId(pub String);

impl SignalId {
    pub fn generate() -> Self {
        Self(format!("sig_{}", uuid::Uuid::new_v4().simple()))
    }
}

/// A discrete signal event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Unique identifier.
    #[serde(default = "SignalId::generate")]
    pub id: SignalId,

    /// Type of signal.
    #[serde(rename = "type")]
    pub signal_type: SignalType,

    /// Domain this signal relates to (if applicable).
    #[serde(default)]
    pub domain: Option<String>,

    /// URL this signal relates to (if applicable).
    #[serde(default)]
    pub url: Option<String>,

    /// Recipe ID this signal relates to (if applicable).
    #[serde(default)]
    pub recipe_id: Option<String>,

    /// Session ID this signal occurred in.
    #[serde(default)]
    pub session_id: Option<String>,

    /// User ID (for multi-user scenarios).
    #[serde(default)]
    pub user_id: Option<String>,

    /// Timestamp of the signal.
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,

    /// Additional context-specific data.
    #[serde(default)]
    pub context: serde_json::Map<String, serde_json::Value>,
}

impl Signal {
    pub fn new(signal_type: SignalType) -> Self {
        Self {
            id: SignalId::generate(),
            signal_type,
            domain: None,
            url: None,
            recipe_id: None,
            session_id: None,
            user_id: None,
            timestamp: Utc::now(),
            context: serde_json::Map::new(),
        }
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    pub fn with_recipe(mut self, recipe_id: impl Into<String>) -> Self {
        self.recipe_id = Some(recipe_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_context(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }
}

/// Types of signals the agent can record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SignalType {
    // Parse outcome signals
    ParseSuccess {
        method: ParseMethod,
        time_ms: u64,
    },
    ParseFailure {
        method: ParseMethod,
        error: String,
        attempted_methods: Vec<ParseMethod>,
    },

    // Recipe lifecycle signals
    RecipeSaved {
        recipe_id: String,
        has_image: bool,
        ingredient_count: u32,
    },
    RecipeDeleted {
        recipe_id: String,
        reason: Option<String>,
    },
    RecipeModified {
        recipe_id: String,
        modification_type: RecipeModification,
    },
    RecipeFavorited {
        recipe_id: String,
    },

    // Explicit user feedback
    ExplicitFeedback {
        feedback: String,
        url: Option<String>,
        recipe_id: Option<String>,
        sentiment: Sentiment,
    },

    // Performance signals
    RateLimitHit {
        wait_time_ms: u64,
    },
    Timeout {
        duration_ms: u64,
    },
    RetrySuccess {
        attempt_count: u32,
    },

    // Configuration signals
    ConfigUpdated {
        domain: String,
        field: String,
        old_value: Option<String>,
        new_value: String,
    },

    // Compression signals
    CompressionRun {
        signals_processed: u64,
        patterns_extracted: u64,
    },
}



/// Type of recipe modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeModification {
    IngredientChange,
    InstructionChange,
    NoteAdded,
    ServingAdjustment,
    Other,
}

/// Sentiment of explicit feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

/// Statistics about signal compression.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Number of signals processed.
    pub signals_processed: u64,

    /// Number of signals pruned.
    pub signals_pruned: u64,

    /// Number of patterns extracted.
    pub patterns_extracted: u64,

    /// Number of site configs updated.
    pub configs_updated: u64,

    /// Time taken for compression in milliseconds.
    pub time_ms: u64,

    /// When compression was run.
    #[serde(default = "Utc::now")]
    pub completed_at: DateTime<Utc>,
}

/// Aggregated statistics for a domain over a time period.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DomainStats {
    /// Domain name.
    pub domain: String,

    /// Total parse attempts.
    pub total_attempts: u64,

    /// Successful parses.
    pub successes: u64,

    /// Failed parses.
    pub failures: u64,

    /// Successes by method.
    #[serde(default)]
    pub successes_by_method: HashMap<ParseMethod, u64>,

    /// Failures by method.
    #[serde(default)]
    pub failures_by_method: HashMap<ParseMethod, u64>,

    /// Average parse time in milliseconds.
    pub avg_time_ms: f64,

    /// Most common errors.
    #[serde(default)]
    pub common_errors: Vec<String>,

    /// Time period start.
    pub period_start: DateTime<Utc>,

    /// Time period end.
    pub period_end: DateTime<Utc>,
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_builder_pattern() {
        let signal = Signal::new(SignalType::ParseSuccess {
            method: ParseMethod::SchemaOrg,
            time_ms: 350,
        })
        .with_domain("example.com")
        .with_url("https://example.com/recipe/123")
        .with_recipe("rc_abc123");

        assert_eq!(signal.domain, Some("example.com".to_string()));
        assert_eq!(signal.url, Some("https://example.com/recipe/123".to_string()));
        assert_eq!(signal.recipe_id, Some("rc_abc123".to_string()));
    }

    #[test]
    fn signal_serialization() {
        let signal = Signal::new(SignalType::ParseSuccess {
            method: ParseMethod::SchemaOrg,
            time_ms: 350,
        });

        let json = serde_json::to_string(&signal).unwrap();
        let parsed: Signal = serde_json::from_str(&json).unwrap();

        match parsed.signal_type {
            SignalType::ParseSuccess { method, time_ms } => {
                assert_eq!(method, ParseMethod::SchemaOrg);
                assert_eq!(time_ms, 350);
            }
            _ => panic!("Wrong signal type"),
        }
    }

    #[test]
    fn explicit_feedback_signal() {
        let signal = Signal::new(SignalType::ExplicitFeedback {
            feedback: "That's a video page, not a recipe".to_string(),
            url: Some("https://tasty.co/recipe/123/video".to_string()),
            recipe_id: None,
            sentiment: Sentiment::Negative,
        });

        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("explicit_feedback"));
        assert!(json.contains("video page"));
    }
}

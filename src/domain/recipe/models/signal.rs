//! Signal logging models for recipe domain.
//!
//! Signals are discrete events that capture parse outcomes,
//! user feedback, and performance metrics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::recipe::ParseMethod;

/// Unique identifier for a signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SignalId(pub String);

impl SignalId {
    pub fn generate() -> Self {
        Self(format!("sig_{}", uuid::Uuid::new_v4().simple()))
    }
}

/// A discrete signal event in the recipe domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeSignal {
    /// Unique identifier.
    #[serde(default = "SignalId::generate")]
    pub id: SignalId,

    /// Type of signal.
    #[serde(rename = "type")]
    pub signal_type: RecipeSignalType,

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

impl RecipeSignal {
    pub fn new(signal_type: RecipeSignalType) -> Self {
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

/// Types of signals the recipe agent can record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecipeSignalType {
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
        reason: String,
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
}

/// Sentiment of explicit feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_builder_pattern() {
        let signal = RecipeSignal::new(RecipeSignalType::ParseSuccess {
            method: ParseMethod::SchemaOrg,
            time_ms: 350,
        })
        .with_domain("example.com")
        .with_url("https://example.com/recipe/123")
        .with_recipe("rc_abc123");

        assert_eq!(signal.domain, Some("example.com".to_string()));
        assert_eq!(
            signal.url,
            Some("https://example.com/recipe/123".to_string())
        );
        assert_eq!(signal.recipe_id, Some("rc_abc123".to_string()));
    }

    #[test]
    fn signal_serialization() {
        let signal = RecipeSignal::new(RecipeSignalType::ParseSuccess {
            method: ParseMethod::SchemaOrg,
            time_ms: 350,
        });

        let json = serde_json::to_string(&signal).unwrap();
        let parsed: RecipeSignal = serde_json::from_str(&json).unwrap();

        match parsed.signal_type {
            RecipeSignalType::ParseSuccess { method, time_ms } => {
                assert_eq!(method, ParseMethod::SchemaOrg);
                assert_eq!(time_ms, 350);
            }
            _ => panic!("Wrong signal type"),
        }
    }

    #[test]
    fn explicit_feedback_signal() {
        let signal = RecipeSignal::new(RecipeSignalType::ExplicitFeedback {
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

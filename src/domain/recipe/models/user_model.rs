//! User preference model.
//!
//! Learned preferences for a user.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::super::models::Difficulty;

/// Learned preferences for a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub preferred_difficulty: Option<Difficulty>,

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_model_creation() {
        let model = UserModel::default_user();
        assert_eq!(model.user_id, "default");
        assert_eq!(model.sample_size, 0);
        assert_eq!(model.confidence, 0.0);
    }
}

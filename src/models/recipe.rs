//! Recipe data models.

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecipeId(pub String);

impl RecipeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn generate() -> Self {
        Self(format!("rc_{}", uuid::Uuid::new_v4().simple()))
    }
}

impl std::fmt::Display for RecipeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for RecipeId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A parsed recipe with all its components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    /// Unique identifier (generated on save).
    #[serde(default = "RecipeId::generate")]
    pub id: RecipeId,

    /// Recipe display name.
    pub name: String,

    /// Source URL where the recipe was scraped from.
    pub source_url: url::Url,

    /// Site domain for quick lookups.
    pub source_domain: String,

    /// List of ingredients with quantities.
    #[serde(default)]
    pub ingredients: Vec<Ingredient>,

    /// Step-by-step instructions.
    #[serde(default)]
    pub instructions: Vec<String>,

    /// Preparation time in minutes (if available).
    #[serde(default)]
    pub prep_time_minutes: Option<u32>,

    /// Cooking time in minutes (if available).
    #[serde(default)]
    pub cook_time_minutes: Option<u32>,

    /// Total time in minutes (if available).
    #[serde(default)]
    pub total_time_minutes: Option<u32>,

    /// Number of servings/yield.
    #[serde(default)]
    pub r#yield: Option<Servings>,

    /// Cuisine type (e.g., "Italian", "Japanese").
    #[serde(default)]
    pub cuisine: Option<String>,

    /// Difficulty level.
    #[serde(default)]
    pub difficulty: Option<Difficulty>,

    /// Freeform tags/categories.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Nutritional information (if available).
    #[serde(default)]
    pub nutrition: Option<NutritionInfo>,

    /// Recipe image URL.
    #[serde(default)]
    pub image_url: Option<url::Url>,

    /// Author/creator name.
    #[serde(default)]
    pub author: Option<String>,

    /// Description or summary text.
    #[serde(default)]
    pub description: Option<String>,

    /// When this recipe was scraped.
    #[serde(default = "Utc::now")]
    pub scraped_at: DateTime<Utc>,

    /// Hash for deduplication (computed from source_url + name).
    #[serde(default)]
    pub content_hash: Option<String>,

    /// Arbitrary metadata.
    #[serde(default)]
    pub meta: HashMap<String, serde_json::Value>,
}

impl Recipe {
    /// Compute a content hash for deduplication.
    pub fn compute_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.source_url.hash(&mut hasher);
        self.name.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// An ingredient with optional quantity and unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ingredient {
    /// Raw text as it appears in the recipe.
    pub raw: String,

    /// Parsed quantity (numeric part).
    #[serde(default)]
    pub quantity: Option<f64>,

    /// Unit of measurement (e.g., "cup", "g", "tbsp").
    #[serde(default)]
    pub unit: Option<String>,

    /// Ingredient name (e.g., "flour", "chicken breast").
    #[serde(default)]
    pub name: Option<String>,

    /// Additional notes (e.g., "diced", "room temperature").
    #[serde(default)]
    pub notes: Option<String>,
}

impl Ingredient {
    /// Create an ingredient from raw text only.
    pub fn from_raw(raw: impl Into<String>) -> Self {
        Self {
            raw: raw.into(),
            quantity: None,
            unit: None,
            name: None,
            notes: None,
        }
    }
}

/// Servings/yield representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Servings {
    /// Minimum servings.
    pub min: u32,

    /// Maximum servings (for ranges like "4-6 servings").
    #[serde(default)]
    pub max: Option<u32>,

    /// Raw text from the recipe.
    #[serde(default)]
    pub raw: Option<String>,
}

impl Servings {
    pub fn single(count: u32) -> Self {
        Self {
            min: count,
            max: None,
            raw: None,
        }
    }

    pub fn range(min: u32, max: u32) -> Self {
        Self {
            min,
            max: Some(max),
            raw: None,
        }
    }
}

/// Recipe difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// Nutritional information per serving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutritionInfo {
    #[serde(default)]
    pub calories: Option<u32>,

    #[serde(default)]
    pub fat_g: Option<f64>,

    #[serde(default)]
    pub saturated_fat_g: Option<f64>,

    #[serde(default)]
    pub carbohydrates_g: Option<f64>,

    #[serde(default)]
    pub protein_g: Option<f64>,

    #[serde(default)]
    pub fiber_g: Option<f64>,

    #[serde(default)]
    pub sodium_mg: Option<u32>,

    /// Additional nutritional data.
    #[serde(default)]
    pub extra: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_serialization_roundtrip() {
        let recipe = Recipe {
            id: RecipeId::new("test-123"),
            name: "Test Recipe".to_string(),
            source_url: "https://example.com/recipe".parse().unwrap(),
            source_domain: "example.com".to_string(),
            ingredients: vec![
                Ingredient {
                    raw: "1 cup flour".to_string(),
                    quantity: Some(1.0),
                    unit: Some("cup".to_string()),
                    name: Some("flour".to_string()),
                    notes: None,
                },
            ],
            instructions: vec!["Mix ingredients".to_string()],
            prep_time_minutes: Some(10),
            cook_time_minutes: Some(30),
            total_time_minutes: Some(40),
            r#yield: Some(Servings::single(4)),
            cuisine: Some("American".to_string()),
            difficulty: Some(Difficulty::Easy),
            tags: vec!["quick".to_string()],
            nutrition: None,
            image_url: None,
            author: None,
            description: None,
            scraped_at: Utc::now(),
            content_hash: None,
            meta: HashMap::new(),
        };

        let json = serde_json::to_string(&recipe).unwrap();
        let parsed: Recipe = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, recipe.name);
        assert_eq!(parsed.ingredients.len(), 1);
        assert_eq!(parsed.r#yield.unwrap().min, 4);
    }

    #[test]
    fn ingredient_from_raw() {
        let ing = Ingredient::from_raw("2 cups all-purpose flour, sifted");
        assert_eq!(ing.raw, "2 cups all-purpose flour, sifted");
        assert!(ing.quantity.is_none());
    }

    #[test]
    fn servings_range() {
        let servings = Servings::range(4, 6);
        assert_eq!(servings.min, 4);
        assert_eq!(servings.max, Some(6));
    }
}

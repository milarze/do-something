//! User preference model management.
//!
//! Tracks learned user preferences from recipe interactions
//! (kept vs deleted recipes) to improve future recommendations.

use crate::models::knowledge::UserModel;
use crate::models::recipe::{Difficulty, Recipe};

/// Manager for user preference models.
pub struct UserModelManager;

impl UserModelManager {
    /// Create a new empty user model.
    pub fn create(user_id: &str) -> UserModel {
        UserModel {
            user_id: user_id.to_string(),
            max_prep_time_minutes: None,
            max_cook_time_minutes: None,
            max_total_time_minutes: None,
            require_quantities: false,
            max_ingredients: None,
            preferred_difficulty: None,
            dietary_restrictions: Vec::new(),
            sample_size: 0,
            updated_at: chrono::Utc::now(),
            confidence: 0.0,
        }
    }

    /// Record that a user kept (saved) a recipe.
    ///
    /// Updates preferences based on the recipe's characteristics.
    /// Call this when a recipe is successfully scraped and saved.
    pub fn record_recipe_kept(model: &mut UserModel, recipe: &Recipe) {
        model.sample_size += 1;

        // Track time preferences
        if let Some(prep) = recipe.prep_time_minutes {
            Self::update_max_preference(&mut model.max_prep_time_minutes, prep);
        }
        if let Some(cook) = recipe.cook_time_minutes {
            Self::update_max_preference(&mut model.max_cook_time_minutes, cook);
        }
        if let Some(total) = recipe.total_time_minutes {
            Self::update_max_preference(&mut model.max_total_time_minutes, total);
        }

        // Track ingredient count
        let ingredient_count = recipe.ingredients.len() as u32;
        Self::update_max_preference(&mut model.max_ingredients, ingredient_count);

        // Track difficulty preference
        if let Some(difficulty) = &recipe.difficulty {
            Self::update_difficulty_preference(model, difficulty);
        }

        // Track dietary tags
        for tag in &recipe.tags {
            if !model.dietary_restrictions.contains(tag) {
                // Only add if it appears multiple times (avoid noise)
                // For simplicity, we add all unique tags seen
                model.dietary_restrictions.push(tag.clone());
            }
        }

        model.updated_at = chrono::Utc::now();
    }

    /// Record that a user deleted a recipe.
    ///
    /// Adjusts preferences to avoid similar recipes in the future.
    /// Call this when a user explicitly removes a saved recipe.
    pub fn record_recipe_deleted(model: &mut UserModel, recipe: &Recipe) {
        // When a recipe is deleted, we become more restrictive
        // This is a signal that something about the recipe was undesirable

        // Tighten time constraints
        if let Some(prep) = recipe.prep_time_minutes {
            Self::tighten_max_preference(&mut model.max_prep_time_minutes, prep);
        }
        if let Some(cook) = recipe.cook_time_minutes {
            Self::tighten_max_preference(&mut model.max_cook_time_minutes, cook);
        }
        if let Some(total) = recipe.total_time_minutes {
            Self::tighten_max_preference(&mut model.max_total_time_minutes, total);
        }

        // Tighten ingredient count
        let ingredient_count = recipe.ingredients.len() as u32;
        Self::tighten_max_preference(&mut model.max_ingredients, ingredient_count);

        model.updated_at = chrono::Utc::now();
    }

    /// Update confidence based on sample size.
    ///
    /// Confidence increases logarithmically with more samples.
    /// Call this before persisting the model.
    pub fn update_confidence(model: &mut UserModel) {
        // Confidence formula: log2(sample_size + 1) / 10
        // Reaches 1.0 at ~1000 samples
        let sample_based_confidence = (model.sample_size as f64 + 1.0).log2() / 10.0;
        model.confidence = sample_based_confidence.min(1.0);
    }

    /// Check how well a recipe matches user preferences.
    ///
    /// Returns a score from 0.0 to 1.0, where 1.0 is a perfect match.
    /// A score below 0.5 suggests the recipe likely won't be wanted.
    pub fn matches_preferences(model: &UserModel, recipe: &Recipe) -> f64 {
        if model.sample_size == 0 {
            return 1.0; // No data, assume neutral
        }

        let mut score = 1.0;
        let mut factors = 0;

        // Check time constraints
        if let Some(max_prep) = model.max_prep_time_minutes
            && let Some(prep) = recipe.prep_time_minutes
        {
            factors += 1;
            if prep > max_prep {
                score *= 1.0 - ((prep - max_prep) as f64 / max_prep as f64).min(1.0);
            }
        }

        if let Some(max_cook) = model.max_cook_time_minutes
            && let Some(cook) = recipe.cook_time_minutes
        {
            factors += 1;
            if cook > max_cook {
                score *= 1.0 - ((cook - max_cook) as f64 / max_cook as f64).min(1.0);
            }
        }

        if let Some(max_total) = model.max_total_time_minutes
            && let Some(total) = recipe.total_time_minutes
        {
            factors += 1;
            if total > max_total {
                score *= 1.0 - ((total - max_total) as f64 / max_total as f64).min(1.0);
            }
        }

        // Check ingredient count
        if let Some(max_ing) = model.max_ingredients {
            let count = recipe.ingredients.len() as u32;
            factors += 1;
            if count > max_ing {
                score *= 1.0 - ((count - max_ing) as f64 / max_ing as f64).min(1.0);
            }
        }

        // Check difficulty
        if let Some(pref_diff) = &model.preferred_difficulty
            && let Some(recipe_diff) = &recipe.difficulty
        {
            factors += 1;
            let diff_score = difficulty_distance(pref_diff, recipe_diff);
            score *= diff_score;
        }

        if factors == 0 { 1.0 } else { score }
    }

    /// Update a max preference value (relax constraint).
    fn update_max_preference(current: &mut Option<u32>, new_value: u32) {
        match current {
            Some(max) => {
                // Allow up to 50% more than the largest seen
                *max = (*max).max(new_value);
            }
            None => {
                *current = Some(new_value);
            }
        }
    }

    /// Tighten a max preference value (more restrictive).
    fn tighten_max_preference(current: &mut Option<u32>, rejected_value: u32) {
        match current {
            Some(max) => {
                // Only tighten if the rejected value was at or below current max
                if rejected_value <= *max {
                    // Reduce by 20%, but not below the rejected value
                    *max = ((*max as f64 * 0.8) as u32).max(1);
                }
            }
            None => {
                // Set a restrictive value based on what was rejected
                *current = Some(((rejected_value as f64 * 0.8) as u32).max(1));
            }
        }
    }

    /// Update difficulty preference.
    fn update_difficulty_preference(model: &mut UserModel, difficulty: &Difficulty) {
        // Prefer the difficulty that appears most often
        // For simplicity, track the most recent
        model.preferred_difficulty = Some(*difficulty);
    }
}

/// Calculate distance between difficulties (0.0 to 1.0).
fn difficulty_distance(preferred: &Difficulty, actual: &Difficulty) -> f64 {
    use Difficulty::*;
    let pref_level: i32 = match preferred {
        Easy => 0,
        Medium => 1,
        Hard => 2,
    };
    let actual_level: i32 = match actual {
        Easy => 0,
        Medium => 1,
        Hard => 2,
    };

    let diff = (pref_level - actual_level).abs();
    match diff {
        0 => 1.0,
        1 => 0.7,
        _ => 0.4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::recipe::{Ingredient, Servings};
    use url::Url;

    fn test_recipe() -> Recipe {
        Recipe {
            id: crate::models::recipe::RecipeId::generate(),
            name: "Test Recipe".to_string(),
            source_url: Url::parse("https://example.com/recipe").unwrap(),
            source_domain: "example.com".to_string(),
            ingredients: vec![
                Ingredient::from_raw("1 cup flour"),
                Ingredient::from_raw("2 eggs"),
            ],
            instructions: vec!["Mix".to_string()],
            prep_time_minutes: Some(15),
            cook_time_minutes: Some(30),
            total_time_minutes: Some(45),
            servings: Some(Servings::single(4)),
            cuisine: None,
            difficulty: Some(Difficulty::Easy),
            tags: vec!["quick".to_string()],
            nutrition: None,
            image_url: None,
            author: None,
            description: None,
            scraped_at: chrono::Utc::now(),
            content_hash: None,
            meta: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn create_creates_empty_model() {
        let model = UserModelManager::create("test-user");
        assert_eq!(model.user_id, "test-user");
        assert_eq!(model.sample_size, 0);
        assert_eq!(model.confidence, 0.0);
    }

    #[test]
    fn record_recipe_kept_increments_sample_size() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        UserModelManager::record_recipe_kept(&mut model, &recipe);
        
        assert_eq!(model.sample_size, 1);
    }

    #[test]
    fn record_recipe_kept_updates_time_preferences() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        UserModelManager::record_recipe_kept(&mut model, &recipe);
        
        assert_eq!(model.max_prep_time_minutes, Some(15));
        assert_eq!(model.max_cook_time_minutes, Some(30));
        assert_eq!(model.max_total_time_minutes, Some(45));
    }

    #[test]
    fn record_recipe_kept_updates_ingredient_count() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        UserModelManager::record_recipe_kept(&mut model, &recipe);
        
        assert_eq!(model.max_ingredients, Some(2));
    }

    #[test]
    fn record_recipe_kept_tracks_difficulty() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        UserModelManager::record_recipe_kept(&mut model, &recipe);
        
        assert_eq!(model.preferred_difficulty, Some(Difficulty::Easy));
    }

    #[test]
    fn record_recipe_kept_tracks_tags() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        UserModelManager::record_recipe_kept(&mut model, &recipe);
        
        assert!(model.dietary_restrictions.contains(&"quick".to_string()));
    }

    #[test]
    fn record_recipe_deleted_tightens_preferences() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        // First record a kept recipe
        UserModelManager::record_recipe_kept(&mut model, &recipe);
        
        // Then delete a similar recipe
        UserModelManager::record_recipe_deleted(&mut model, &recipe);
        
        // Preferences should be tighter
        assert!(model.max_prep_time_minutes.unwrap() <= 15);
        assert!(model.max_cook_time_minutes.unwrap() <= 30);
    }

    #[test]
    fn update_confidence_increases_with_samples() {
        let mut model = UserModel::default_user();
        
        model.sample_size = 0;
        UserModelManager::update_confidence(&mut model);
        assert!(model.confidence < 0.5);

        model.sample_size = 10;
        UserModelManager::update_confidence(&mut model);
        assert!(model.confidence > 0.3);

        model.sample_size = 100;
        UserModelManager::update_confidence(&mut model);
        assert!(model.confidence > 0.5);

        model.sample_size = 1023; // log2(1024) = 10, so confidence = 1.0
        UserModelManager::update_confidence(&mut model);
        assert!(model.confidence >= 1.0);
    }

    #[test]
    fn matches_preferences_returns_high_for_matching_recipe() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        // Train on the same recipe
        UserModelManager::record_recipe_kept(&mut model, &recipe);
        UserModelManager::update_confidence(&mut model);

        // Should match well
        let score = UserModelManager::matches_preferences(&model, &recipe);
        assert!(score >= 0.8);
    }

    #[test]
    fn matches_preferences_returns_low_for_non_matching_recipe() {
        let mut model = UserModel::default_user();
        let recipe = test_recipe();

        // Train on the recipe
        UserModelManager::record_recipe_kept(&mut model, &recipe);
        UserModelManager::update_confidence(&mut model);

        // Create a very different recipe
        let hard_recipe = Recipe {
            difficulty: Some(Difficulty::Hard),
            prep_time_minutes: Some(120),
            cook_time_minutes: Some(180),
            total_time_minutes: Some(300),
            ingredients: vec![Ingredient::from_raw("many"); 20],
            ..test_recipe()
        };

        let score = UserModelManager::matches_preferences(&model, &hard_recipe);
        assert!(score < 0.7);
    }

    #[test]
    fn matches_preferences_returns_one_for_no_data() {
        let model = UserModel::default_user();
        let recipe = test_recipe();

        // No training data, should return neutral
        let score = UserModelManager::matches_preferences(&model, &recipe);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn multiple_kept_recipes_expand_preferences() {
        let mut model = UserModel::default_user();

        // Keep recipes with increasing prep times
        for prep_time in [10, 20, 30].iter() {
            let mut recipe = test_recipe();
            recipe.prep_time_minutes = Some(*prep_time);
            UserModelManager::record_recipe_kept(&mut model, &recipe);
        }

        // Max should be the largest seen
        assert_eq!(model.max_prep_time_minutes, Some(30));
    }

    #[test]
    fn deleted_recipe_reduces_max() {
        let mut model = UserModel::default_user();
        model.max_prep_time_minutes = Some(60);

        // Delete a recipe with 30 min prep time
        let mut recipe = test_recipe();
        recipe.prep_time_minutes = Some(30);
        
        UserModelManager::record_recipe_deleted(&mut model, &recipe);

        // Max should be reduced (tightened)
        assert!(model.max_prep_time_minutes.unwrap() < 60);
    }
}

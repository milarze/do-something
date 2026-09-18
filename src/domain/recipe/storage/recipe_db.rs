//! Recipe database built over framework AppendOnlyDb.
//!
//! Provides recipe-specific storage with search capabilities.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::framework::storage::append_only_db::HasId;
use crate::framework::storage::{AppendOnlyDb, StorageError};

use super::super::models::{Recipe, RecipeId};
use super::traits::RecipesStorage;

/// Append-only recipe storage with search index.
pub struct RecipeDb {
    inner: AppendOnlyDb<Recipe>,
    /// Mutex for check-and-insert operations
    lock: Mutex<()>,
}

impl RecipeDb {
    /// Open the recipe database at the given directory.
    pub fn open(dir: PathBuf) -> crate::framework::storage::Result<Self> {
        let inner = AppendOnlyDb::open(dir)?;
        Ok(Self {
            inner,
            lock: Mutex::new(()),
        })
    }

    /// Rebuild the search index.
    pub fn rebuild_index(&self) -> crate::framework::storage::Result<()> {
        self.inner.rebuild_index()
    }
}

impl RecipesStorage for RecipeDb {
    fn insert(&self, recipe: &Recipe) -> crate::framework::storage::Result<RecipeId> {
        // Lock to prevent race condition between check and insert
        let _guard = self
            .lock
            .lock()
            .map_err(|e| StorageError::LockError(format!("lock poisoned: {}", e)))?;

        // Check for duplicates by URL
        let all = self.inner.all()?;
        for existing in &all {
            if existing.source_url == recipe.source_url {
                return Ok(existing.id.clone());
            }
        }

        // Compute content hash if not present
        let mut recipe = recipe.clone();
        if recipe.content_hash.is_none() {
            recipe.content_hash = Some(recipe.compute_hash());
        }

        // Check for duplicates by content hash
        if let Some(hash) = &recipe.content_hash {
            for existing in &all {
                if existing.content_hash.as_deref() == Some(hash) {
                    return Ok(existing.id.clone());
                }
            }
        }

        let id = self.inner.insert(&recipe)?;
        Ok(RecipeId::new(id))
    }

    fn get(&self, id: &RecipeId) -> crate::framework::storage::Result<Option<Recipe>> {
        self.inner.get(id.as_ref())
    }

    fn exists_by_url(&self, url: &str) -> crate::framework::storage::Result<bool> {
        let all = self.inner.all()?;
        Ok(all.iter().any(|r| r.source_url.as_str() == url))
    }

    fn search(&self, query: &str) -> crate::framework::storage::Result<Vec<RecipeId>> {
        let query = query.to_lowercase();
        let terms: Vec<&str> = query.split_whitespace().collect();

        let all = self.inner.all()?;
        let mut results = Vec::new();

        for recipe in all {
            let name_lower = recipe.name.to_lowercase();
            let ingredients_text: String = recipe
                .ingredients
                .iter()
                .map(|i| i.raw.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");

            // Check if all terms match either in name or ingredients
            if terms
                .iter()
                .all(|t| name_lower.contains(t) || ingredients_text.contains(t))
            {
                results.push(recipe.id);
            }
        }

        Ok(results)
    }

    fn count(&self) -> crate::framework::storage::Result<u64> {
        self.inner.count()
    }

    fn all(&self) -> crate::framework::storage::Result<Vec<Recipe>> {
        self.inner.all()
    }
}

// Implement HasId for Recipe to work with AppendOnlyDb
impl HasId for Recipe {
    fn generate_id() -> String {
        format!("rc_{}", uuid::Uuid::new_v4().simple())
    }

    fn get_id(&self) -> Option<String> {
        Some(self.id.0.clone())
    }

    fn set_id(&mut self, id: String) {
        self.id = RecipeId::new(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::recipe::models::Ingredient;
    use chrono::Utc;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn test_recipe(name: &str, url: &str) -> Recipe {
        Recipe {
            id: RecipeId::generate(),
            name: name.to_string(),
            source_url: url.parse().unwrap(),
            source_domain: "example.com".to_string(),
            ingredients: vec![],
            instructions: vec![],
            prep_time_minutes: None,
            cook_time_minutes: None,
            total_time_minutes: None,
            servings: None,
            cuisine: None,
            difficulty: None,
            tags: vec![],
            nutrition: None,
            image_url: None,
            author: None,
            description: None,
            scraped_at: Utc::now(),
            content_hash: None,
            meta: HashMap::new(),
        }
    }

    fn test_recipe_with_ingredients(name: &str, url: &str, ingredients: Vec<&str>) -> Recipe {
        Recipe {
            id: RecipeId::generate(),
            name: name.to_string(),
            source_url: url.parse().unwrap(),
            source_domain: "example.com".to_string(),
            ingredients: ingredients.into_iter().map(Ingredient::from_raw).collect(),
            instructions: vec![],
            prep_time_minutes: None,
            cook_time_minutes: None,
            total_time_minutes: None,
            servings: None,
            cuisine: None,
            difficulty: None,
            tags: vec![],
            nutrition: None,
            image_url: None,
            author: None,
            description: None,
            scraped_at: Utc::now(),
            content_hash: None,
            meta: HashMap::new(),
        }
    }

    #[test]
    fn insert_generates_unique_ids() {
        let dir = tempdir().unwrap();
        let db = RecipeDb::open(dir.path().to_path_buf()).unwrap();

        let recipe1 = test_recipe("Recipe 1", "https://example.com/recipe1");
        let recipe2 = test_recipe("Recipe 2", "https://example.com/recipe2");

        let id1 = db.insert(&recipe1).unwrap();
        let id2 = db.insert(&recipe2).unwrap();

        assert_ne!(id1, id2);
    }

    #[test]
    fn get_returns_inserted_recipe() {
        let dir = tempdir().unwrap();
        let db = RecipeDb::open(dir.path().to_path_buf()).unwrap();

        let recipe = test_recipe("Test Recipe", "https://example.com/test");
        let id = db.insert(&recipe).unwrap();

        let retrieved = db.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Test Recipe");
    }

    #[test]
    fn exists_by_url_detects_duplicates() {
        let dir = tempdir().unwrap();
        let db = RecipeDb::open(dir.path().to_path_buf()).unwrap();

        let recipe = test_recipe("Test", "https://example.com/test");
        db.insert(&recipe).unwrap();

        assert!(db.exists_by_url("https://example.com/test").unwrap());
        assert!(!db.exists_by_url("https://example.com/other").unwrap());
    }

    #[test]
    fn search_finds_matching_recipes() {
        let dir = tempdir().unwrap();
        let db = RecipeDb::open(dir.path().to_path_buf()).unwrap();

        let recipe1 = test_recipe("Chocolate Cake", "https://example.com/cake");
        let recipe2 = test_recipe("Apple Pie", "https://example.com/pie");
        let recipe3 = test_recipe("Chocolate Mousse", "https://example.com/mousse");

        db.insert(&recipe1).unwrap();
        db.insert(&recipe2).unwrap();
        db.insert(&recipe3).unwrap();

        let results = db.search("chocolate").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_finds_by_ingredients() {
        let dir = tempdir().unwrap();
        let db = RecipeDb::open(dir.path().to_path_buf()).unwrap();

        let r1 = test_recipe_with_ingredients(
            "Simple Cake",
            "https://example.com/cake",
            vec!["flour", "sugar", "eggs"],
        );
        let r2 = test_recipe_with_ingredients(
            "Pasta",
            "https://example.com/pasta",
            vec!["pasta", "tomatoes", "basil"],
        );

        db.insert(&r1).unwrap();
        db.insert(&r2).unwrap();

        // Search for ingredient
        let results = db.search("flour").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn duplicate_by_url_not_inserted() {
        let dir = tempdir().unwrap();
        let db = RecipeDb::open(dir.path().to_path_buf()).unwrap();

        let r1 = test_recipe("Recipe 1", "https://example.com/test");
        let r2 = test_recipe("Recipe 2", "https://example.com/test"); // Same URL, different name

        let id1 = db.insert(&r1).unwrap();
        let id2 = db.insert(&r2).unwrap();

        // Should return the same ID for duplicate URL
        assert_eq!(id1, id2);
        assert_eq!(db.count().unwrap(), 1);
    }
}

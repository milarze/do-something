//! File-based recipe storage using JSONL format.
//!
//! Provides simple append-only storage. For indexed/search-heavy workloads,
//! use [`SqliteRecipesDb`] instead.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use crate::models::recipe::{Recipe, RecipeId};

use super::error::Result;
use super::traits::RecipesStorage;

/// File-based, append-only recipe storage.
///
/// Note: Search operations scan all recipes. For better performance
/// with large datasets, use [`SqliteRecipesDb`].
#[derive(Debug)]
pub struct FileRecipesDb {
    dir: PathBuf,
}

impl FileRecipesDb {
    /// Open the recipes database at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Get the path to the recipes JSONL file.
    fn recipes_file(&self) -> PathBuf {
        self.dir.join("recipes.jsonl")
    }

    /// Scan recipes file and find by predicate.
    fn find_by<P>(&self, predicate: P) -> Result<Option<Recipe>>
    where
        P: Fn(&Recipe) -> bool,
    {
        if !self.recipes_file().exists() {
            return Ok(None);
        }

        let file = File::open(self.recipes_file())?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Ok(recipe) = serde_json::from_str::<Recipe>(&line)
                && predicate(&recipe)
            {
                return Ok(Some(recipe));
            }
        }

        Ok(None)
    }

    /// Check if content hash already exists.
    pub fn exists_by_content_hash(&self, hash: &str) -> Result<bool> {
        Ok(self
            .find_by(|r| r.content_hash.as_deref() == Some(hash))?
            .is_some())
    }
}

impl RecipesStorage for FileRecipesDb {
    fn insert(&self, recipe: &Recipe) -> Result<RecipeId> {
        // Check for duplicates by content hash
        if let Some(hash) = &recipe.content_hash
            && self.exists_by_content_hash(hash)?
        {
            // Return existing ID
            if let Some(existing) =
                self.find_by(|r| r.content_hash.as_deref() == Some(hash))?
            {
                return Ok(existing.id);
            }
        }

        let id = RecipeId::generate();
        let mut recipe = recipe.clone();
        recipe.id = id.clone();

        // Compute content hash if not present
        if recipe.content_hash.is_none() {
            recipe.content_hash = Some(recipe.compute_hash());
        }

        // Append to JSONL file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.recipes_file())?;
        let mut writer = BufWriter::new(file);
        let json = serde_json::to_string(&recipe)?;
        writeln!(writer, "{}", json)?;

        Ok(id)
    }

    fn get(&self, id: &RecipeId) -> Result<Option<Recipe>> {
        self.find_by(|r| &r.id == id)
    }

    fn exists_by_url(&self, url: &str) -> Result<bool> {
        Ok(self.find_by(|r| r.source_url.as_str() == url)?.is_some())
    }

    fn search(&self, query: &str) -> Result<Vec<RecipeId>> {
        let query = query.to_lowercase();
        let terms: Vec<&str> = query.split_whitespace().collect();

        let mut results = Vec::new();

        if !self.recipes_file().exists() {
            return Ok(results);
        }

        let file = File::open(self.recipes_file())?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Ok(recipe) = serde_json::from_str::<Recipe>(&line) {
                let name_lower = recipe.name.to_lowercase();
                if terms.iter().all(|t| name_lower.contains(t)) {
                    results.push(recipe.id);
                }
            }
        }

        Ok(results)
    }

    fn count(&self) -> Result<u64> {
        if !self.recipes_file().exists() {
            return Ok(0);
        }

        let file = File::open(self.recipes_file())?;
        let reader = BufReader::new(file);
        Ok(reader.lines().filter(|l| l.is_ok()).count() as u64)
    }

    fn all(&self) -> Result<Vec<Recipe>> {
        let mut recipes = Vec::new();

        if !self.recipes_file().exists() {
            return Ok(recipes);
        }

        let file = File::open(self.recipes_file())?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Ok(recipe) = serde_json::from_str::<Recipe>(&line) {
                recipes.push(recipe);
            }
        }

        Ok(recipes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
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
            meta: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn insert_generates_unique_ids() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        let recipe1 = test_recipe("Recipe 1", "https://example.com/recipe1");
        let recipe2 = test_recipe("Recipe 2", "https://example.com/recipe2");

        let id1 = db.insert(&recipe1).unwrap();
        let id2 = db.insert(&recipe2).unwrap();

        assert_ne!(id1, id2);
    }

    #[test]
    fn get_returns_inserted_recipe() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        let recipe = test_recipe("Test Recipe", "https://example.com/test");
        let id = db.insert(&recipe).unwrap();

        let retrieved = db.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Test Recipe");
    }

    #[test]
    fn exists_by_url_detects_duplicates() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        let recipe = test_recipe("Test", "https://example.com/test");
        db.insert(&recipe).unwrap();

        assert!(db.exists_by_url("https://example.com/test").unwrap());
        assert!(!db.exists_by_url("https://example.com/other").unwrap());
    }

    #[test]
    fn search_finds_matching_recipes() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

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
    fn duplicate_by_content_hash_not_inserted() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        let mut recipe1 = test_recipe("Test Recipe", "https://example.com/test1");
        recipe1.content_hash = Some("hash123".to_string());

        let mut recipe2 = test_recipe("Test Recipe", "https://example.com/test2");
        recipe2.content_hash = Some("hash123".to_string());

        let id1 = db.insert(&recipe1).unwrap();
        let id2 = db.insert(&recipe2).unwrap();

        // Should return the same ID for duplicates
        assert_eq!(id1, id2);
        assert_eq!(db.count().unwrap(), 1);
    }
}

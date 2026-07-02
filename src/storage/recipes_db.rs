//! File-based recipe storage using JSONL format.
//!
//! Provides simple append-only storage. Search is O(n) - scans all
//! recipes.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::models::recipe::{Recipe, RecipeId};

use super::error::Result;
use super::traits::RecipesStorage;

/// File-based, append-only recipe storage.
///
/// Uses a mutex to ensure thread-safe operations. Search operations scan all
/// recipes.
#[derive(Debug)]
pub struct FileRecipesDb {
    dir: PathBuf,
    /// Mutex to prevent race conditions between check-and-insert operations
    lock: Mutex<()>,
}

impl FileRecipesDb {
    /// Open the recipes database at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            lock: Mutex::new(()),
        })
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

    /// Find recipe by URL.
    fn find_by_url(&self, url: &str) -> Result<Option<Recipe>> {
        self.find_by(|r| r.source_url.as_str() == url)
    }
}

impl RecipesStorage for FileRecipesDb {
    fn insert(&self, recipe: &Recipe) -> Result<RecipeId> {
        // Lock to prevent race condition between check and insert
        let _guard = self.lock.lock().map_err(|e| {
            std::io::Error::other(format!("lock poisoned: {}", e))
        })?;

        // Check for duplicates by content hash
        if let Some(hash) = &recipe.content_hash
            && let Some(existing) =
                self.find_by(|r| r.content_hash.as_deref() == Some(hash))?
        {
            return Ok(existing.id);
        }

        // Also check for duplicates by URL
        let url_str = recipe.source_url.to_string();
        if let Some(existing) = self.find_by_url(&url_str)? {
            return Ok(existing.id);
        }

        let id = RecipeId::generate();
        let mut recipe = recipe.clone();
        recipe.id = id.clone();

        // Compute content hash if not present
        if recipe.content_hash.is_none() {
            recipe.content_hash = Some(recipe.compute_hash());
        }

        // Append to JSONL file with explicit flush for durability
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.recipes_file())?;
        let mut writer = BufWriter::new(file);
        let json = serde_json::to_string(&recipe)?;
        writeln!(writer, "{}", json)?;
        writer.flush()?;

        Ok(id)
    }

    fn get(&self, id: &RecipeId) -> Result<Option<Recipe>> {
        self.find_by(|r| &r.id == id)
    }

    fn exists_by_url(&self, url: &str) -> Result<bool> {
        Ok(self.find_by_url(url)?.is_some())
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
                // Search in both name and ingredients (consistent with SQLite)
                let name_lower = recipe.name.to_lowercase();
                let ingredients_text: String = recipe
                    .ingredients
                    .iter()
                    .map(|i| i.raw.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ");
                
                // Check if all terms match either in name or ingredients
                if terms.iter().all(|t| {
                    name_lower.contains(t) || ingredients_text.contains(t)
                }) {
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
            ingredients: ingredients
                .into_iter()
                .map(crate::models::recipe::Ingredient::from_raw)
                .collect(),
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
    fn search_finds_by_ingredients() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

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

        let results = db.search("pasta").unwrap();
        assert_eq!(results.len(), 1);

        // Both have no chocolate
        let results = db.search("chocolate").unwrap();
        assert_eq!(results.len(), 0);
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

    #[test]
    fn duplicate_by_url_not_inserted() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        let r1 = test_recipe("Recipe 1", "https://example.com/test");
        let r2 = test_recipe("Recipe 2", "https://example.com/test"); // Same URL, different name

        let id1 = db.insert(&r1).unwrap();
        let id2 = db.insert(&r2).unwrap();

        // Should return the same ID for duplicate URL
        assert_eq!(id1, id2);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn concurrent_inserts_safe() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempdir().unwrap();
        let db = Arc::new(FileRecipesDb::open(dir.path().to_path_buf()).unwrap());
        let mut handles = vec![];

        for i in 0..10 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let recipe = test_recipe(
                    &format!("Recipe {}", i),
                    &format!("https://example.com/r{}", i),
                );
                db_clone.insert(&recipe).unwrap()
            });
            handles.push(handle);
        }

        let ids: Vec<RecipeId> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All IDs should be unique
        let unique_ids: std::collections::HashSet<_> = ids.into_iter().collect();
        assert_eq!(unique_ids.len(), 10);

        // All recipes should be stored
        assert_eq!(db.count().unwrap(), 10);
    }

    #[test]
    fn concurrent_reads_safe() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempdir().unwrap();
        let db = Arc::new(FileRecipesDb::open(dir.path().to_path_buf()).unwrap());

        // Insert some recipes first
        for i in 0..5 {
            db.insert(&test_recipe(
                &format!("Recipe {}", i),
                &format!("https://example.com/r{}", i),
            ))
            .unwrap();
        }

        let mut handles = vec![];
        for _ in 0..10 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || db_clone.count().unwrap());
            handles.push(handle);
        }

        let counts: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All reads should return the same count
        assert!(counts.iter().all(|c| *c == 5));
    }
}

//! File-based recipe storage with append-only JSONL and search index.
//!
//! Provides CRUD operations for recipes with duplicate detection
//! and full-text search capabilities.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::models::recipe::{Recipe, RecipeId};

use super::error::Result;
use super::traits::RecipesStorage;

/// In-memory index for fast lookups.
#[derive(Debug, Default)]
struct SearchIndex {
    by_url: std::collections::HashMap<String, RecipeId>,
    by_content_hash: std::collections::HashMap<String, RecipeId>,
    by_name: Vec<(String, RecipeId)>,
}

/// File-based, append-only recipe storage with search index.
#[derive(Debug)]
pub struct FileRecipesDb {
    dir: PathBuf,
    index: Arc<RwLock<SearchIndex>>,
}

impl FileRecipesDb {
    /// Open the recipes database at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        let db = Self {
            dir,
            index: Arc::new(RwLock::new(SearchIndex::default())),
        };
        db.rebuild_index()?;
        Ok(db)
    }

    /// Get the path to the recipes JSONL file.
    fn recipes_file(&self) -> PathBuf {
        self.dir.join("recipes.jsonl")
    }

    /// Get the path to the index file.
    fn index_file(&self) -> PathBuf {
        self.dir.join("index.json")
    }

    /// Check if content hash already exists.
    pub fn exists_by_content_hash(&self, hash: &str) -> Result<bool> {
        let index = self.index.read().unwrap();
        Ok(index.by_content_hash.contains_key(hash))
    }

    /// Save index to disk.
    fn save_index(&self) -> Result<()> {
        let index = self.index.read().unwrap();
        let persisted = PersistedIndex {
            urls: index.by_url.clone(),
            hashes: index.by_content_hash.clone(),
            names: index.by_name.clone(),
        };
        let file = File::create(self.index_file())?;
        serde_json::to_writer(file, &persisted)?;
        Ok(())
    }

    /// Load index from disk.
    #[allow(dead_code)]
    fn load_index(&self) -> Result<()> {
        if !self.index_file().exists() {
            return Ok(());
        }

        let file = File::open(self.index_file())?;
        let persisted: PersistedIndex = serde_json::from_reader(file)?;

        let mut index = self.index.write().unwrap();
        index.by_url = persisted.urls;
        index.by_content_hash = persisted.hashes;
        index.by_name = persisted.names;

        Ok(())
    }
}

impl RecipesStorage for FileRecipesDb {
    fn insert(&self, recipe: &Recipe) -> Result<RecipeId> {
        // Check for duplicates by content hash
        if let Some(hash) = &recipe.content_hash
            && self.exists_by_content_hash(hash)?
        {
            // Return existing ID
            let index = self.index.read().unwrap();
            if let Some(id) = index.by_content_hash.get(hash) {
                return Ok(id.clone());
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

        // Update index
        {
            let mut index = self.index.write().unwrap();
            index
                .by_url
                .insert(recipe.source_url.to_string(), id.clone());
            if let Some(hash) = &recipe.content_hash {
                index.by_content_hash.insert(hash.clone(), id.clone());
            }
            index
                .by_name
                .push((recipe.name.to_lowercase(), id.clone()));
        }

        // Persist index
        self.save_index()?;

        Ok(id)
    }

    fn get(&self, id: &RecipeId) -> Result<Option<Recipe>> {
        let file = File::open(self.recipes_file())?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Ok(recipe) = serde_json::from_str::<Recipe>(&line)
                && &recipe.id == id
            {
                return Ok(Some(recipe));
            }
        }

        Ok(None)
    }

    fn exists_by_url(&self, url: &str) -> Result<bool> {
        let index = self.index.read().unwrap();
        Ok(index.by_url.contains_key(url))
    }

    fn search(&self, query: &str) -> Result<Vec<RecipeId>> {
        let index = self.index.read().unwrap();
        let query = query.to_lowercase();
        let terms: Vec<&str> = query.split_whitespace().collect();

        let mut results: Vec<RecipeId> = index
            .by_name
            .iter()
            .filter(|(name, _)| terms.iter().all(|t| name.contains(t)))
            .map(|(_, id)| id.clone())
            .collect();

        // Deduplicate
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results.dedup_by(|a, b| a.0 == b.0);

        Ok(results)
    }

    fn rebuild_index(&self) -> Result<()> {
        let mut index = SearchIndex::default();

        if let Ok(file) = File::open(self.recipes_file()) {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if let Ok(recipe) = serde_json::from_str::<Recipe>(&line) {
                    index
                        .by_url
                        .insert(recipe.source_url.to_string(), recipe.id.clone());
                    if let Some(hash) = &recipe.content_hash {
                        index.by_content_hash.insert(hash.clone(), recipe.id.clone());
                    }
                    index
                        .by_name
                        .push((recipe.name.to_lowercase(), recipe.id.clone()));
                }
            }
        }

        *self.index.write().unwrap() = index;
        self.save_index()?;

        Ok(())
    }

    fn count(&self) -> Result<u64> {
        let index = self.index.read().unwrap();
        Ok(index.by_url.len() as u64)
    }

    fn all(&self) -> Result<Vec<Recipe>> {
        let mut recipes = Vec::new();
        if let Ok(file) = File::open(self.recipes_file()) {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if let Ok(recipe) = serde_json::from_str::<Recipe>(&line) {
                    recipes.push(recipe);
                }
            }
        }
        Ok(recipes)
    }
}

/// Persisted index structure.
#[derive(Debug, Serialize, Deserialize)]
struct PersistedIndex {
    urls: std::collections::HashMap<String, RecipeId>,
    hashes: std::collections::HashMap<String, RecipeId>,
    names: Vec<(String, RecipeId)>,
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
    fn rebuild_index_restores_state() {
        let dir = tempdir().unwrap();
        let db = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        let recipe = test_recipe("Test", "https://example.com/test");
        db.insert(&recipe).unwrap();

        // Open a new instance
        let db2 = FileRecipesDb::open(dir.path().to_path_buf()).unwrap();

        assert!(db2.exists_by_url("https://example.com/test").unwrap());
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

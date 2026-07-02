//! SQLite-based recipe storage.
//!
//! Provides recipe persistence using SQLite for efficient querying
//! and full-text search capabilities.

use rusqlite::{Connection, params, OptionalExtension, Transaction};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::models::recipe::{Recipe, RecipeId};

use super::error::Result;
use super::traits::RecipesStorage;

/// SQLite-based recipe storage.
#[derive(Debug)]
pub struct SqliteRecipesDb {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteRecipesDb {
    /// Open the recipes database at the given path.
    pub fn open(path: PathBuf) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.init_tables()?;
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.init_tables()?;
        Ok(db)
    }

    /// Initialize database tables.
    fn init_tables(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_url TEXT UNIQUE NOT NULL,
                source_domain TEXT NOT NULL,
                ingredients_json TEXT NOT NULL,
                instructions_json TEXT NOT NULL,
                prep_time_minutes INTEGER,
                cook_time_minutes INTEGER,
                total_time_minutes INTEGER,
                servings_json TEXT,
                cuisine TEXT,
                difficulty TEXT,
                tags_json TEXT NOT NULL,
                nutrition_json TEXT,
                image_url TEXT,
                author TEXT,
                description TEXT,
                scraped_at TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                meta_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_recipes_url ON recipes(source_url);
            CREATE INDEX IF NOT EXISTS idx_recipes_hash ON recipes(content_hash);
            CREATE INDEX IF NOT EXISTS idx_recipes_domain ON recipes(source_domain);
            CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(name);

            -- FTS5 virtual table for full-text search (standalone, no content table)
            CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5(
                id UNINDEXED,
                name,
                ingredients_text
            );
            "#,
        )?;

        Ok(())
    }

    /// Serialize recipe ingredients to text for FTS.
    fn ingredients_to_text(ingredients: &[crate::models::recipe::Ingredient]) -> String {
        ingredients.iter().map(|i| i.raw.as_str()).collect::<Vec<_>>().join(" ")
    }

    /// Parse a Recipe from a database row.
    /// Uses explicit column names to avoid fragility with schema changes.
    fn parse_recipe_from_row(row: &rusqlite::Row) -> rusqlite::Result<Recipe> {
        Ok(Recipe {
            id: RecipeId::new(row.get::<_, String>("id")?),
            name: row.get::<_, String>("name")?,
            source_url: row.get::<_, String>("source_url")?.parse().map_err(|_| rusqlite::Error::InvalidQuery)?,
            source_domain: row.get::<_, String>("source_domain")?,
            ingredients: serde_json::from_str(&row.get::<_, String>("ingredients_json")?).map_err(|_| rusqlite::Error::InvalidQuery)?,
            instructions: serde_json::from_str(&row.get::<_, String>("instructions_json")?).map_err(|_| rusqlite::Error::InvalidQuery)?,
            prep_time_minutes: row.get::<_, Option<i32>>("prep_time_minutes")?.map(|v| v as u32),
            cook_time_minutes: row.get::<_, Option<i32>>("cook_time_minutes")?.map(|v| v as u32),
            total_time_minutes: row.get::<_, Option<i32>>("total_time_minutes")?.map(|v| v as u32),
            servings: row.get::<_, Option<String>>("servings_json")?.map(|s| serde_json::from_str(&s)).transpose().map_err(|_| rusqlite::Error::InvalidQuery)?,
            cuisine: row.get::<_, Option<String>>("cuisine")?,
            difficulty: row.get::<_, Option<String>>("difficulty")?.map(|s| serde_json::from_str(&s)).transpose().map_err(|_| rusqlite::Error::InvalidQuery)?,
            tags: serde_json::from_str(&row.get::<_, String>("tags_json")?).map_err(|_| rusqlite::Error::InvalidQuery)?,
            nutrition: row.get::<_, Option<String>>("nutrition_json")?.map(|s| serde_json::from_str(&s)).transpose().map_err(|_| rusqlite::Error::InvalidQuery)?,
            image_url: row.get::<_, Option<String>>("image_url")?.map(|s| s.parse()).transpose().map_err(|_| rusqlite::Error::InvalidQuery)?,
            author: row.get::<_, Option<String>>("author")?,
            description: row.get::<_, Option<String>>("description")?,
            scraped_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>("scraped_at")?).map(|dt| dt.with_timezone(&chrono::Utc)).map_err(|_| rusqlite::Error::InvalidQuery)?,
            content_hash: row.get::<_, Option<String>>("content_hash")?,
            meta: serde_json::from_str(&row.get::<_, String>("meta_json")?).map_err(|_| rusqlite::Error::InvalidQuery)?,
        })
    }

    /// Insert a recipe within a transaction.
    fn insert_in_transaction(tx: &Transaction, recipe: &Recipe) -> Result<()> {
        tx.execute(
            r#"
            INSERT INTO recipes (
                id, name, source_url, source_domain, ingredients_json, instructions_json,
                prep_time_minutes, cook_time_minutes, total_time_minutes, servings_json,
                cuisine, difficulty, tags_json, nutrition_json, image_url, author,
                description, scraped_at, content_hash, meta_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)
            "#,
            params![
                recipe.id.0,
                recipe.name,
                recipe.source_url.to_string(),
                recipe.source_domain,
                serde_json::to_string(&recipe.ingredients)?,
                serde_json::to_string(&recipe.instructions)?,
                recipe.prep_time_minutes,
                recipe.cook_time_minutes,
                recipe.total_time_minutes,
                recipe.servings.as_ref().map(serde_json::to_string).transpose()?,
                recipe.cuisine,
                recipe.difficulty.map(|d| serde_json::to_string(&d)).transpose()?,
                serde_json::to_string(&recipe.tags)?,
                recipe.nutrition.as_ref().map(serde_json::to_string).transpose()?,
                recipe.image_url.as_ref().map(|u| u.to_string()),
                recipe.author,
                recipe.description,
                recipe.scraped_at.to_rfc3339(),
                recipe.content_hash,
                serde_json::to_string(&recipe.meta)?,
            ],
        )?;

        // Insert into FTS
        let ingredients_text = Self::ingredients_to_text(&recipe.ingredients);
        tx.execute(
            "INSERT INTO recipes_fts(id, name, ingredients_text) VALUES (?1, ?2, ?3)",
            params![recipe.id.0, recipe.name, ingredients_text],
        )?;

        Ok(())
    }

    /// Rebuild the full-text search index.
    ///
    /// This is SQLite-specific and not part of the trait.
    pub fn rebuild_fts_index(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("INSERT INTO recipes_fts(recipes_fts) VALUES ('rebuild')", [])?;
        Ok(())
    }

    /// Search with custom result limit.
    pub fn search_with_limit(&self, query: &str, limit: usize) -> Result<Vec<RecipeId>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id FROM recipes_fts WHERE recipes_fts MATCH ? ORDER BY rank LIMIT ?"
        )?;

        let ids = stmt
            .query_map(params![query, limit as i64], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .map(RecipeId::new)
            .collect();

        Ok(ids)
    }
}

impl RecipesStorage for SqliteRecipesDb {
    fn insert(&self, recipe: &Recipe) -> Result<RecipeId> {
        let mut conn = self.conn.lock().unwrap();
        
        // First check for duplicates by content hash or URL
        if let Some(hash) = &recipe.content_hash {
            let existing: Option<String> = conn
                .query_row(
                    "SELECT id FROM recipes WHERE content_hash = ?",
                    params![hash],
                    |row| row.get(0),
                )
                .optional()?;
            
            if let Some(id) = existing {
                return Ok(RecipeId::new(id));
            }
        }
        
        // Also check for duplicates by URL
        let url_str = recipe.source_url.to_string();
        let existing_url: Option<String> = conn
            .query_row(
                "SELECT id FROM recipes WHERE source_url = ?",
                params![url_str],
                |row| row.get(0),
            )
            .optional()?;
        
        if let Some(id) = existing_url {
            return Ok(RecipeId::new(id));
        }

        let id = RecipeId::generate();
        let mut recipe = recipe.clone();
        recipe.id = id.clone();

        // Compute content hash if not present
        if recipe.content_hash.is_none() {
            recipe.content_hash = Some(recipe.compute_hash());
        }

        // Use a transaction to ensure FTS and main table stay consistent
        let tx = conn.transaction()?;
        Self::insert_in_transaction(&tx, &recipe)?;
        tx.commit()?;

        Ok(id)
    }

    fn get(&self, id: &RecipeId) -> Result<Option<Recipe>> {
        let conn = self.conn.lock().unwrap();
        
        let result = conn
            .query_row(
                "SELECT * FROM recipes WHERE id = ?",
                params![id.0],
                Self::parse_recipe_from_row,
            )
            .optional()?;

        Ok(result)
    }

    fn exists_by_url(&self, url: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM recipes WHERE source_url = ?",
            params![url],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn search(&self, query: &str) -> Result<Vec<RecipeId>> {
        self.search_with_limit(query, 100)
    }

    fn count(&self) -> Result<u64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM recipes", [], |row| row.get(0))?;
        Ok(count as u64)
    }

    fn all(&self) -> Result<Vec<Recipe>> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT * FROM recipes"
        )?;

        let recipes = stmt
            .query_map([], Self::parse_recipe_from_row)?
            .filter_map(|r| r.ok())
            .collect();

        Ok(recipes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

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
    fn insert_and_get() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let recipe = test_recipe("Test Recipe", "https://example.com/test");
        let id = db.insert(&recipe).unwrap();
        
        let retrieved = db.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Test Recipe");
    }

    #[test]
    fn generates_unique_ids() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let r1 = test_recipe("Recipe 1", "https://example.com/r1");
        let r2 = test_recipe("Recipe 2", "https://example.com/r2");
        
        let id1 = db.insert(&r1).unwrap();
        let id2 = db.insert(&r2).unwrap();
        
        assert_ne!(id1, id2);
    }

    #[test]
    fn exists_by_url() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let recipe = test_recipe("Test", "https://example.com/test");
        db.insert(&recipe).unwrap();
        
        assert!(db.exists_by_url("https://example.com/test").unwrap());
        assert!(!db.exists_by_url("https://example.com/other").unwrap());
    }

    #[test]
    fn search_finds_matches() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let r1 = test_recipe("Chocolate Cake", "https://example.com/cake");
        let r2 = test_recipe("Apple Pie", "https://example.com/pie");
        
        db.insert(&r1).unwrap();
        db.insert(&r2).unwrap();
        
        let results = db.search("chocolate").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn duplicate_by_hash_not_inserted() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let mut r1 = test_recipe("Recipe 1", "https://example.com/r1");
        r1.content_hash = Some("hash123".to_string());
        
        let mut r2 = test_recipe("Recipe 2", "https://example.com/r2");
        r2.content_hash = Some("hash123".to_string());
        
        let id1 = db.insert(&r1).unwrap();
        let id2 = db.insert(&r2).unwrap();
        
        assert_eq!(id1, id2);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn count_works() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        assert_eq!(db.count().unwrap(), 0);
        
        db.insert(&test_recipe("R1", "https://example.com/r1")).unwrap();
        db.insert(&test_recipe("R2", "https://example.com/r2")).unwrap();
        
        assert_eq!(db.count().unwrap(), 2);
    }

    #[test]
    fn duplicate_by_url_not_inserted() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let r1 = test_recipe("Recipe 1", "https://example.com/test");
        let r2 = test_recipe("Recipe 2", "https://example.com/test"); // Same URL, different name
        
        let id1 = db.insert(&r1).unwrap();
        let id2 = db.insert(&r2).unwrap();
        
        // Should return the same ID for duplicate URL
        assert_eq!(id1, id2);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn search_finds_by_ingredients() {
        let db = SqliteRecipesDb::open_in_memory().unwrap();
        
        let r1 = test_recipe_with_ingredients(
            "Chocolate Cake",
            "https://example.com/cake",
            vec!["chocolate", "flour", "sugar"],
        );
        let r2 = test_recipe_with_ingredients(
            "Apple Pie",
            "https://example.com/pie",
            vec!["apples", "flour", "butter"],
        );
        
        db.insert(&r1).unwrap();
        db.insert(&r2).unwrap();
        
        // Search for ingredient
        let results = db.search("chocolate").unwrap();
        assert_eq!(results.len(), 1);
        
        let results = db.search("flour").unwrap();
        assert_eq!(results.len(), 2); // Both recipes have flour
    }

    #[test]
    fn concurrent_inserts_safe() {
        use std::thread;
        
        let db = Arc::new(SqliteRecipesDb::open_in_memory().unwrap());
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
        use std::thread;
        
        let db = Arc::new(SqliteRecipesDb::open_in_memory().unwrap());
        
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
            let handle = thread::spawn(move || {
                db_clone.count().unwrap()
            });
            handles.push(handle);
        }
        
        let counts: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // All reads should return the same count
        assert!(counts.iter().all(|c| *c == 5));
    }
}

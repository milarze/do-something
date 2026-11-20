//! Save a parsed recipe to the database.
//!
//! [`SaveRecipeTool`] validates a recipe, rejects duplicates by URL, computes
//! a content hash, and persists via the [`RecipesStorage`] trait so tests can
//! inject a fake store.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::domain::recipe::models::recipe::Recipe;
use crate::domain::recipe::storage::traits::RecipesStorage;
use crate::framework::tools::traits::{Tool, ToolResult};

/// Persist a parsed recipe to the database.
pub struct SaveRecipeTool {
    db: Arc<dyn RecipesStorage>,
}

impl SaveRecipeTool {
    /// Create a tool backed by any [`RecipesStorage`] implementation.
    pub fn new(db: Arc<dyn RecipesStorage>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SaveRecipeTool {
    fn name(&self) -> &str {
        "save_recipe"
    }

    fn description(&self) -> &str {
        "Save a parsed recipe to the database. Returns recipe ID."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "recipe": {
                    "type": "object",
                    "description": "Recipe object as returned by parse_recipe."
                }
            },
            "required": ["recipe"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let Some(recipe_value) = args.get("recipe") else {
            return ToolResult::error("missing `recipe`");
        };

        let mut recipe: Recipe = match serde_json::from_value(recipe_value.clone()) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("invalid recipe: {e}")),
        };

        // Validate required fields.
        if recipe.name.trim().is_empty() {
            return ToolResult::error("recipe must have a name");
        }

        // Reject duplicates by source URL.
        match self.db.exists_by_url(recipe.source_url.as_str()) {
            Ok(true) => return ToolResult::error("recipe already exists"),
            Ok(false) => {}
            Err(e) => return ToolResult::error(format!("duplicate check failed: {e}")),
        }

        // Compute the content hash if not already present.
        if recipe.content_hash.is_none() {
            recipe.content_hash = Some(recipe.compute_hash());
        }

        match self.db.insert(&recipe) {
            Ok(id) => ToolResult::success(json!({
                "recipe_id": id.to_string(),
                "name": recipe.name,
            })),
            Err(e) => ToolResult::error(format!("save failed: {e}")),
        }
    }
}

/// Minimal in-memory recipe store for unit tests.
#[cfg(test)]
#[derive(Default)]
struct FakeRecipeDb {
    recipes: std::sync::Mutex<Vec<Recipe>>,
    urls: std::sync::Mutex<std::collections::HashSet<String>>,
}

#[cfg(test)]
impl RecipesStorage for FakeRecipeDb {
    fn insert(&self, recipe: &Recipe) -> crate::framework::storage::Result<crate::domain::recipe::models::recipe::RecipeId> {
        use crate::domain::recipe::models::recipe::RecipeId;
        let mut recipes = self.recipes.lock().unwrap();
        let id = RecipeId::generate();
        let mut stored = recipe.clone();
        stored.id = id.clone();
        let url = stored.source_url.to_string();
        self.urls.lock().unwrap().insert(url);
        recipes.push(stored);
        let _ = id.0; // suppress unused warning path
        Ok(RecipeId::generate())
    }
    fn get(
        &self,
        _id: &crate::domain::recipe::models::recipe::RecipeId,
    ) -> crate::framework::storage::Result<Option<Recipe>> {
        Ok(None)
    }
    fn exists_by_url(&self, url: &str) -> crate::framework::storage::Result<bool> {
        Ok(self.urls.lock().unwrap().contains(url))
    }
    fn search(
        &self,
        _query: &str,
    ) -> crate::framework::storage::Result<Vec<crate::domain::recipe::models::recipe::RecipeId>> {
        Ok(Vec::new())
    }
    fn count(&self) -> crate::framework::storage::Result<u64> {
        Ok(self.recipes.lock().unwrap().len() as u64)
    }
    fn all(&self) -> crate::framework::storage::Result<Vec<Recipe>> {
        Ok(self.recipes.lock().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::tools::traits::Tool;
    use serde_json::json;
    use std::sync::Arc;

    fn make_recipe_json(name: &str, url: &str) -> Value {
        json!({
            "name": name,
            "source_url": url,
            "source_domain": "example.com",
            "ingredients": [{"raw": "flour"}],
            "instructions": ["mix"],
        })
    }

    fn make_db() -> Arc<FakeRecipeDb> {
        Arc::new(FakeRecipeDb::default())
    }

    #[tokio::test]
    async fn missing_recipe_arg_errors() {
        let tool = SaveRecipeTool::new(make_db());
        let res = tool.execute(json!({})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("missing `recipe`"));
    }

    #[tokio::test]
    async fn invalid_recipe_value_errors() {
        let tool = SaveRecipeTool::new(make_db());
        let res = tool.execute(json!({"recipe": "not an object"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("invalid recipe"));
    }

    #[tokio::test]
    async fn rejects_empty_name() {
        let tool = SaveRecipeTool::new(make_db());
        let recipe = json!({
            "name": "   ",
            "source_url": "https://example.com/r",
            "source_domain": "example.com",
        });
        let res = tool.execute(json!({"recipe": recipe})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("recipe must have a name"));
    }

    #[tokio::test]
    async fn rejects_duplicate_by_url() {
        let db = make_db();
        // Pre-populate the URL set so exists_by_url returns true.
        db.urls
            .lock()
            .unwrap()
            .insert("https://example.com/dup".to_string());
        let tool = SaveRecipeTool::new(db);
        let res = tool
            .execute(json!({"recipe": make_recipe_json("Dup", "https://example.com/dup")}))
            .await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("recipe already exists"));
    }

    #[tokio::test]
    async fn saves_valid_recipe_and_returns_id() {
        let db = make_db();
        let tool = SaveRecipeTool::new(db.clone());
        let res = tool
            .execute(json!({"recipe": make_recipe_json("Cake", "https://example.com/cake")}))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success");
        };
        assert_eq!(v["name"], "Cake");
        assert!(v["recipe_id"].as_str().unwrap().starts_with("rc_"));
        assert_eq!(db.count().unwrap(), 1);
    }

    #[tokio::test]
    async fn computes_content_hash_when_missing() {
        let db = make_db();
        let tool = SaveRecipeTool::new(db.clone());
        tool.execute(json!({"recipe": make_recipe_json("Pie", "https://example.com/pie")}))
            .await;
        let stored = &db.recipes.lock().unwrap()[0];
        assert!(stored.content_hash.is_some());
    }

    #[tokio::test]
    async fn preserves_existing_content_hash() {
        let db = make_db();
        let tool = SaveRecipeTool::new(db.clone());
        let mut recipe = make_recipe_json("Tart", "https://example.com/tart");
        recipe["content_hash"] = json!("preexisting");
        tool.execute(json!({"recipe": recipe})).await;
        let stored = &db.recipes.lock().unwrap()[0];
        assert_eq!(stored.content_hash.as_deref(), Some("preexisting"));
    }

    #[tokio::test]
    #[allow(unreachable_patterns)]
    async fn db_error_surfaces_as_tool_error() {
        // A RecipesStorage implementation that always fails exists_by_url.
        struct ErrorDb;
        impl RecipesStorage for ErrorDb {
            fn insert(
                &self,
                _r: &Recipe,
            ) -> crate::framework::storage::Result<crate::domain::recipe::models::recipe::RecipeId> {
                Ok(crate::domain::recipe::models::recipe::RecipeId::new("x"))
            }
            fn get(
                &self,
                _id: &crate::domain::recipe::models::recipe::RecipeId,
            ) -> crate::framework::storage::Result<Option<Recipe>> {
                Ok(None)
            }
            fn exists_by_url(&self, _url: &str) -> crate::framework::storage::Result<bool> {
                Err(crate::framework::storage::StorageError::LockError("simulated".to_string()))
            }
            fn search(
                &self,
                _q: &str,
            ) -> crate::framework::storage::Result<Vec<crate::domain::recipe::models::recipe::RecipeId>> {
                Ok(Vec::new())
            }
            fn count(&self) -> crate::framework::storage::Result<u64> {
                Ok(0)
            }
            fn all(&self) -> crate::framework::storage::Result<Vec<Recipe>> {
                Ok(Vec::new())
            }
        }
        let tool = SaveRecipeTool::new(Arc::new(ErrorDb));
        let res = tool
            .execute(json!({"recipe": make_recipe_json("X", "https://example.com/x")}))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("duplicate check failed"));
    }

    // Ensure the Recipe deserialize shape used by tool args round-trips.
    #[test]
    fn recipe_deserializes_from_tool_args() {
        let r: Recipe =
            serde_json::from_value(make_recipe_json("Y", "https://example.com/y")).unwrap();
        assert_eq!(r.name, "Y");
    }
}

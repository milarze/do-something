//! Recipe domain storage traits.
//!
//! These traits name domain types and therefore live in the domain,
//! not the framework. Implementations wrap framework primitives.

use chrono::NaiveDate;

use super::super::models::{
    Recipe, RecipeId, patterns::RecipePatterns, signal::RecipeSignal, site_config::SiteConfig,
    user_model::UserModel,
};

/// Recipe storage interface.
pub trait RecipesStorage: Send + Sync {
    /// Insert a recipe and return its ID.
    fn insert(&self, recipe: &Recipe) -> crate::framework::storage::Result<RecipeId>;

    /// Get a recipe by ID.
    fn get(&self, id: &RecipeId) -> crate::framework::storage::Result<Option<Recipe>>;

    /// Check if a URL has already been scraped.
    fn exists_by_url(&self, url: &str) -> crate::framework::storage::Result<bool>;

    /// Full-text search across recipes.
    fn search(&self, query: &str) -> crate::framework::storage::Result<Vec<RecipeId>>;

    /// Count total recipes.
    fn count(&self) -> crate::framework::storage::Result<u64>;

    /// Get all recipes (for iteration/export).
    fn all(&self) -> crate::framework::storage::Result<Vec<Recipe>>;
}

/// Signal logging interface.
pub trait SignalStorage: Send + Sync {
    /// Append a signal to today's log.
    fn append(&self, signal: &RecipeSignal) -> crate::framework::storage::Result<()>;

    /// Read signals from a date range.
    fn read_range(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> crate::framework::storage::Result<Vec<RecipeSignal>>;

    /// Count signals for a domain in the last N days.
    fn count_for_domain(&self, domain: &str, days: u32) -> crate::framework::storage::Result<u64>;

    /// Prune signals older than retention period.
    fn prune(&self, older_than_days: u32) -> crate::framework::storage::Result<u64>;

    /// Get signals for a specific date.
    fn read_date(&self, date: NaiveDate) -> crate::framework::storage::Result<Vec<RecipeSignal>>;

    /// Get all available log dates.
    fn available_dates(&self) -> crate::framework::storage::Result<Vec<NaiveDate>>;
}

/// Knowledge storage interface.
pub trait KnowledgeStorage: Send + Sync {
    /// Get site configuration for a domain.
    fn get_site_config(
        &self,
        domain: &str,
    ) -> crate::framework::storage::Result<Option<SiteConfig>>;

    /// Save site configuration.
    fn save_site_config(&self, config: &SiteConfig) -> crate::framework::storage::Result<()>;

    /// List all configured domains.
    fn list_site_configs(&self) -> crate::framework::storage::Result<Vec<String>>;

    /// Delete site configuration for a domain.
    fn delete_site_config(&self, domain: &str) -> crate::framework::storage::Result<()>;

    /// Get user model for a user.
    fn get_user_model(&self, user_id: &str)
    -> crate::framework::storage::Result<Option<UserModel>>;

    /// Save user model.
    fn save_user_model(&self, model: &UserModel) -> crate::framework::storage::Result<()>;

    /// List all user IDs with models.
    fn list_user_models(&self) -> crate::framework::storage::Result<Vec<String>>;

    /// Get patterns.
    fn get_patterns(&self) -> crate::framework::storage::Result<RecipePatterns>;

    /// Save patterns.
    fn save_patterns(&self, patterns: &RecipePatterns) -> crate::framework::storage::Result<()>;
}

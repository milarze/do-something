//! Storage trait definitions.
//!
//! These traits define the interfaces for storage backends, allowing
//! for multiple implementations (file-based, database, cloud, etc.).

use crate::models::agent_state::{KnowledgeContext, SessionId, SessionState};
use crate::models::knowledge::{Patterns, SiteConfig, UserModel};
use crate::models::recipe::{Recipe, RecipeId};
use crate::models::signal::Signal;

use super::error::Result;
use chrono::NaiveDate;

/// Recipe storage interface.
pub trait RecipesStorage: Send + Sync {
    /// Insert a recipe and return its ID.
    fn insert(&self, recipe: &Recipe) -> Result<RecipeId>;

    /// Get a recipe by ID.
    fn get(&self, id: &RecipeId) -> Result<Option<Recipe>>;

    /// Check if a URL has already been scraped.
    fn exists_by_url(&self, url: &str) -> Result<bool>;

    /// Full-text search across recipes.
    fn search(&self, query: &str) -> Result<Vec<RecipeId>>;

    /// Count total recipes.
    fn count(&self) -> Result<u64>;

    /// Get all recipes (for iteration/export).
    fn all(&self) -> Result<Vec<Recipe>>;
}

/// Signal logging interface.
pub trait SignalStorage: Send + Sync {
    /// Append a signal to today's log.
    fn append(&self, signal: &Signal) -> Result<()>;

    /// Read signals from a date range.
    fn read_range(&self, from: NaiveDate, to: NaiveDate) -> Result<Vec<Signal>>;

    /// Count signals for a domain in the last N days.
    fn count_for_domain(&self, domain: &str, days: u32) -> Result<u64>;

    /// Prune signals older than retention period.
    fn prune(&self, older_than_days: u32) -> Result<u64>;

    /// Get signals for a specific date.
    fn read_date(&self, date: NaiveDate) -> Result<Vec<Signal>>;

    /// Get all available log dates.
    fn available_dates(&self) -> Result<Vec<NaiveDate>>;
}

/// Knowledge storage interface.
pub trait KnowledgeStorage: Send + Sync {
    /// Get site configuration for a domain.
    fn get_site_config(&self, domain: &str) -> Result<Option<SiteConfig>>;

    /// Save site configuration.
    fn save_site_config(&self, config: &SiteConfig) -> Result<()>;

    /// List all configured domains.
    fn list_site_configs(&self) -> Result<Vec<String>>;

    /// Delete site configuration for a domain.
    fn delete_site_config(&self, domain: &str) -> Result<()>;

    /// Get user model for a user.
    fn get_user_model(&self, user_id: &str) -> Result<Option<UserModel>>;

    /// Save user model.
    fn save_user_model(&self, model: &UserModel) -> Result<()>;

    /// List all user IDs with models.
    fn list_user_models(&self) -> Result<Vec<String>>;

    /// Get patterns.
    fn get_patterns(&self) -> Result<Patterns>;

    /// Save patterns.
    fn save_patterns(&self, patterns: &Patterns) -> Result<()>;

    /// Load knowledge for context injection.
    fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext>;
}

/// Session storage interface.
pub trait SessionStorage: Send + Sync {
    /// Save session state.
    fn save(&self, session: &SessionState) -> Result<()>;

    /// Load session by ID.
    fn load(&self, id: &SessionId) -> Result<Option<SessionState>>;

    /// List all session IDs (sorted by modification time, newest first).
    fn list(&self) -> Result<Vec<SessionId>>;

    /// Delete a session.
    fn delete(&self, id: &SessionId) -> Result<()>;

    /// Check if a session exists.
    fn exists(&self, id: &SessionId) -> Result<bool>;

    /// Get the most recent session.
    fn latest(&self) -> Result<Option<SessionState>>;

    /// Count total sessions.
    fn count(&self) -> Result<usize>;

    /// Delete all sessions.
    fn clear(&self) -> Result<u64>;
}

/// Combined storage interface providing all storage components.
pub trait Storage: Send + Sync {
    type Recipes: RecipesStorage;
    type Signals: SignalStorage;
    type Knowledge: KnowledgeStorage;
    type Sessions: SessionStorage;

    fn recipes(&self) -> &Self::Recipes;
    fn signals(&self) -> &Self::Signals;
    fn knowledge(&self) -> &Self::Knowledge;
    fn sessions(&self) -> &Self::Sessions;
}

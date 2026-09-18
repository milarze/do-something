//! Recipe knowledge store built over framework JsonStore.
//!
//! Provides persistence for site configs, user models, and patterns.

use std::path::PathBuf;

use crate::framework::storage::JsonStore;

use super::super::models::{
    patterns::RecipePatterns, site_config::SiteConfig, user_model::UserModel,
};
use super::traits::KnowledgeStorage;

/// Default token budget for knowledge context.
const DEFAULT_TOKEN_BUDGET: u64 = 8000;

/// Knowledge context for prompt injection.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeContext {
    /// Site configs loaded for current domains.
    pub site_configs: Vec<String>, // JSON strings

    /// Relevant patterns.
    pub patterns: Vec<String>, // JSON strings

    /// User model summary.
    pub user_model: Option<String>, // JSON string

    /// Token budget allocated for knowledge.
    pub token_budget: u64,

    /// Estimated tokens used.
    pub estimated_tokens: u64,
}

impl KnowledgeContext {
    pub fn is_within_budget(&self) -> bool {
        self.estimated_tokens <= self.token_budget
    }
}

/// Persistence for site configs, user models, and patterns.
pub struct RecipeKnowledgeStore {
    configs: JsonStore<SiteConfig>,
    users: JsonStore<UserModel>,
    patterns: JsonStore<RecipePatterns>,
}

impl RecipeKnowledgeStore {
    /// Open the knowledge store at the given directory.
    pub fn open(dir: PathBuf) -> crate::framework::storage::Result<Self> {
        let configs = JsonStore::open(dir.join("site_configs"))?;
        let users = JsonStore::open(dir.join("user_models"))?;
        let patterns = JsonStore::open(dir.join("patterns"))?;
        Ok(Self {
            configs,
            users,
            patterns,
        })
    }

    /// Load knowledge for context injection.
    pub fn load_for_context(
        &self,
        domain: Option<&str>,
    ) -> crate::framework::storage::Result<KnowledgeContext> {
        let mut ctx = KnowledgeContext {
            token_budget: DEFAULT_TOKEN_BUDGET,
            ..Default::default()
        };

        // Load site config if domain specified
        if let Some(d) = domain
            && let Some(config) = self.configs.get(d)?
        {
            ctx.site_configs.push(serde_json::to_string(&config)?);
        }

        // Load patterns
        let patterns = self.patterns.load("patterns")?;
        if !patterns.success_patterns.is_empty() || !patterns.anti_patterns.is_empty() {
            ctx.patterns.push(serde_json::to_string(&patterns)?);
        }

        // Load default user model
        if let Some(model) = self.users.get("default")? {
            ctx.user_model = Some(serde_json::to_string(&model)?);
        }

        // Estimate tokens (rough: ~4 chars per token)
        let total_len: usize = ctx.site_configs.iter().map(|s| s.len()).sum::<usize>()
            + ctx.patterns.iter().map(|s| s.len()).sum::<usize>()
            + ctx.user_model.as_ref().map(|s| s.len()).unwrap_or(0);
        ctx.estimated_tokens = (total_len / 4) as u64;

        Ok(ctx)
    }
}

impl KnowledgeStorage for RecipeKnowledgeStore {
    fn get_site_config(
        &self,
        domain: &str,
    ) -> crate::framework::storage::Result<Option<SiteConfig>> {
        self.configs.get(domain)
    }

    fn save_site_config(&self, config: &SiteConfig) -> crate::framework::storage::Result<()> {
        self.configs.put(&config.domain, config)
    }

    fn list_site_configs(&self) -> crate::framework::storage::Result<Vec<String>> {
        self.configs.list()
    }

    fn delete_site_config(&self, domain: &str) -> crate::framework::storage::Result<()> {
        self.configs.delete(domain)
    }

    fn get_user_model(
        &self,
        user_id: &str,
    ) -> crate::framework::storage::Result<Option<UserModel>> {
        self.users.get(user_id)
    }

    fn save_user_model(&self, model: &UserModel) -> crate::framework::storage::Result<()> {
        self.users.put(&model.user_id, model)
    }

    fn list_user_models(&self) -> crate::framework::storage::Result<Vec<String>> {
        self.users.list()
    }

    fn get_patterns(&self) -> crate::framework::storage::Result<RecipePatterns> {
        self.patterns.load("patterns")
    }

    fn save_patterns(&self, patterns: &RecipePatterns) -> crate::framework::storage::Result<()> {
        self.patterns.save("patterns", patterns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::recipe::models::ParseMethod;
    use tempfile::tempdir;

    #[test]
    fn site_config_roundtrip() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let config = SiteConfig::new("test.com");
        store.save_site_config(&config).unwrap();

        let loaded = store.get_site_config("test.com").unwrap().unwrap();
        assert_eq!(loaded.domain, "test.com");
        assert_eq!(loaded.preferred_method, ParseMethod::SchemaOrg);
    }

    #[test]
    fn user_model_roundtrip() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let model = UserModel::default_user();
        store.save_user_model(&model).unwrap();

        let loaded = store.get_user_model("default").unwrap().unwrap();
        assert_eq!(loaded.user_id, "default");
    }

    #[test]
    fn patterns_roundtrip() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let mut patterns = RecipePatterns::default();
        patterns
            .success_patterns
            .push(crate::domain::recipe::models::patterns::SuccessPattern {
                description: "Test pattern".to_string(),
                sites: vec!["example.com".to_string()],
                method: ParseMethod::SchemaOrg,
                success_rate: 0.9,
                sample_size: 100,
                confidence: 0.85,
            });
        store.save_patterns(&patterns).unwrap();

        let loaded = store.get_patterns().unwrap();
        assert_eq!(loaded.success_patterns.len(), 1);
        assert_eq!(loaded.success_patterns[0].description, "Test pattern");
    }

    #[test]
    fn list_site_configs() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        store.save_site_config(&SiteConfig::new("a.com")).unwrap();
        store.save_site_config(&SiteConfig::new("b.com")).unwrap();
        store.save_site_config(&SiteConfig::new("c.com")).unwrap();

        let list = store.list_site_configs().unwrap();
        assert_eq!(list, vec!["a.com", "b.com", "c.com"]);
    }

    #[test]
    fn missing_config_returns_none() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let config = store.get_site_config("nonexistent.com").unwrap();
        assert!(config.is_none());
    }

    #[test]
    fn delete_site_config() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save_site_config(&SiteConfig::new("example.com"))
            .unwrap();
        assert!(store.get_site_config("example.com").unwrap().is_some());

        store.delete_site_config("example.com").unwrap();
        assert!(store.get_site_config("example.com").unwrap().is_none());
    }

    #[test]
    fn load_for_context_includes_relevant_data() {
        let dir = tempdir().unwrap();
        let store = RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save_site_config(&SiteConfig::new("example.com"))
            .unwrap();
        store.save_user_model(&UserModel::default_user()).unwrap();

        let ctx = store.load_for_context(Some("example.com")).unwrap();
        assert_eq!(ctx.site_configs.len(), 1);
        assert!(ctx.user_model.is_some());
    }
}

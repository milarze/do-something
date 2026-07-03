//! Main knowledge store with in-memory caching.
//!
//! Provides coordinated access to site configs, user models, and patterns
//! with automatic cache management.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use chrono::{DateTime, Utc};

use crate::models::knowledge::{Patterns, SiteConfig, UserModel};
use crate::storage::traits::KnowledgeStorage;
use crate::storage::StorageError;

use super::context::KnowledgeContextAssembler;
use super::patterns::PatternMatcher;
use super::site_config::SiteConfigManager;
use super::user_model::UserModelManager;

/// In-memory cache for knowledge data.
#[derive(Debug, Default)]
struct KnowledgeCache {
    /// Cached site configs by domain.
    site_configs: HashMap<String, SiteConfig>,
    
    /// Cached user models by user ID.
    user_models: HashMap<String, UserModel>,
    
    /// Cached patterns (single instance).
    patterns: Option<Patterns>,
    
    /// When cache was last refreshed.
    last_refresh: Option<DateTime<Utc>>,
}

/// Main knowledge store providing cached access to all knowledge.
///
/// Thread-safe through `RwLock`. Multiple readers can access simultaneously;
/// writers (updates) are rare and block briefly.
pub struct KnowledgeStore {
    /// Persistent storage backend.
    storage: Arc<dyn KnowledgeStorage>,
    
    /// In-memory cache with read-write lock.
    cache: RwLock<KnowledgeCache>,
}

impl std::fmt::Debug for KnowledgeStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeStore")
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}

impl KnowledgeStore {
    /// Create a new knowledge store with the given storage backend.
    pub fn new(storage: Arc<dyn KnowledgeStorage>) -> Self {
        Self {
            storage,
            cache: RwLock::new(KnowledgeCache::default()),
        }
    }

    /// Get site configuration for a domain.
    ///
    /// Returns from cache if available, otherwise loads from storage.
    /// Falls back to default config if not found.
    pub fn get_site_config(&self, domain: &str) -> crate::storage::Result<SiteConfig> {
        // Try read from cache first
        {
            let cache = self.cache.read().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            if let Some(config) = cache.site_configs.get(domain) {
                return Ok(config.clone());
            }
        }

        // Not in cache, load from storage
        let config = match self.storage.get_site_config(domain)? {
            Some(c) => c,
            None => SiteConfig::new(domain),
        };

        // Update cache
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            cache.site_configs.insert(domain.to_string(), config.clone());
        }

        Ok(config)
    }

    /// Update site configuration.
    ///
    /// Validates before saving. Updates cache on success.
    pub fn update_site_config(&self, config: &SiteConfig) -> crate::storage::Result<()> {
        // Validate
        SiteConfigManager::validate(config)?;
        
        // Persist
        self.storage.save_site_config(config)?;
        
        // Update cache
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            cache.site_configs.insert(config.domain.clone(), config.clone());
        }

        Ok(())
    }

    /// Get user model for a user ID.
    ///
    /// Creates default model if not exists.
    pub fn get_user_model(&self, user_id: &str) -> crate::storage::Result<UserModel> {
        // Try read from cache first
        {
            let cache = self.cache.read().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            if let Some(model) = cache.user_models.get(user_id) {
                return Ok(model.clone());
            }
        }

        // Not in cache, load from storage
        let model = match self.storage.get_user_model(user_id)? {
            Some(m) => m,
            None => UserModel::default_user(),
        };

        // Update cache
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            cache.user_models.insert(user_id.to_string(), model.clone());
        }

        Ok(model)
    }

    /// Update user model.
    ///
    /// Updates confidence based on sample size. Updates cache on success.
    pub fn update_user_model(&self, model: &UserModel) -> crate::storage::Result<()> {
        // Update confidence before saving
        let mut model = model.clone();
        UserModelManager::update_confidence(&mut model);
        
        // Persist
        self.storage.save_user_model(&model)?;
        
        // Update cache
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            cache.user_models.insert(model.user_id.clone(), model);
        }

        Ok(())
    }

    /// Get current patterns.
    ///
    /// Returns default if not exists.
    pub fn get_patterns(&self) -> crate::storage::Result<Patterns> {
        // Try read from cache first
        {
            let cache = self.cache.read().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            if let Some(ref patterns) = cache.patterns {
                return Ok(patterns.clone());
            }
        }

        // Not in cache, load from storage
        let patterns = self.storage.get_patterns()?;
        let patterns = if patterns.success_patterns.is_empty() && patterns.anti_patterns.is_empty() {
            // Storage returned empty, check for defaults
            let default_patterns = PatternMatcher::default_patterns();
            if !default_patterns.success_patterns.is_empty() || !default_patterns.anti_patterns.is_empty() {
                default_patterns
            } else {
                patterns
            }
        } else {
            patterns
        };

        // Update cache
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            cache.patterns = Some(patterns.clone());
        }

        Ok(patterns)
    }

    /// Update patterns.
    ///
    /// Updates cache on success.
    pub fn update_patterns(&self, patterns: &Patterns) -> crate::storage::Result<()> {
        // Persist
        self.storage.save_patterns(patterns)?;
        
        // Update cache
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            cache.patterns = Some(patterns.clone());
        }

        Ok(())
    }

    /// Load knowledge context for LLM prompt injection.
    ///
    /// Assembles relevant site config, patterns, and user preferences
    /// within the specified token budget.
    pub fn load_for_context(
        &self,
        domain: Option<&str>,
        token_budget: u64,
    ) -> crate::storage::Result<crate::models::agent_state::KnowledgeContext> {
        KnowledgeContextAssembler::build(self, domain, token_budget)
    }

    /// Invalidate all caches and reload from storage.
    pub fn refresh(&self) -> crate::storage::Result<()> {
        let mut cache = self.cache.write().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
        })?;
        
        cache.site_configs.clear();
        cache.user_models.clear();
        cache.patterns = None;
        cache.last_refresh = Some(Utc::now());
        
        Ok(())
    }

    /// Get the underlying storage backend (for advanced operations).
    pub fn storage(&self) -> &dyn KnowledgeStorage {
        self.storage.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ParseMethod;
    use tempfile::tempdir;

    // Helper to create a test knowledge store
    fn test_store() -> (KnowledgeStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let storage = Arc::new(crate::storage::FileKnowledgeStore::open(
            dir.path().join("knowledge"),
        ).unwrap());
        (KnowledgeStore::new(storage), dir)
    }

    #[test]
    fn get_site_config_returns_default_for_unknown() {
        let (store, _dir) = test_store();
        
        let config = store.get_site_config("unknown-site.com").unwrap();
        assert_eq!(config.domain, "unknown-site.com");
        assert_eq!(config.preferred_method, ParseMethod::SchemaOrg);
    }

    #[test]
    fn site_config_roundtrip() {
        let (store, _dir) = test_store();
        
        let mut config = SiteConfig::new("test.com");
        config.preferred_method = ParseMethod::Selectors;
        config.rate_limit_ms = 2000;
        
        store.update_site_config(&config).unwrap();
        
        let loaded = store.get_site_config("test.com").unwrap();
        assert_eq!(loaded.preferred_method, ParseMethod::Selectors);
        assert_eq!(loaded.rate_limit_ms, 2000);
    }

    #[test]
    fn get_user_model_returns_default() {
        let (store, _dir) = test_store();
        
        let model = store.get_user_model("default").unwrap();
        assert_eq!(model.user_id, "default");
        assert_eq!(model.confidence, 0.0);
    }

    #[test]
    fn user_model_roundtrip() {
        let (store, _dir) = test_store();
        
        let mut model = UserModel::default_user();
        model.max_prep_time_minutes = Some(30);
        model.preferred_difficulty = Some(crate::models::recipe::Difficulty::Easy);
        model.sample_size = 10;
        
        store.update_user_model(&model).unwrap();
        
        let loaded = store.get_user_model("default").unwrap();
        assert_eq!(loaded.max_prep_time_minutes, Some(30));
        assert_eq!(loaded.preferred_difficulty, Some(crate::models::recipe::Difficulty::Easy));
    }

    #[test]
    fn patterns_roundtrip() {
        let (store, _dir) = test_store();
        
        let mut patterns = Patterns::default();
        patterns.success_patterns.push(crate::models::knowledge::SuccessPattern {
            description: "Test pattern".to_string(),
            sites: vec!["example.com".to_string()],
            success_rate: 0.9,
            sample_size: 50,
            confidence: 0.85,
        });
        
        store.update_patterns(&patterns).unwrap();
        
        let loaded = store.get_patterns().unwrap();
        assert_eq!(loaded.success_patterns.len(), 1);
        assert_eq!(loaded.success_patterns[0].description, "Test pattern");
    }

    #[test]
    fn refresh_clears_cache() {
        let (store, _dir) = test_store();
        
        // Load something into cache
        let _ = store.get_site_config("example.com").unwrap();
        let _ = store.get_user_model("default").unwrap();
        
        // Refresh
        store.refresh().unwrap();
        
        // Cache should be cleared (verified by checking last_refresh)
        {
            let cache = store.cache.read().unwrap();
            assert!(cache.last_refresh.is_some());
            assert!(cache.site_configs.is_empty());
            assert!(cache.user_models.is_empty());
        }
    }

    #[test]
    fn load_for_context_respects_budget() {
        let (store, _dir) = test_store();
        
        let ctx = store.load_for_context(Some("example.com"), 1000).unwrap();
        assert!(ctx.token_budget <= 1000);
    }
}

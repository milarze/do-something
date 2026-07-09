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

/// Cache time-to-live in seconds.
///
/// After this duration, cached values will be considered stale
/// and reloaded from storage on next access.
///
/// A 5-minute TTL balances:
/// - Freshness: Updates from other processes/threads are picked up quickly
/// - Performance: Avoids excessive storage reads for frequently accessed data
/// - Memory: Allows periodic cache refresh to clear unused entries
const CACHE_TTL_SECONDS: i64 = 300; // 5 minutes

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

    /// Check if cache needs refresh due to TTL expiry.
    ///
    /// Returns true if cache has never been refreshed or if TTL has elapsed.
    fn is_cache_stale(&self) -> crate::storage::Result<bool> {
        let cache = self.cache.read().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
        })?;
        
        Ok(cache.last_refresh
            .map(|t| (Utc::now() - t).num_seconds() > CACHE_TTL_SECONDS)
            .unwrap_or(true))
    }

    /// Get site configuration for a domain.
    ///
    /// Returns from cache if available and fresh (within TTL),
    /// otherwise loads from storage.
    /// Falls back to default config if not found.
    pub fn get_site_config(&self, domain: &str) -> crate::storage::Result<SiteConfig> {
        // Check if we need to refresh due to TTL
        if self.is_cache_stale()? {
            self.refresh()?;
        }
        
        // Fast path: try read from cache first
        {
            let cache = self.cache.read().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            if let Some(config) = cache.site_configs.get(domain) {
                return Ok(config.clone());
            }
        }

        // Slow path: load from storage
        let config = match self.storage.get_site_config(domain)? {
            Some(c) => c,
            None => SiteConfig::new(domain),
        };

        // Update cache with write lock, re-checking to prevent race
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            // Another thread may have inserted while we were loading
            if let Some(existing) = cache.site_configs.get(domain) {
                return Ok(existing.clone());
            }
            
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
        // Fast path: try read from cache first
        {
            let cache = self.cache.read().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            if let Some(model) = cache.user_models.get(user_id) {
                return Ok(model.clone());
            }
        }

        // Slow path: load from storage
        let model = match self.storage.get_user_model(user_id)? {
            Some(m) => m,
            None => UserModelManager::create(user_id),
        };

        // Update cache with write lock, re-checking to prevent race
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            // Another thread may have inserted while we were loading
            if let Some(existing) = cache.user_models.get(user_id) {
                return Ok(existing.clone());
            }
            
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
        // Fast path: try read from cache first
        {
            let cache = self.cache.read().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            if let Some(ref patterns) = cache.patterns {
                return Ok(patterns.clone());
            }
        }

        // Slow path: load from storage
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

        // Update cache with write lock, re-checking to prevent race
        {
            let mut cache = self.cache.write().map_err(|e| {
                StorageError::Io(std::io::Error::other(format!("Cache lock error: {e}")))
            })?;
            
            // Another thread may have inserted while we were loading
            if let Some(ref existing) = cache.patterns {
                return Ok(existing.clone());
            }
            
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
    use crate::knowledge::mock::MockKnowledgeStorage;
    use crate::models::ParseMethod;
    use tempfile::tempdir;

    // Helper to create a test knowledge store with real storage
    fn test_store() -> (KnowledgeStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let storage = Arc::new(crate::storage::FileKnowledgeStore::open(
            dir.path().join("knowledge"),
        ).unwrap());
        (KnowledgeStore::new(storage), dir)
    }

    // Helper to create a test store with mock storage (faster, no filesystem)
    fn test_store_mock() -> KnowledgeStore {
        KnowledgeStore::new(Arc::new(MockKnowledgeStorage::new()))
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

    #[test]
    fn concurrent_cache_access_no_race() {
        use std::sync::Barrier;
        use std::thread;

        let (store, _dir) = test_store();
        let store = Arc::new(store);
        let barrier = Arc::new(Barrier::new(10));

        // First, save a config to storage (bypass cache)
        let config = SiteConfig::new("concurrent-test.com");
        store.storage.save_site_config(&config).unwrap();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    // All threads try to get the same config simultaneously
                    let result = store.get_site_config("concurrent-test.com").unwrap();
                    result
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same config
        assert!(results.iter().all(|r| r.domain == "concurrent-test.com"));
        
        // Verify cache only has one entry
        let cache = store.cache.read().unwrap();
        assert_eq!(cache.site_configs.len(), 1);
    }

    #[test]
    fn concurrent_user_model_access_no_race() {
        use std::sync::Barrier;
        use std::thread;

        let (store, _dir) = test_store();
        let store = Arc::new(store);
        let barrier = Arc::new(Barrier::new(5));

        let handles: Vec<_> = (0..5)
            .map(|_| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    store.get_user_model("test-user").unwrap()
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same model
        assert!(results.iter().all(|r| r.user_id == "test-user"));
    }

    #[test]
    fn cache_ttl_triggers_refresh() {
        let (store, _dir) = test_store();
        
        // Load a config into cache
        let _ = store.get_site_config("example.com").unwrap();
        
        // Verify it's cached
        {
            let cache = store.cache.read().unwrap();
            assert!(cache.site_configs.contains_key("example.com"));
            assert!(cache.last_refresh.is_some());
        }
        
        // Manually set last_refresh to simulate TTL expiry
        {
            let mut cache = store.cache.write().unwrap();
            // Set last_refresh to 10 minutes ago (beyond the 5-minute TTL)
            cache.last_refresh = Some(Utc::now() - chrono::Duration::seconds(600));
        }
        
        // Access should trigger refresh due to stale cache
        let is_stale = store.is_cache_stale().unwrap();
        assert!(is_stale, "Cache should be stale after TTL expiry");
    }

    #[test]
    fn fresh_cache_within_ttl() {
        let (store, _dir) = test_store();
        
        // Fresh store should have stale cache (never refreshed)
        assert!(store.is_cache_stale().unwrap());
        
        // Load something
        let _ = store.get_site_config("example.com").unwrap();
        
        // Cache should now be fresh
        assert!(!store.is_cache_stale().unwrap());
    }

    // Tests using mock storage (no filesystem dependency)
    #[test]
    fn mock_storage_site_config() {
        let store = test_store_mock();
        
        let config = SiteConfig::new("mock-test.com");
        store.update_site_config(&config).unwrap();
        
        let loaded = store.get_site_config("mock-test.com").unwrap();
        assert_eq!(loaded.domain, "mock-test.com");
    }

    #[test]
    fn mock_storage_user_model() {
        let store = test_store_mock();
        
        let mut model = UserModelManager::create("test-user");
        model.max_prep_time_minutes = Some(45);
        store.update_user_model(&model).unwrap();
        
        let loaded = store.get_user_model("test-user").unwrap();
        assert_eq!(loaded.user_id, "test-user");
        assert_eq!(loaded.max_prep_time_minutes, Some(45));
    }

    #[test]
    fn mock_storage_patterns() {
        let store = test_store_mock();
        
        let patterns = PatternMatcher::default_patterns();
        store.update_patterns(&patterns).unwrap();
        
        let loaded = store.get_patterns().unwrap();
        assert!(!loaded.success_patterns.is_empty());
    }

    #[test]
    fn mock_storage_concurrent_access() {
        use std::sync::Barrier;
        use std::thread;

        let store = Arc::new(test_store_mock());
        let barrier = Arc::new(Barrier::new(10));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let domain = format!("domain-{}.com", i % 3);
                    store.get_site_config(&domain).unwrap()
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}

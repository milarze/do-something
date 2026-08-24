//! Mock storage implementation for testing.
//!
//! Provides an in-memory mock that implements KnowledgeStorage trait,
//! allowing unit tests to run without filesystem dependencies.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::models::agent_state::KnowledgeContext;
use crate::models::knowledge::{Patterns, SiteConfig, UserModel};
use crate::storage::error::Result;
use crate::storage::traits::KnowledgeStorage;

/// In-memory mock storage for testing.
pub struct MockKnowledgeStorage {
    site_configs: Mutex<HashMap<String, SiteConfig>>,
    user_models: Mutex<HashMap<String, UserModel>>,
    patterns: Mutex<Option<Patterns>>,
}

impl MockKnowledgeStorage {
    /// Create a new empty mock storage.
    pub fn new() -> Self {
        Self {
            site_configs: Mutex::new(HashMap::new()),
            user_models: Mutex::new(HashMap::new()),
            patterns: Mutex::new(None),
        }
    }

    /// Create a mock storage pre-populated with test data.
    pub fn with_test_data() -> Self {
        let storage = Self::new();
        
        // Add a test site config
        let config = SiteConfig::new("test.example.com");
        storage.save_site_config(&config).ok();
        
        // Add a test user model
        let model = UserModel::default_user();
        storage.save_user_model(&model).ok();
        
        storage
    }

    /// Clear all stored data.
    pub fn clear(&self) {
        self.site_configs.lock().unwrap().clear();
        self.user_models.lock().unwrap().clear();
        *self.patterns.lock().unwrap() = None;
    }
}

impl Default for MockKnowledgeStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeStorage for MockKnowledgeStorage {
    fn get_site_config(&self, domain: &str) -> Result<Option<SiteConfig>> {
        Ok(self.site_configs.lock().unwrap().get(domain).cloned())
    }

    fn save_site_config(&self, config: &SiteConfig) -> Result<()> {
        self.site_configs
            .lock()
            .unwrap()
            .insert(config.domain.clone(), config.clone());
        Ok(())
    }

    fn list_site_configs(&self) -> Result<Vec<String>> {
        Ok(self.site_configs.lock().unwrap().keys().cloned().collect())
    }

    fn delete_site_config(&self, domain: &str) -> Result<()> {
        self.site_configs.lock().unwrap().remove(domain);
        Ok(())
    }

    fn get_user_model(&self, user_id: &str) -> Result<Option<UserModel>> {
        Ok(self.user_models.lock().unwrap().get(user_id).cloned())
    }

    fn save_user_model(&self, model: &UserModel) -> Result<()> {
        self.user_models
            .lock()
            .unwrap()
            .insert(model.user_id.clone(), model.clone());
        Ok(())
    }

    fn list_user_models(&self) -> Result<Vec<String>> {
        Ok(self.user_models.lock().unwrap().keys().cloned().collect())
    }

    fn get_patterns(&self) -> Result<Patterns> {
        Ok(self
            .patterns
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default())
    }

    fn save_patterns(&self, patterns: &Patterns) -> Result<()> {
        *self.patterns.lock().unwrap() = Some(patterns.clone());
        Ok(())
    }

    fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext> {
        let mut ctx = KnowledgeContext {
            token_budget: 8000,
            ..Default::default()
        };

        if let Some(d) = domain
            && let Some(config) = self.get_site_config(d)?
        {
            ctx.site_configs.push(serde_json::to_string(&config)?);
        }

        let patterns = self.get_patterns()?;
        if !patterns.success_patterns.is_empty() || !patterns.anti_patterns.is_empty() {
            ctx.patterns.push(serde_json::to_string(&patterns)?);
        }

        if let Some(model) = self.get_user_model("default")? {
            ctx.user_model = Some(serde_json::to_string(&model)?);
        }

        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_storage_roundtrip() {
        let storage = MockKnowledgeStorage::new();
        
        let config = SiteConfig::new("example.com");
        storage.save_site_config(&config).unwrap();
        
        let loaded = storage.get_site_config("example.com").unwrap().unwrap();
        assert_eq!(loaded.domain, "example.com");
    }

    #[test]
    fn mock_storage_missing_returns_none() {
        let storage = MockKnowledgeStorage::new();
        assert!(storage.get_site_config("nonexistent.com").unwrap().is_none());
    }

    #[test]
    fn mock_storage_delete() {
        let storage = MockKnowledgeStorage::new();
        
        let config = SiteConfig::new("example.com");
        storage.save_site_config(&config).unwrap();
        assert!(storage.get_site_config("example.com").unwrap().is_some());
        
        storage.delete_site_config("example.com").unwrap();
        assert!(storage.get_site_config("example.com").unwrap().is_none());
    }

    #[test]
    fn mock_storage_user_model() {
        let storage = MockKnowledgeStorage::new();
        
        let mut model = UserModel::default_user();
        model.max_prep_time_minutes = Some(30);
        storage.save_user_model(&model).unwrap();
        
        let loaded = storage.get_user_model("default").unwrap().unwrap();
        assert_eq!(loaded.max_prep_time_minutes, Some(30));
    }

    #[test]
    fn mock_storage_patterns() {
        let storage = MockKnowledgeStorage::new();
        
        let mut patterns = Patterns::default();
        patterns.success_patterns.push(crate::models::knowledge::SuccessPattern {
            description: "Test pattern".to_string(),
            sites: vec![],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.8,
        });
        
        storage.save_patterns(&patterns).unwrap();
        let loaded = storage.get_patterns().unwrap();
        assert_eq!(loaded.success_patterns.len(), 1);
    }

    #[test]
    fn mock_storage_clear() {
        let storage = MockKnowledgeStorage::with_test_data();
        
        assert!(!storage.list_site_configs().unwrap().is_empty());
        
        storage.clear();
        
        assert!(storage.list_site_configs().unwrap().is_empty());
        assert!(storage.list_user_models().unwrap().is_empty());
    }

    #[test]
    fn mock_storage_load_for_context() {
        let storage = MockKnowledgeStorage::new();
        
        let config = SiteConfig::new("test.example.com");
        storage.save_site_config(&config).unwrap();
        
        let ctx = storage.load_for_context(Some("test.example.com")).unwrap();
        assert_eq!(ctx.site_configs.len(), 1);
    }
}

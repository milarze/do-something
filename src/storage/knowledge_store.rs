//! File-based knowledge store for site configs, user models, and patterns.
//!
//! Provides persistence for learned configurations and patterns
//! that improve scraping over time.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use crate::models::agent_state::KnowledgeContext;
use crate::models::knowledge::{Patterns, SiteConfig, UserModel};

use super::error::Result;
use super::file_utils::validate_path_component;
use super::traits::KnowledgeStorage;

/// Default token budget for knowledge context.
const DEFAULT_TOKEN_BUDGET: u64 = 8000;

/// File-based persistence for site configs, user models, and patterns.
#[derive(Debug)]
pub struct FileKnowledgeStore {
    dir: PathBuf,
}

impl FileKnowledgeStore {
    /// Open the knowledge store at the given directory.
    pub fn open(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        fs::create_dir_all(dir.join("site_configs"))?;
        fs::create_dir_all(dir.join("user_models"))?;
        fs::create_dir_all(dir.join("patterns"))?;
        Ok(Self { dir })
    }

    /// Get the path for a site config file.
    ///
    /// Returns an error if the domain contains path traversal characters.
    fn site_config_path(&self, domain: &str) -> Result<PathBuf> {
        validate_path_component(domain, "domain")?;
        Ok(self.dir.join("site_configs").join(format!("{}.json", domain)))
    }

    /// Get the path for a user model file.
    ///
    /// Returns an error if the user ID contains path traversal characters.
    fn user_model_path(&self, user_id: &str) -> Result<PathBuf> {
        validate_path_component(user_id, "user_id")?;
        Ok(self.dir.join("user_models").join(format!("{}.json", user_id)))
    }

    /// Get the path for the patterns file.
    fn patterns_path(&self) -> PathBuf {
        self.dir.join("patterns").join("patterns.json")
    }

    /// Load JSON from a file, returning None if it doesn't exist.
    fn load_json<T: serde::de::DeserializeOwned>(&self, path: &PathBuf) -> Result<Option<T>> {
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let value = serde_json::from_reader(reader)?;
        Ok(Some(value))
    }

    /// Load JSON from a file, erroring if it doesn't exist.
    fn load_json_required<T: serde::de::DeserializeOwned>(&self, path: &PathBuf) -> Result<T> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let value = serde_json::from_reader(reader)?;
        Ok(value)
    }

    /// Save JSON to a file.
    fn save_json<T: serde::Serialize>(&self, path: &PathBuf, value: &T) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, value)?;
        Ok(())
    }
}

impl KnowledgeStorage for FileKnowledgeStore {
    fn get_site_config(&self, domain: &str) -> Result<Option<SiteConfig>> {
        let path = self.site_config_path(domain)?;
        if !path.exists() {
            // Try default config
            let default_path = self.site_config_path("_default")?;
            if default_path.exists() {
                return self.load_json(&default_path);
            }
            return Ok(None);
        }
        self.load_json(&path)
    }

    fn save_site_config(&self, config: &SiteConfig) -> Result<()> {
        let path = self.site_config_path(&config.domain)?;
        self.save_json(&path, config)
    }

    fn list_site_configs(&self) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let configs_dir = self.dir.join("site_configs");

        for entry in fs::read_dir(configs_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(stem) = path.file_stem()
            {
                let name = stem.to_string_lossy();
                if name != "_default" {
                    domains.push(name.to_string());
                }
            }
        }

        domains.sort();
        Ok(domains)
    }

    fn delete_site_config(&self, domain: &str) -> Result<()> {
        let path = self.site_config_path(domain)?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    fn get_user_model(&self, user_id: &str) -> Result<Option<UserModel>> {
        let path = self.user_model_path(user_id)?;
        if !path.exists() {
            return Ok(None);
        }
        self.load_json(&path)
    }

    fn save_user_model(&self, model: &UserModel) -> Result<()> {
        let path = self.user_model_path(&model.user_id)?;
        self.save_json(&path, model)
    }

    fn list_user_models(&self) -> Result<Vec<String>> {
        let mut users = Vec::new();
        let models_dir = self.dir.join("user_models");

        for entry in fs::read_dir(models_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(stem) = path.file_stem()
            {
                let name = stem.to_string_lossy();
                users.push(name.to_string());
            }
        }

        users.sort();
        Ok(users)
    }

    fn get_patterns(&self) -> Result<Patterns> {
        let path = self.patterns_path();
        if !path.exists() {
            return Ok(Patterns::default());
        }
        self.load_json_required(&path)
    }

    fn save_patterns(&self, patterns: &Patterns) -> Result<()> {
        let path = self.patterns_path();
        self.save_json(&path, patterns)
    }

    fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext> {
        let mut ctx = KnowledgeContext {
            token_budget: DEFAULT_TOKEN_BUDGET,
            ..Default::default()
        };

        // Load site config if domain specified
        if let Some(d) = domain
            && let Some(config) = self.get_site_config(d)?
        {
            ctx.site_configs.push(serde_json::to_string(&config)?);
        }

        // Load patterns
        let patterns = self.get_patterns()?;
        if !patterns.success_patterns.is_empty() || !patterns.anti_patterns.is_empty() {
            ctx.patterns.push(serde_json::to_string(&patterns)?);
        }

        // Load default user model
        if let Some(model) = self.get_user_model("default")? {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ParseMethod;
    use crate::StorageError;
    use tempfile::tempdir;

    #[test]
    fn site_config_roundtrip() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let config = SiteConfig::new("test.com");
        store.save_site_config(&config).unwrap();

        let loaded = store.get_site_config("test.com").unwrap().unwrap();
        assert_eq!(loaded.domain, "test.com");
        assert_eq!(loaded.preferred_method, ParseMethod::SchemaOrg);
    }

    #[test]
    fn user_model_roundtrip() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let model = UserModel::default_user();
        store.save_user_model(&model).unwrap();

        let loaded = store.get_user_model("default").unwrap().unwrap();
        assert_eq!(loaded.user_id, "default");
    }

    #[test]
    fn patterns_roundtrip() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let mut patterns = Patterns::default();
        patterns.success_patterns.push(crate::models::knowledge::SuccessPattern {
            description: "Test pattern".to_string(),
            sites: vec!["example.com".to_string()],
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
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        store.save_site_config(&SiteConfig::new("a.com")).unwrap();
        store.save_site_config(&SiteConfig::new("b.com")).unwrap();
        store.save_site_config(&SiteConfig::new("c.com")).unwrap();

        let list = store.list_site_configs().unwrap();
        assert_eq!(list, vec!["a.com", "b.com", "c.com"]);
    }

    #[test]
    fn missing_config_returns_none() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let config = store.get_site_config("nonexistent.com").unwrap();
        assert!(config.is_none());
    }

    #[test]
    fn load_for_context_includes_relevant_data() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save_site_config(&SiteConfig::new("example.com"))
            .unwrap();
        store.save_user_model(&UserModel::default_user()).unwrap();

        let ctx = store.load_for_context(Some("example.com")).unwrap();
        assert_eq!(ctx.site_configs.len(), 1);
        assert!(ctx.user_model.is_some());
    }

    #[test]
    fn delete_site_config() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        store
            .save_site_config(&SiteConfig::new("example.com"))
            .unwrap();
        assert!(store.get_site_config("example.com").unwrap().is_some());

        store.delete_site_config("example.com").unwrap();
        assert!(store.get_site_config("example.com").unwrap().is_none());
    }

    // Path traversal security tests
    #[test]
    fn rejects_path_traversal_in_domain() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get_site_config("../etc/passwd");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_path_traversal_in_user_id() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get_user_model("../../etc/shadow");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_empty_domain() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get_site_config("");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_slash_in_domain() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get_site_config("example.com/subdir");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }

    #[test]
    fn rejects_null_byte_in_domain() {
        let dir = tempdir().unwrap();
        let store = FileKnowledgeStore::open(dir.path().to_path_buf()).unwrap();

        let result = store.get_site_config("example\0.com");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::InvalidPath(_)));
    }
}

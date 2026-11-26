//! Retrieve a knowledge snapshot.
//!
//! [`GetKnowledgeTool`] reads site configs, the default user model, and
//! patterns through the [`KnowledgeStorage`] trait, returning a JSON
//! snapshot the agent consults before making changes.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::domain::recipe::storage::KnowledgeStorage;
use crate::framework::tools::traits::{Tool, ToolResult};

/// Retrieve current knowledge for a domain.
pub struct GetKnowledgeTool {
    knowledge: Arc<dyn KnowledgeStorage>,
}

impl GetKnowledgeTool {
    /// Create a tool backed by any [`KnowledgeStorage`] implementation.
    pub fn new(knowledge: Arc<dyn KnowledgeStorage>) -> Self {
        Self { knowledge }
    }

    /// Load site config JSON values for `domain` (or all domains if `None`).
    fn load_configs(&self, domain: Option<&str>) -> crate::framework::storage::Result<Vec<Value>> {
        match domain {
            Some(d) => {
                let mut out = Vec::new();
                if let Some(config) = self.knowledge.get_site_config(d)? {
                    out.push(serde_json::to_value(&config)?);
                }
                Ok(out)
            }
            None => {
                let domains = self.knowledge.list_site_configs()?;
                let mut out = Vec::new();
                for d in domains {
                    if let Some(config) = self.knowledge.get_site_config(&d)? {
                        out.push(serde_json::to_value(&config)?);
                    }
                }
                Ok(out)
            }
        }
    }
}

#[async_trait]
impl Tool for GetKnowledgeTool {
    fn name(&self) -> &str {
        "get_knowledge"
    }

    fn description(&self) -> &str {
        "Retrieve current knowledge for a domain. \
         Use to understand what's known before making changes."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter. Returns all if omitted."
                },
                "include_patterns": {
                    "type": "boolean",
                    "description": "Include success/anti-patterns (default: true)"
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let domain = args
            .get("domain")
            .and_then(Value::as_str)
            .map(str::to_string);
        let include_patterns = args
            .get("include_patterns")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        let configs = match self.load_configs(domain.as_deref()) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("load configs failed: {e}")),
        };

        let user_model = match self.knowledge.get_user_model("default") {
            Ok(Some(m)) => serde_json::to_value(&m).ok(),
            Ok(None) => None,
            Err(e) => return ToolResult::error(format!("load user model failed: {e}")),
        };

        let mut result = json!({
            "site_configs": configs,
            "user_model": user_model,
        });

        if include_patterns {
            match self.knowledge.get_patterns() {
                Ok(p) => {
                    result["patterns"] = serde_json::to_value(&p).unwrap_or(json!(null));
                }
                Err(e) => return ToolResult::error(format!("load patterns failed: {e}")),
            }
        }

        ToolResult::success(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::recipe::models::ParseMethod;
    use crate::domain::recipe::models::patterns::{AntiPattern, AntiPatternAction, RecipePatterns};
    use crate::domain::recipe::models::site_config::{Selectors, SiteConfig};
    use crate::domain::recipe::models::user_model::UserModel;
    use crate::framework::storage::StorageError;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// In-memory knowledge store for unit tests.
    #[derive(Default)]
    struct FakeKnowledgeStore {
        configs: Mutex<HashMap<String, SiteConfig>>,
        patterns: Mutex<RecipePatterns>,
        users: Mutex<HashMap<String, UserModel>>,
    }

    impl KnowledgeStorage for FakeKnowledgeStore {
        fn get_site_config(
            &self,
            domain: &str,
        ) -> crate::framework::storage::Result<Option<SiteConfig>> {
            Ok(self.configs.lock().unwrap().get(domain).cloned())
        }
        fn save_site_config(&self, config: &SiteConfig) -> crate::framework::storage::Result<()> {
            self.configs
                .lock()
                .unwrap()
                .insert(config.domain.clone(), config.clone());
            Ok(())
        }
        fn list_site_configs(&self) -> crate::framework::storage::Result<Vec<String>> {
            let mut keys: Vec<_> = self.configs.lock().unwrap().keys().cloned().collect();
            keys.sort();
            Ok(keys)
        }
        fn delete_site_config(&self, domain: &str) -> crate::framework::storage::Result<()> {
            self.configs.lock().unwrap().remove(domain);
            Ok(())
        }
        fn get_user_model(
            &self,
            user_id: &str,
        ) -> crate::framework::storage::Result<Option<UserModel>> {
            Ok(self.users.lock().unwrap().get(user_id).cloned())
        }
        fn save_user_model(&self, model: &UserModel) -> crate::framework::storage::Result<()> {
            self.users
                .lock()
                .unwrap()
                .insert(model.user_id.clone(), model.clone());
            Ok(())
        }
        fn list_user_models(&self) -> crate::framework::storage::Result<Vec<String>> {
            let mut keys: Vec<_> = self.users.lock().unwrap().keys().cloned().collect();
            keys.sort();
            Ok(keys)
        }
        fn get_patterns(&self) -> crate::framework::storage::Result<RecipePatterns> {
            Ok(self.patterns.lock().unwrap().clone())
        }
        fn save_patterns(
            &self,
            patterns: &RecipePatterns,
        ) -> crate::framework::storage::Result<()> {
            *self.patterns.lock().unwrap() = patterns.clone();
            Ok(())
        }
    }

    /// A store whose `get_site_config` always fails.
    struct FailingKnowledgeStore;

    impl KnowledgeStorage for FailingKnowledgeStore {
        fn get_site_config(
            &self,
            _domain: &str,
        ) -> crate::framework::storage::Result<Option<SiteConfig>> {
            Err(StorageError::LockError("simulated".to_string()))
        }
        fn save_site_config(&self, _config: &SiteConfig) -> crate::framework::storage::Result<()> {
            Ok(())
        }
        fn list_site_configs(&self) -> crate::framework::storage::Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn delete_site_config(&self, _domain: &str) -> crate::framework::storage::Result<()> {
            Ok(())
        }
        fn get_user_model(
            &self,
            _user_id: &str,
        ) -> crate::framework::storage::Result<Option<UserModel>> {
            Ok(None)
        }
        fn save_user_model(&self, _model: &UserModel) -> crate::framework::storage::Result<()> {
            Ok(())
        }
        fn list_user_models(&self) -> crate::framework::storage::Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn get_patterns(&self) -> crate::framework::storage::Result<RecipePatterns> {
            Ok(RecipePatterns::default())
        }
        fn save_patterns(
            &self,
            _patterns: &RecipePatterns,
        ) -> crate::framework::storage::Result<()> {
            Ok(())
        }
    }

    fn populated_store() -> Arc<FakeKnowledgeStore> {
        let store = FakeKnowledgeStore::default();
        let mut config = SiteConfig::new("example.com");
        config.preferred_method = ParseMethod::Microdata;
        config.selectors = Selectors {
            title: Some(".recipe-title".to_string()),
            ..Default::default()
        };
        store.save_site_config(&config).unwrap();
        store.save_user_model(&UserModel::default_user()).unwrap();
        let mut patterns = RecipePatterns::default();
        patterns.anti_patterns.push(AntiPattern {
            description: "video pages".to_string(),
            sites: vec!["example.com".to_string()],
            failure_rate: 0.9,
            action: AntiPatternAction::SkipUrl,
            sample_size: 10,
            confidence: 0.8,
        });
        store.save_patterns(&patterns).unwrap();
        Arc::new(store)
    }

    #[tokio::test]
    async fn returns_config_user_model_and_patterns_for_domain() {
        let tool = GetKnowledgeTool::new(populated_store());
        let res = tool.execute(json!({"domain": "example.com"})).await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        assert_eq!(v["site_configs"][0]["domain"], "example.com");
        assert_eq!(v["site_configs"][0]["preferred_method"], "microdata");
        assert!(v["user_model"]["user_id"].as_str().is_some());
        assert_eq!(v["patterns"]["anti_patterns"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn exclude_patterns_when_requested() {
        let tool = GetKnowledgeTool::new(populated_store());
        let res = tool
            .execute(json!({"domain": "example.com", "include_patterns": false}))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        assert!(v.get("patterns").is_none());
        assert!(v["site_configs"].is_array());
    }

    #[tokio::test]
    async fn returns_all_configs_when_no_domain() {
        let store = populated_store();
        let mut second = SiteConfig::new("other.com");
        second.preferred_method = ParseMethod::Heuristic;
        store.save_site_config(&second).unwrap();
        let tool = GetKnowledgeTool::new(store);
        let res = tool.execute(json!({})).await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        let configs = v["site_configs"].as_array().unwrap();
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0]["domain"], "example.com");
        assert_eq!(configs[1]["domain"], "other.com");
    }

    #[tokio::test]
    async fn missing_domain_returns_empty_configs() {
        let tool = GetKnowledgeTool::new(Arc::new(FakeKnowledgeStore::default()));
        let res = tool.execute(json!({"domain": "nope.com"})).await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        assert!(v["site_configs"].as_array().unwrap().is_empty());
        assert!(v["user_model"].is_null());
    }

    #[tokio::test]
    async fn load_error_surfaces_as_tool_error() {
        let tool = GetKnowledgeTool::new(Arc::new(FailingKnowledgeStore));
        let res = tool.execute(json!({"domain": "example.com"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("load configs failed"));
    }
}

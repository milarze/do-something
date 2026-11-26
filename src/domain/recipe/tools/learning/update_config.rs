//! Update site-specific parsing configuration.
//!
//! [`UpdateConfigTool`] merges a partial config patch into the existing
//! [`SiteConfig`] for a domain (or replaces it), validates the result, backs
//! up the prior config, persists it, and records a [`ConfigUpdated`] signal
//! for each changed field. The knowledge store, signal log, and backup store
//! are all injected so every dependency can be faked in tests.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};

use crate::domain::recipe::models::signal::{RecipeSignal, RecipeSignalType};
use crate::domain::recipe::models::site_config::SiteConfig;
use crate::domain::recipe::storage::{KnowledgeStorage, SignalStorage};
use crate::framework::storage::{Result as StorageResult, StorageError};
use crate::framework::tools::traits::{Tool, ToolResult};

/// Writes a backup of an existing site config before it is overwritten.
///
/// Injectable so tests can capture backups in memory.
pub trait ConfigBackupStore: Send + Sync {
    /// Persist a backup of `content` for `domain`.
    fn backup(&self, domain: &str, content: &str) -> StorageResult<()>;
}

/// Filesystem-backed config backup store.
///
/// Writes `<domain>.json.bak` into the provided directory.
pub struct FileConfigBackupStore {
    dir: PathBuf,
}

impl FileConfigBackupStore {
    /// Create a store writing backups into `dir`.
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }
}

impl ConfigBackupStore for FileConfigBackupStore {
    fn backup(&self, domain: &str, content: &str) -> StorageResult<()> {
        if domain.is_empty()
            || domain.contains('/')
            || domain.contains('\\')
            || domain.contains("..")
            || domain.contains('\0')
        {
            return Err(StorageError::InvalidPath(format!(
                "invalid domain for backup: {domain}"
            )));
        }
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{domain}.json.bak"));
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Update parsing configuration for a specific site.
pub struct UpdateConfigTool {
    knowledge: Arc<dyn KnowledgeStorage>,
    signal_log: Arc<dyn SignalStorage>,
    backups: Arc<dyn ConfigBackupStore>,
}

impl UpdateConfigTool {
    /// Create a tool backed by injected storage, signal, and backup deps.
    pub fn new(
        knowledge: Arc<dyn KnowledgeStorage>,
        signal_log: Arc<dyn SignalStorage>,
        backups: Arc<dyn ConfigBackupStore>,
    ) -> Self {
        Self {
            knowledge,
            signal_log,
            backups,
        }
    }
}

/// Field names that are bookkeeping rather than meaningful config changes.
const IGNORED_DIFF_FIELDS: &[&str] = &["version", "updated_at"];

#[async_trait]
impl Tool for UpdateConfigTool {
    fn name(&self) -> &str {
        "update_site_config"
    }

    fn description(&self) -> &str {
        "Update parsing configuration for a specific site. \
         Creates backup of existing config. Use when parsing fails."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Site domain (e.g., allrecipes.com)"
                },
                "config": {
                    "type": "object",
                    "description": "New configuration (partial update allowed)"
                },
                "merge": {
                    "type": "boolean",
                    "description": "Merge with existing config (default: true)"
                }
            },
            "required": ["domain", "config"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let domain = match args.get("domain").and_then(Value::as_str) {
            Some(d) if !d.is_empty() => d.to_string(),
            _ => return ToolResult::error("missing or empty `domain`"),
        };
        let merge = args.get("merge").and_then(Value::as_bool).unwrap_or(true);
        let patch = match args.get("config") {
            Some(v) if v.is_object() => v.clone(),
            Some(_) => return ToolResult::error("`config` must be an object"),
            None => return ToolResult::error("missing `config`"),
        };

        // Existing config (or a fresh default).
        let existing = match self.knowledge.get_site_config(&domain) {
            Ok(Some(c)) => c,
            Ok(None) => SiteConfig::new(&domain),
            Err(e) => return ToolResult::error(format!("load config failed: {e}")),
        };
        let existing_json = serde_json::to_value(&existing).unwrap_or_else(|_| json!({}));

        // Merge or replace.
        let merged_json = if merge {
            merge_json_objects(existing_json.clone(), patch)
        } else {
            patch
        };

        // Deserialize the merged patch into a concrete config.
        let mut new_config: SiteConfig = match serde_json::from_value(merged_json) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("invalid config: {e}")),
        };
        new_config.domain = domain.clone();
        new_config.version = existing.version + 1;
        new_config.updated_at = Utc::now();

        // Validate before persisting.
        if let Err(msg) = validate_config(&new_config) {
            return ToolResult::error(msg);
        }

        // Back up the existing config.
        let backup_content = serde_json::to_string_pretty(&existing).unwrap_or_default();
        if let Err(e) = self.backups.backup(&domain, &backup_content) {
            return ToolResult::error(format!("backup failed: {e}"));
        }

        // Persist.
        if let Err(e) = self.knowledge.save_site_config(&new_config) {
            return ToolResult::error(format!("save failed: {e}"));
        }

        // Record a signal per changed field (excluding bookkeeping fields).
        let new_json = serde_json::to_value(&new_config).unwrap_or_else(|_| json!({}));
        let diffs = config_diffs(&existing_json, &new_json);
        for (field, old_value, new_value) in &diffs {
            let signal = RecipeSignal::new(RecipeSignalType::ConfigUpdated {
                domain: domain.clone(),
                field: field.clone(),
                old_value: old_value.clone(),
                new_value: new_value.clone(),
            })
            .with_domain(&domain);
            let _ = self.signal_log.append(&signal);
        }

        let updated_fields: Vec<&str> = diffs.iter().map(|(f, _, _)| f.as_str()).collect();
        ToolResult::success(json!({
            "domain": domain,
            "version": new_config.version,
            "updated_fields": updated_fields,
        }))
    }
}

/// Validate a config before it is persisted.
fn validate_config(config: &SiteConfig) -> Result<(), String> {
    if config.domain.is_empty() {
        return Err("config domain must not be empty".to_string());
    }
    if config.rate_limit_ms == 0 {
        return Err("rate_limit_ms must be greater than 0".to_string());
    }
    Ok(())
}

/// Deep-merge `patch` into `base`, returning a new JSON object value.
///
/// For object-valued keys present in both, the merge recurses; for all other
/// keys, the patch value overrides the base value.
fn merge_json_objects(base: Value, patch: Value) -> Value {
    let (mut base_map, patch_map) = match (base, patch) {
        (Value::Object(b), Value::Object(p)) => (b, p),
        (b, _) => return b,
    };
    for (key, patch_val) in patch_map {
        if let Some(base_val) = base_map.get_mut(&key)
            && base_val.is_object()
            && patch_val.is_object()
        {
            *base_val = merge_json_objects(std::mem::take(base_val), patch_val);
        } else {
            base_map.insert(key, patch_val);
        }
    }
    Value::Object(base_map)
}

/// Compute the changed fields between two serialized configs.
///
/// Returns `(field, old_value, new_value)` triples, excluding the bookkeeping
/// fields in [`IGNORED_DIFF_FIELDS`].
fn config_diffs(old: &Value, new: &Value) -> Vec<(String, Option<String>, String)> {
    let mut diffs = Vec::new();
    if let (Value::Object(old_map), Value::Object(new_map)) = (old, new) {
        for (key, new_val) in new_map {
            if IGNORED_DIFF_FIELDS.contains(&key.as_str()) {
                continue;
            }
            let old_val = old_map.get(key).unwrap_or(&Value::Null);
            if old_val != new_val {
                let old_str = if old_val == &Value::Null {
                    None
                } else {
                    Some(old_val.to_string())
                };
                diffs.push((key.clone(), old_str, new_val.to_string()));
            }
        }
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::recipe::models::patterns::RecipePatterns;
    use crate::domain::recipe::models::signal::RecipeSignalType;
    use crate::domain::recipe::models::user_model::UserModel;
    use chrono::NaiveDate;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;

    // --- Shared in-memory fakes ---

    #[derive(Default)]
    struct FakeKnowledgeStore {
        configs: Mutex<HashMap<String, SiteConfig>>,
        patterns: Mutex<RecipePatterns>,
        users: Mutex<HashMap<String, UserModel>>,
        fail_save: bool,
    }

    impl KnowledgeStorage for FakeKnowledgeStore {
        fn get_site_config(
            &self,
            domain: &str,
        ) -> crate::framework::storage::Result<Option<SiteConfig>> {
            Ok(self.configs.lock().unwrap().get(domain).cloned())
        }
        fn save_site_config(&self, config: &SiteConfig) -> crate::framework::storage::Result<()> {
            if self.fail_save {
                return Err(StorageError::LockError("simulated".to_string()));
            }
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

    #[derive(Default)]
    struct FakeSignalLog {
        signals: Mutex<Vec<RecipeSignal>>,
    }

    impl SignalStorage for FakeSignalLog {
        fn append(&self, signal: &RecipeSignal) -> crate::framework::storage::Result<()> {
            self.signals.lock().unwrap().push(signal.clone());
            Ok(())
        }
        fn read_range(
            &self,
            _from: NaiveDate,
            _to: NaiveDate,
        ) -> crate::framework::storage::Result<Vec<RecipeSignal>> {
            Ok(Vec::new())
        }
        fn count_for_domain(
            &self,
            _domain: &str,
            _days: u32,
        ) -> crate::framework::storage::Result<u64> {
            Ok(0)
        }
        fn prune(&self, _older_than_days: u32) -> crate::framework::storage::Result<u64> {
            Ok(0)
        }
        fn read_date(
            &self,
            _date: NaiveDate,
        ) -> crate::framework::storage::Result<Vec<RecipeSignal>> {
            Ok(Vec::new())
        }
        fn available_dates(&self) -> crate::framework::storage::Result<Vec<NaiveDate>> {
            Ok(Vec::new())
        }
    }

    #[derive(Default)]
    struct RecordingBackupStore {
        backups: Mutex<Vec<(String, String)>>,
        fail: bool,
    }

    impl ConfigBackupStore for RecordingBackupStore {
        fn backup(&self, domain: &str, content: &str) -> StorageResult<()> {
            if self.fail {
                return Err(StorageError::LockError("simulated".to_string()));
            }
            self.backups
                .lock()
                .unwrap()
                .push((domain.to_string(), content.to_string()));
            Ok(())
        }
    }

    fn make_tool(
        knowledge: Arc<FakeKnowledgeStore>,
        signal_log: Arc<FakeSignalLog>,
        backups: Arc<RecordingBackupStore>,
    ) -> UpdateConfigTool {
        UpdateConfigTool::new(knowledge, signal_log, backups)
    }

    fn deps() -> (
        Arc<FakeKnowledgeStore>,
        Arc<FakeSignalLog>,
        Arc<RecordingBackupStore>,
    ) {
        (
            Arc::new(FakeKnowledgeStore::default()),
            Arc::new(FakeSignalLog::default()),
            Arc::new(RecordingBackupStore::default()),
        )
    }

    // --- Argument validation ---

    #[tokio::test]
    async fn missing_domain_errors() {
        let (k, s, b) = deps();
        let tool = make_tool(k, s, b);
        let res = tool.execute(json!({"config": {}})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("missing or empty `domain`"));
    }

    #[tokio::test]
    async fn missing_config_errors() {
        let (k, s, b) = deps();
        let tool = make_tool(k, s, b);
        let res = tool.execute(json!({"domain": "example.com"})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("missing `config`"));
    }

    #[tokio::test]
    async fn config_must_be_object() {
        let (k, s, b) = deps();
        let tool = make_tool(k, s, b);
        let res = tool
            .execute(json!({"domain": "example.com", "config": "oops"}))
            .await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("`config` must be an object"));
    }

    // --- Validation ---

    #[tokio::test]
    async fn rejects_zero_rate_limit() {
        let (k, s, b) = deps();
        let tool = make_tool(k.clone(), s, b.clone());
        let res = tool
            .execute(json!({
                "domain": "example.com",
                "config": {"rate_limit_ms": 0}
            }))
            .await;
        assert!(res.is_error());
        assert_eq!(
            res.error_message(),
            Some("rate_limit_ms must be greater than 0")
        );
        // Validation happens before backup/save.
        assert!(k.configs.lock().unwrap().is_empty());
        assert!(b.backups.lock().unwrap().is_empty());
    }

    // --- Merge behavior ---

    #[tokio::test]
    async fn merges_partial_config_keeping_existing_fields() {
        let (k, s, b) = deps();
        let mut existing = SiteConfig::new("example.com");
        existing.rate_limit_ms = 2000;
        k.save_site_config(&existing).unwrap();

        let tool = make_tool(k.clone(), s, b);
        let res = tool
            .execute(json!({
                "domain": "example.com",
                "config": {"selectors": {"title": ".recipe-title"}}
            }))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        assert_eq!(v["updated_fields"][0], "selectors");

        let saved = k.get_site_config("example.com").unwrap().unwrap();
        assert_eq!(saved.rate_limit_ms, 2000); // preserved
        assert_eq!(saved.selectors.title.as_deref(), Some(".recipe-title")); // merged in
        assert_eq!(saved.version, 1); // bumped
    }

    #[tokio::test]
    async fn replaces_config_when_merge_false() {
        let (k, s, b) = deps();
        let mut existing = SiteConfig::new("example.com");
        existing.rate_limit_ms = 2000;
        existing.preferred_method = crate::domain::recipe::models::recipe::ParseMethod::Microdata;
        k.save_site_config(&existing).unwrap();

        let tool = make_tool(k.clone(), s, b);
        tool.execute(json!({
            "domain": "example.com",
            "merge": false,
            "config": {"domain": "example.com", "rate_limit_ms": 500}
        }))
        .await;

        let saved = k.get_site_config("example.com").unwrap().unwrap();
        // Replaced: rate_limit from patch, method reset to default (SchemaOrg).
        assert_eq!(saved.rate_limit_ms, 500);
        assert_eq!(
            saved.preferred_method,
            crate::domain::recipe::models::recipe::ParseMethod::SchemaOrg
        );
    }

    #[tokio::test]
    async fn creates_default_config_when_none_exists() {
        let (k, s, b) = deps();
        let tool = make_tool(k.clone(), s, b);
        let res = tool
            .execute(json!({
                "domain": "new.com",
                "config": {"rate_limit_ms": 3000}
            }))
            .await;
        assert!(res.is_success());
        let saved = k.get_site_config("new.com").unwrap().unwrap();
        assert_eq!(saved.rate_limit_ms, 3000);
        assert_eq!(saved.version, 1);
    }

    // --- Backup ---

    #[tokio::test]
    async fn creates_backup_of_existing_config() {
        let (k, s, b) = deps();
        let mut existing = SiteConfig::new("example.com");
        existing.rate_limit_ms = 1500;
        k.save_site_config(&existing).unwrap();

        let tool = make_tool(k, s, b.clone());
        tool.execute(json!({
            "domain": "example.com",
            "config": {"rate_limit_ms": 999}
        }))
        .await;

        let backups = b.backups.lock().unwrap();
        assert_eq!(backups.len(), 1);
        assert_eq!(backups[0].0, "example.com");
        assert!(backups[0].1.contains("1500"));
    }

    #[tokio::test]
    async fn backup_failure_surfaces_as_error() {
        let (k, s, _) = deps();
        let b = Arc::new(RecordingBackupStore {
            fail: true,
            ..Default::default()
        });
        let tool = make_tool(k.clone(), s, b);
        let res = tool
            .execute(json!({
                "domain": "example.com",
                "config": {"rate_limit_ms": 1000}
            }))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("backup failed"));
        // Nothing saved.
        assert!(k.get_site_config("example.com").unwrap().is_none());
    }

    // --- Signal recording ---

    #[tokio::test]
    async fn records_config_updated_signal_per_changed_field() {
        let (k, s, b) = deps();
        let tool = make_tool(k, s.clone(), b);
        tool.execute(json!({
            "domain": "example.com",
            "config": {"rate_limit_ms": 5000, "requires_js": true}
        }))
        .await;

        let signals = s.signals.lock().unwrap();
        assert_eq!(signals.len(), 2);
        for s in signals.iter() {
            match &s.signal_type {
                RecipeSignalType::ConfigUpdated { domain, field, .. } => {
                    assert_eq!(domain, "example.com");
                    assert!(field == "rate_limit_ms" || field == "requires_js");
                }
                other => panic!("unexpected signal: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn no_changes_records_no_signals() {
        let (k, s, b) = deps();
        // Save a config, then "update" with a no-op patch.
        let existing = SiteConfig::new("example.com");
        k.save_site_config(&existing).unwrap();
        let tool = make_tool(k, s.clone(), b);
        tool.execute(json!({"domain": "example.com", "config": {}}))
            .await;
        assert!(s.signals.lock().unwrap().is_empty());
    }

    // --- Storage errors ---

    #[tokio::test]
    async fn save_error_surfaces_as_tool_error() {
        let k = Arc::new(FakeKnowledgeStore {
            fail_save: true,
            ..Default::default()
        });
        let (.., s, b) = deps();
        let tool = make_tool(k, s, b);
        let res = tool
            .execute(json!({
                "domain": "example.com",
                "config": {"rate_limit_ms": 1000}
            }))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("save failed"));
    }

    #[tokio::test]
    async fn load_error_surfaces_as_tool_error() {
        // A knowledge store that fails get_site_config.
        struct FailingLoad;
        impl KnowledgeStorage for FailingLoad {
            fn get_site_config(
                &self,
                _d: &str,
            ) -> crate::framework::storage::Result<Option<SiteConfig>> {
                Err(StorageError::LockError("load boom".to_string()))
            }
            fn save_site_config(&self, _c: &SiteConfig) -> crate::framework::storage::Result<()> {
                Ok(())
            }
            fn list_site_configs(&self) -> crate::framework::storage::Result<Vec<String>> {
                Ok(Vec::new())
            }
            fn delete_site_config(&self, _d: &str) -> crate::framework::storage::Result<()> {
                Ok(())
            }
            fn get_user_model(
                &self,
                _u: &str,
            ) -> crate::framework::storage::Result<Option<UserModel>> {
                Ok(None)
            }
            fn save_user_model(&self, _m: &UserModel) -> crate::framework::storage::Result<()> {
                Ok(())
            }
            fn list_user_models(&self) -> crate::framework::storage::Result<Vec<String>> {
                Ok(Vec::new())
            }
            fn get_patterns(&self) -> crate::framework::storage::Result<RecipePatterns> {
                Ok(RecipePatterns::default())
            }
            fn save_patterns(&self, _p: &RecipePatterns) -> crate::framework::storage::Result<()> {
                Ok(())
            }
        }
        let (.., s, b) = deps();
        let tool = UpdateConfigTool::new(Arc::new(FailingLoad), s, b);
        let res = tool
            .execute(json!({
                "domain": "example.com",
                "config": {"rate_limit_ms": 1000}
            }))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("load config failed"));
    }

    // --- Pure helper tests ---

    #[test]
    fn merge_recurses_into_nested_objects() {
        let base = json!({"selectors": {"title": "a", "ingredients": "b"}, "rate_limit_ms": 1000});
        let patch = json!({"selectors": {"title": "c"}});
        let merged = merge_json_objects(base, patch);
        assert_eq!(merged["selectors"]["title"], "c");
        assert_eq!(merged["selectors"]["ingredients"], "b");
        assert_eq!(merged["rate_limit_ms"], 1000);
    }

    #[test]
    fn merge_non_object_patch_returns_base() {
        let base = json!({"a": 1});
        let merged = merge_json_objects(base, json!("not an object"));
        assert_eq!(merged, json!({"a": 1}));
    }

    #[test]
    fn diff_excludes_bookkeeping_fields() {
        let old = json!({"rate_limit_ms": 1000, "version": 1, "updated_at": "x"});
        let new = json!({"rate_limit_ms": 1000, "version": 2, "updated_at": "y"});
        let diffs = config_diffs(&old, &new);
        assert!(diffs.is_empty());
    }

    #[test]
    fn diff_reports_changed_and_added_fields() {
        let old = json!({"rate_limit_ms": 1000, "requires_js": false});
        let new = json!({"rate_limit_ms": 2000, "requires_js": false, "skip_patterns": ["x"]});
        let diffs = config_diffs(&old, &new);
        let fields: Vec<&str> = diffs.iter().map(|(f, _, _)| f.as_str()).collect();
        assert_eq!(fields, vec!["rate_limit_ms", "skip_patterns"]);
        // rate_limit_ms changed 1000 -> 2000.
        let rl = diffs.iter().find(|(f, _, _)| f == "rate_limit_ms").unwrap();
        assert_eq!(rl.1.as_deref(), Some("1000"));
        assert_eq!(rl.2, "2000");
        // skip_patterns added (old None).
        let sp = diffs.iter().find(|(f, _, _)| f == "skip_patterns").unwrap();
        assert_eq!(sp.1, None);
    }

    // --- FileConfigBackupStore ---

    #[test]
    fn file_backup_writes_bak_file() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let store = FileConfigBackupStore::new(dir.path().to_path_buf());
        store.backup("example.com", "{}").unwrap();
        assert!(dir.path().join("example.com.json.bak").exists());
    }

    #[test]
    fn file_backup_rejects_path_traversal() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let store = FileConfigBackupStore::new(dir.path().to_path_buf());
        let res = store.backup("../etc", "{}");
        assert!(res.is_err());
    }
}

//! Record a feedback signal.
//!
//! [`RecordSignalTool`] deserializes a [`RecipeSignalType`] from the tool
//! arguments, wraps it in a [`RecipeSignal`], and appends it to a
//! [`SignalStorage`] backend. The storage backend is injected so tests can
//! capture signals in memory.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::domain::recipe::models::signal::{RecipeSignal, RecipeSignalType};
use crate::domain::recipe::storage::SignalStorage;
use crate::framework::tools::traits::{Tool, ToolResult};

/// Record a feedback signal for learning.
pub struct RecordSignalTool {
    signal_log: Arc<dyn SignalStorage>,
}

impl RecordSignalTool {
    /// Create a tool backed by any [`SignalStorage`] implementation.
    pub fn new(signal_log: Arc<dyn SignalStorage>) -> Self {
        Self { signal_log }
    }
}

#[async_trait]
impl Tool for RecordSignalTool {
    fn name(&self) -> &str {
        "record_signal"
    }

    fn description(&self) -> &str {
        "Record a feedback signal for learning. \
         Use after parse success/failure, or when user provides feedback."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "signal_type": {
                    "type": "object",
                    "description": "Type of signal (parse_success, parse_failure, etc.)"
                },
                "domain": { "type": "string" },
                "url": { "type": "string" },
                "recipe_id": { "type": "string" }
            },
            "required": ["signal_type"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let Some(signal_type_value) = args.get("signal_type") else {
            return ToolResult::error("missing `signal_type`");
        };

        let signal_type: RecipeSignalType = match serde_json::from_value(signal_type_value.clone())
        {
            Ok(t) => t,
            Err(e) => return ToolResult::error(format!("invalid signal_type: {e}")),
        };

        let mut signal = RecipeSignal::new(signal_type);
        if let Some(d) = args.get("domain").and_then(Value::as_str) {
            signal = signal.with_domain(d);
        }
        if let Some(u) = args.get("url").and_then(Value::as_str) {
            signal = signal.with_url(u);
        }
        if let Some(r) = args.get("recipe_id").and_then(Value::as_str) {
            signal = signal.with_recipe(r);
        }

        match self.signal_log.append(&signal) {
            Ok(()) => ToolResult::success(json!({
                "signal_id": signal.id.0,
                "recorded_at": signal.timestamp,
            })),
            Err(e) => ToolResult::error(format!("record failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::recipe::models::ParseMethod;
    use crate::domain::recipe::models::signal::RecipeSignalType;
    use crate::framework::storage::StorageError;
    use chrono::NaiveDate;
    use serde_json::json;
    use std::sync::Mutex;

    /// In-memory signal log that captures appended signals.
    #[derive(Default)]
    struct FakeSignalLog {
        signals: Mutex<Vec<RecipeSignal>>,
        fail: bool,
    }

    impl SignalStorage for FakeSignalLog {
        fn append(&self, signal: &RecipeSignal) -> crate::framework::storage::Result<()> {
            if self.fail {
                return Err(StorageError::LockError("simulated".to_string()));
            }
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

    #[tokio::test]
    async fn missing_signal_type_errors() {
        let tool = RecordSignalTool::new(Arc::new(FakeSignalLog::default()));
        let res = tool.execute(json!({})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("missing `signal_type`"));
    }

    #[tokio::test]
    async fn invalid_signal_type_errors() {
        let tool = RecordSignalTool::new(Arc::new(FakeSignalLog::default()));
        let res = tool.execute(json!({"signal_type": "not an object"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("invalid signal_type"));
    }

    #[tokio::test]
    async fn records_signal_with_metadata() {
        let log = Arc::new(FakeSignalLog::default());
        let tool = RecordSignalTool::new(log.clone());
        let res = tool
            .execute(json!({
                "signal_type": {
                    "type": "parse_success",
                    "method": "schema_org",
                    "time_ms": 350
                },
                "domain": "example.com",
                "url": "https://example.com/r",
                "recipe_id": "rc_1"
            }))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        assert!(v["signal_id"].as_str().unwrap().starts_with("sig_"));
        assert!(v["recorded_at"].as_str().is_some());

        let signals = log.signals.lock().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].domain.as_deref(), Some("example.com"));
        assert_eq!(signals[0].url.as_deref(), Some("https://example.com/r"));
        assert_eq!(signals[0].recipe_id.as_deref(), Some("rc_1"));
        match &signals[0].signal_type {
            RecipeSignalType::ParseSuccess { method, time_ms } => {
                assert_eq!(*method, ParseMethod::SchemaOrg);
                assert_eq!(*time_ms, 350);
            }
            other => panic!("unexpected signal type: {other:?}"),
        }
    }

    #[tokio::test]
    async fn records_signal_without_optional_fields() {
        let log = Arc::new(FakeSignalLog::default());
        let tool = RecordSignalTool::new(log.clone());
        tool.execute(json!({
            "signal_type": {"type": "rate_limit_hit", "wait_time_ms": 1000}
        }))
        .await;
        let signals = log.signals.lock().unwrap();
        assert_eq!(signals.len(), 1);
        assert!(signals[0].domain.is_none());
        assert!(signals[0].url.is_none());
        assert!(signals[0].recipe_id.is_none());
    }

    #[tokio::test]
    async fn append_error_surfaces_as_tool_error() {
        let log = Arc::new(FakeSignalLog {
            fail: true,
            ..Default::default()
        });
        let tool = RecordSignalTool::new(log);
        let res = tool
            .execute(json!({
                "signal_type": {"type": "parse_success", "method": "schema_org", "time_ms": 1}
            }))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("record failed"));
    }
}

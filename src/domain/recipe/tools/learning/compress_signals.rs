//! Manually trigger signal compression.
//!
//! [`CompressSignalsTool`] is a thin wrapper that delegates to a
//! [`CompressionRunnerPort`]. It contains no compression logic of its own.
//! The port abstraction lets the tool be unit-tested without a real signal
//! log, checkpoint store, or learning domain.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::domain::recipe::learning::RecipeLearningDomain;
use crate::framework::learning::runner::CompressionRunner;
use crate::framework::storage::CompressionStats;
use crate::framework::tools::traits::{Tool, ToolResult};

/// Abstraction over the compression runner.
///
/// Implemented by [`CompressionRunner<RecipeLearningDomain>`]; tests inject a
/// fake to avoid standing up a real signal log and checkpoint store.
pub trait CompressionRunnerPort: Send + Sync {
    /// Run compression for `domain` (or all domains if `None`) over `days`.
    fn run(&self, domain: Option<&str>, days: u32) -> anyhow::Result<CompressionStats>;
}

impl CompressionRunnerPort for CompressionRunner<RecipeLearningDomain> {
    fn run(&self, domain: Option<&str>, days: u32) -> anyhow::Result<CompressionStats> {
        CompressionRunner::run(self, domain, days)
    }
}

/// Trigger signal compression manually.
pub struct CompressSignalsTool {
    runner: Arc<dyn CompressionRunnerPort>,
}

impl CompressSignalsTool {
    /// Create a tool backed by any [`CompressionRunnerPort`].
    pub fn new(runner: Arc<dyn CompressionRunnerPort>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl Tool for CompressSignalsTool {
    fn name(&self) -> &str {
        "compress_signals"
    }

    fn description(&self) -> &str {
        "Run signal compression to extract patterns. \
         Automatically runs periodically, but can be triggered manually."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional domain to limit compression scope"
                },
                "days": {
                    "type": "integer",
                    "description": "Days of signals to process (default: 7)"
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let domain = args
            .get("domain")
            .and_then(Value::as_str)
            .map(str::to_string);
        let days = args.get("days").and_then(Value::as_u64).unwrap_or(7) as u32;

        match self.runner.run(domain.as_deref(), days) {
            Ok(stats) => ToolResult::success(json!({
                "signals_processed": stats.signals_processed,
                "signals_pruned": stats.signals_pruned,
                "patterns_extracted": stats.patterns_extracted,
                "configs_updated": stats.configs_updated,
            })),
            Err(e) => ToolResult::error(format!("compression failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::storage::CompressionStats;
    use chrono::Utc;
    use serde_json::json;

    /// A fake runner that returns a canned result or error.
    struct FakeRunner {
        stats: CompressionStats,
        error: Option<String>,
        last_call: std::sync::Mutex<Option<(Option<String>, u32)>>,
    }

    impl FakeRunner {
        fn ok(stats: CompressionStats) -> Self {
            Self {
                stats,
                error: None,
                last_call: std::sync::Mutex::new(None),
            }
        }

        fn failing() -> Self {
            Self {
                stats: canned_stats(),
                error: Some("boom".to_string()),
                last_call: std::sync::Mutex::new(None),
            }
        }

        fn last_call(&self) -> Option<(Option<String>, u32)> {
            self.last_call.lock().unwrap().clone()
        }
    }

    impl CompressionRunnerPort for FakeRunner {
        fn run(&self, domain: Option<&str>, days: u32) -> anyhow::Result<CompressionStats> {
            self.last_call
                .lock()
                .unwrap()
                .replace((domain.map(str::to_string), days));
            match &self.error {
                Some(msg) => Err(anyhow::anyhow!("{msg}")),
                None => Ok(self.stats.clone()),
            }
        }
    }

    fn canned_stats() -> CompressionStats {
        CompressionStats {
            signals_processed: 12,
            patterns_extracted: 2,
            configs_updated: 1,
            signals_pruned: 5,
            time_ms: 40,
            completed_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn delegates_to_runner_and_returns_stats() {
        let runner = Arc::new(FakeRunner::ok(canned_stats()));
        let tool = CompressSignalsTool::new(runner.clone());
        let res = tool
            .execute(json!({"domain": "example.com", "days": 3}))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {res:?}");
        };
        assert_eq!(v["signals_processed"], 12);
        assert_eq!(v["patterns_extracted"], 2);
        assert_eq!(v["configs_updated"], 1);
        assert_eq!(v["signals_pruned"], 5);
        assert_eq!(
            runner.last_call(),
            Some((Some("example.com".to_string()), 3))
        );
    }

    #[tokio::test]
    async fn defaults_days_to_seven_when_omitted() {
        let runner = Arc::new(FakeRunner::ok(canned_stats()));
        let tool = CompressSignalsTool::new(runner.clone());
        tool.execute(json!({})).await;
        assert_eq!(runner.last_call(), Some((None, 7)));
    }

    #[tokio::test]
    async fn no_domain_passes_none() {
        let runner = Arc::new(FakeRunner::ok(canned_stats()));
        let tool = CompressSignalsTool::new(runner.clone());
        tool.execute(json!({"days": 14})).await;
        assert_eq!(runner.last_call(), Some((None, 14)));
    }

    #[tokio::test]
    async fn runner_error_surfaces_as_tool_error() {
        let tool = CompressSignalsTool::new(Arc::new(FakeRunner::failing()));
        let res = tool.execute(json!({})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("compression failed"));
        assert!(res.error_message().unwrap().contains("boom"));
    }
}

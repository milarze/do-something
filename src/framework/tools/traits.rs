//! Generic tool interface.
//!
//! The [`Tool`] trait is the contract every domain tool implements. Tools are
//! async operations invoked by the agent during its tool-calling loop. The
//! framework treats tools as opaque beyond these methods.

use async_trait::async_trait;
use serde_json::Value;

/// A request for user permission before a tool proceeds.
///
/// Surfaced via [`ToolResult::PermissionRequired`] when a tool cannot proceed
/// without explicit user approval.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    /// Name of the tool requesting permission.
    pub tool: String,
    /// Human-readable reason for the request.
    pub message: String,
}

/// The outcome of executing a tool.
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Successful execution with a JSON value for the model to consume.
    Success(Value),
    /// Execution failed with a human-readable message.
    Error(String),
    /// The tool needs user permission before proceeding.
    PermissionRequired {
        /// The permission request details.
        request: PermissionRequest,
    },
}

impl ToolResult {
    /// Create a successful result carrying a JSON value.
    pub fn success(value: Value) -> Self {
        Self::Success(value)
    }

    /// Create an error result with a message.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    /// Create a permission-required result.
    pub fn permission_required(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::PermissionRequired {
            request: PermissionRequest {
                tool: tool.into(),
                message: message.into(),
            },
        }
    }

    /// Whether this result represents a successful execution.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Whether this result represents a failure.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// The error message, if this is an error result.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

/// Generic tool interface implemented by domain tools.
///
/// Implementations must be [`Send`] + [`Sync`] so they can be stored in an
/// [`Arc<dyn Tool`](std::sync::Arc) inside a [`super::ToolRegistry`] and
/// shared across tasks. The trait is dyn-compatible via `async_trait`.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Static tool name — also the registry key and the LLM function name.
    fn name(&self) -> &str;

    /// Human-readable description shown to the model.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given JSON arguments.
    async fn execute(&self, args: Value) -> ToolResult;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn success_helpers() {
        let r = ToolResult::success(json!({"ok": true}));
        assert!(r.is_success());
        assert!(!r.is_error());
        assert!(r.error_message().is_none());
    }

    #[test]
    fn error_helpers() {
        let r = ToolResult::error("boom");
        assert!(r.is_error());
        assert!(!r.is_success());
        assert_eq!(r.error_message(), Some("boom"));
    }

    #[test]
    fn permission_required_carries_request() {
        let r = ToolResult::permission_required("save_recipe", "needs approval");
        match &r {
            ToolResult::PermissionRequired { request } => {
                assert_eq!(request.tool, "save_recipe");
                assert_eq!(request.message, "needs approval");
            }
            _ => panic!("expected PermissionRequired"),
        }
        assert!(!r.is_success());
        assert!(!r.is_error());
    }
}

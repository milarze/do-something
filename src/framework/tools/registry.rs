//! Central tool registry and OpenAI-format tool definitions.
//!
//! [`ToolRegistry`] holds tools keyed by name and produces the
//! `tool_definitions` array the LLM consumes. It is generic over the
//! concrete tools: it stores `Arc<dyn Tool>` and never inspects a tool's
//! internals.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::traits::Tool;

/// An OpenAI-format tool definition, emitted by [`ToolRegistry::tool_definitions`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDef {
    /// Discriminator — always `"function"`.
    #[serde(rename = "type")]
    pub kind: &'static str,

    /// The function definition.
    pub function: ToolDefFunction,
}

impl ToolDef {
    /// Construct a definition from a tool's name, description, and schema.
    pub fn from_tool(tool: &dyn Tool) -> Self {
        Self {
            kind: "function",
            function: ToolDefFunction {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            },
        }
    }
}

/// The function portion of an OpenAI tool definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefFunction {
    /// Function name (matches the tool's registry key).
    pub name: String,

    /// Description shown to the model.
    pub description: String,

    /// JSON Schema for the function parameters.
    pub parameters: Value,
}

/// Central registry of available tools, keyed by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, keyed by its [`Tool::name`].
    ///
    /// If a tool with the same name already exists, it is replaced.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Whether a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// List registered tool names, sorted for determinism.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Generate OpenAI-format tool definitions for the LLM.
    ///
    /// Definitions are emitted in sorted name order for stable output.
    pub fn tool_definitions(&self) -> Vec<ToolDef> {
        let mut defs: Vec<(String, ToolDef)> = self
            .tools
            .iter()
            .map(|(name, tool)| (name.clone(), ToolDef::from_tool(tool.as_ref())))
            .collect();
        defs.sort_by(|a, b| a.0.cmp(&b.0));
        defs.into_iter().map(|(_, d)| d).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::tools::traits::ToolResult;
    use async_trait::async_trait;
    use serde_json::json;

    struct EchoTool {
        name: &'static str,
        desc: &'static str,
        schema: Value,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.desc
        }
        fn parameters_schema(&self) -> Value {
            self.schema.clone()
        }
        async fn execute(&self, _args: Value) -> ToolResult {
            ToolResult::success(json!({"echoed": self.name}))
        }
    }

    fn make_tool(name: &'static str) -> EchoTool {
        EchoTool {
            name,
            desc: "echo tool",
            schema: json!({"type": "object", "properties": {}}),
        }
    }

    #[tokio::test]
    async fn register_and_get() {
        let mut reg = ToolRegistry::new();
        assert!(reg.is_empty());
        reg.register(make_tool("alpha"));
        assert_eq!(reg.len(), 1);
        assert!(reg.contains("alpha"));
        assert!(!reg.contains("beta"));
        let tool = reg.get("alpha").unwrap();
        let res = tool.execute(json!({})).await;
        assert!(res.is_success());
    }

    #[test]
    fn get_missing_returns_none() {
        let reg = ToolRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn register_replaces_existing() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("dup"));
        reg.register(EchoTool {
            name: "dup",
            desc: "replaced",
            schema: json!({}),
        });
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("dup").unwrap().description(), "replaced");
    }

    #[test]
    fn list_is_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("charlie"));
        reg.register(make_tool("alpha"));
        reg.register(make_tool("bravo"));
        assert_eq!(reg.list(), vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn tool_definitions_sorted_and_shaped() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("zeta"));
        reg.register(make_tool("alpha"));

        let defs = reg.tool_definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].function.name, "alpha");
        assert_eq!(defs[1].function.name, "zeta");
        assert_eq!(defs[0].kind, "function");
        assert_eq!(defs[0].function.description, "echo tool");
    }

    #[test]
    fn tool_def_serializes_to_openai_shape() {
        let tool = make_tool("alpha");
        let def = ToolDef::from_tool(&tool);
        let json_str = serde_json::to_string(&def).unwrap();
        assert!(json_str.contains("\"type\":\"function\""));
        assert!(json_str.contains("\"name\":\"alpha\""));
    }

    #[tokio::test]
    async fn execute_dispatches_by_name() {
        let mut reg = ToolRegistry::new();
        reg.register(make_tool("alpha"));
        let tool = reg.get("alpha").unwrap();
        let res = tool.execute(json!({})).await;
        match res {
            ToolResult::Success(v) => assert_eq!(v, json!({"echoed": "alpha"})),
            _ => panic!("expected success"),
        }
    }
}

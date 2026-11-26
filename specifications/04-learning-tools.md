# Learning Tools

> **Placement**: Domain component (see [`00-architecture.md`](./00-architecture.md)).
> These are domain tools (`domain/recipe/tools/learning/`): `update_config`,
> `record_signal`, `get_knowledge`, and `compress_signals`.
>
> `CompressSignalsTool` is a thin wrapper that delegates to a
> `CompressionRunner<RecipeLearningDomain>` (framework) and contains no
> compression logic of its own. The other three operate on concrete recipe
> knowledge types.

## Code Style & Guidelines

1. **Test coverage**: Every public function must have test coverage
2. **Unit tests**: Prefer unit tests over integration tests. Use mocks/fakes for dependencies
3. **Dependency injection**: Accept traits as parameters, not concrete types
4. **Relevant tests**: Test behavior and edge cases, not just line coverage. Test each branch
5. **Linting**: Run `cargo clippy --all-targets --all-features -- -D warnings` before committing

## Overview

Learning tools let the agent modify its own knowledge base in response to
feedback signals. They record signals, inspect knowledge, update site configs,
and trigger compression.

## Components

### `domain/recipe/tools/learning/mod.rs`

Module organization and exports.

```rust
pub mod update_config;
pub mod record_signal;
pub mod get_knowledge;
pub mod compress_signals;

pub use update_config::UpdateConfigTool;
pub use record_signal::RecordSignalTool;
pub use get_knowledge::GetKnowledgeTool;
pub use compress_signals::CompressSignalsTool;
```

### `domain/recipe/tools/learning/update_config.rs`

Update site-specific parsing configuration.

```rust
pub struct UpdateConfigTool {
    knowledge: Arc<RecipeKnowledgeStore>,
    config_dir: Arc<ConfigDir>,
}

impl Tool for UpdateConfigTool {
    fn name(&self) -> &str { "update_site_config" }

    fn description(&self) -> &str {
        "Update parsing configuration for a specific site. \
         Creates backup of existing config. Use when parsing fails."
    }

    fn parameters_schema(&self) -> serde_json::Value {
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
        let domain: String = args["domain"].as_str().ok_or(...)?.into();
        let merge: bool = args["merge"].as_bool().unwrap_or(true);

        // Get existing config or default
        let existing = self.knowledge.get_site_config(&domain)?;

        // Parse and validate new config
        let updates: SiteConfig = serde_json::from_value(args["config"].clone())?;
        updates.validate()?;

        // Merge or replace
        let new_config = if merge {
            existing.merge(&updates)
        } else {
            updates
        };

        // Backup existing
        self.backup_config(&domain)?;

        // Save
        self.knowledge.update_site_config(&new_config)?;

        // Record the update
        self.record_config_change(&domain, &existing, &new_config)?;

        Ok(ToolOutput::success(json!({
            "domain": domain,
            "version": new_config.version,
            "updated_fields": self.diff_fields(&existing, &new_config)
        })))
    }
}

impl UpdateConfigTool {
    fn backup_config(&self, domain: &str) -> Result<()> {
        // Copy current config to .bak file
    }

    fn record_config_change(&self, domain: &str, old: &SiteConfig, new: &SiteConfig) -> Result<()> {
        // Record RecipeSignalType::ConfigUpdated
    }

    fn diff_fields(&self, old: &SiteConfig, new: &SiteConfig) -> Vec<String> {
        // Return list of changed field names
    }
}
```

**Safety Measures:**
- Validate config before saving
- Backup existing config (versioned)
- Track changes in signal log
- Agent can rollback if needed

### `domain/recipe/tools/learning/record_signal.rs`

Record a feedback signal.

```rust
pub struct RecordSignalTool {
    signal_log: Arc<RecipeSignalLog>,
}

impl Tool for RecordSignalTool {
    fn name(&self) -> &str { "record_signal" }

    fn description(&self) -> &str {
        "Record a feedback signal for learning. \
         Use after parse success/failure, or when user provides feedback."
    }

    fn parameters_schema(&self) -> serde_json::Value {
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
        let signal = RecipeSignal::from_json(args)?;

        // Append to daily log
        self.signal_log.append(&signal)?;

        Ok(ToolOutput::success(json!({
            "signal_id": signal.id,
            "recorded_at": signal.timestamp
        })))
    }
}
```

**Signal Types (recipe domain):**
- `ParseSuccess { method, time_ms }`
- `ParseFailure { method, error, attempted_methods }`
- `RecipeSaved { recipe_id, has_image, ingredient_count }`
- `RecipeDeleted { recipe_id, reason }`
- `ExplicitFeedback { feedback, sentiment }`
- `RateLimitHit { wait_time_ms }`
- `ConfigUpdated { domain, field }`

### `domain/recipe/tools/learning/get_knowledge.rs`

Retrieve a knowledge snapshot.

```rust
pub struct GetKnowledgeTool {
    knowledge: Arc<RecipeKnowledgeStore>,
}

impl Tool for GetKnowledgeTool {
    fn name(&self) -> &str { "get_knowledge" }

    fn description(&self) -> &str {
        "Retrieve current knowledge for a domain. \
         Use to understand what's known before making changes."
    }

    fn parameters_schema(&self) -> serde_json::Value {
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
        let domain: Option<String> = args["domain"].as_str().map(Into::into);
        let include_patterns = args["include_patterns"].as_bool().unwrap_or(true);

        let context = self.knowledge.load_for_context(domain.as_deref())?;

        let mut result = json!({
            "site_configs": context.site_configs,
            "user_model": context.user_model,
        });

        if include_patterns {
            let patterns = self.knowledge.get_patterns()?;
            result["patterns"] = serde_json::to_value(patterns)?;
        }

        Ok(ToolOutput::success(result))
    }
}
```

### `domain/recipe/tools/learning/compress_signals.rs`

Manually trigger compression. Delegates to the framework runner.

```rust
pub struct CompressSignalsTool {
    runner: Arc<CompressionRunner<RecipeLearningDomain>>,
}

impl Tool for CompressSignalsTool {
    fn name(&self) -> &str { "compress_signals" }

    fn description(&self) -> &str {
        "Run signal compression to extract patterns. \
         Automatically runs periodically, but can be triggered manually."
    }

    fn parameters_schema(&self) -> serde_json::Value {
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
        let domain: Option<String> = args["domain"].as_str().map(Into::into);
        let days: u32 = args["days"].as_u64().unwrap_or(7) as u32;

        let stats = self.runner.run(domain.as_deref(), days)?;

        Ok(ToolOutput::success(json!({
            "signals_processed": stats.signals_processed,
            "signals_pruned": stats.signals_pruned,
            "patterns_extracted": stats.patterns_extracted,
            "configs_updated": stats.configs_updated
        })))
    }
}
```

## Automatic Signal Recording

Some signals are recorded automatically by task tools:

| Event | Recorded By | Signal Type |
|-------|-------------|-------------|
| Successful parse | `ParseRecipeTool` | `ParseSuccess` |
| Failed parse | `ParseRecipeTool` | `ParseFailure` |
| Recipe saved | `SaveRecipeTool` | `RecipeSaved` |
| Rate limit hit | `FetchHtmlTool` | `RateLimitHit` |

The agent explicitly calls `record_signal` only for:
- User feedback (explicit or inferred from actions)
- Config updates (if using `UpdateConfigTool`)

## Tool Registry Integration

```rust
pub fn register_learning_tools(
    registry: &mut ToolRegistry,
    knowledge: Arc<RecipeKnowledgeStore>,
    signal_log: Arc<RecipeSignalLog>,
    runner: Arc<CompressionRunner<RecipeLearningDomain>>,
) {
    registry.register(UpdateConfigTool::new(knowledge.clone()));
    registry.register(RecordSignalTool::new(signal_log));
    registry.register(GetKnowledgeTool::new(knowledge.clone()));
    registry.register(CompressSignalsTool::new(runner));
}
```

## Tests

- `UpdateConfigTool` validates config before saving
- `UpdateConfigTool` creates backup
- `RecordSignalTool` appends to daily log
- `GetKnowledgeTool` returns relevant knowledge
- `CompressSignalsTool` delegates to the runner and returns stats

## Dependencies

- `framework/tools` — `Tool` trait, `ToolRegistry`
- `framework/learning` — `CompressionRunner`
- `domain/recipe` — knowledge store, signal log, models, learning domain
- `serde_json` — JSON handling

## Size

Estimate: ~500-600 lines

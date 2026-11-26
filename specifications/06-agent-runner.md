# Agent Runner

> **Placement**: This layer spans the framework/domain boundary (see
> [`00-architecture.md`](./00-architecture.md)).
> - **Framework** (`framework/agent/`): `AgentRunner` (tool-calling loop, ACP),
>   `ContextManager` (history trimming, token budgeting, summarization), the
>   generic `KnowledgeContext` / `PromptAssembler` mechanism, and a generic
>   `SessionState` holding an opaque `DomainTask`.
> - **Domain** (`domain/recipe/agent/`): `RecipePromptBuilder` (decides which
>   knowledge sections to inject and their labels) and `RecipeTaskState` /
>   `RecipeTaskTracker` (scraping progress: pending/completed URLs, recipe count).

## Code Style & Guidelines

1. **Test coverage**: Every public function must have test coverage
2. **Unit tests**: Prefer unit tests over integration tests. Use mocks/fakes for dependencies
3. **Dependency injection**: Accept traits as parameters, not concrete types
4. **Relevant tests**: Test behavior and edge cases, not just line coverage. Test each branch
5. **Linting**: Run `cargo clippy --all-targets --all-features -- -D warnings` before committing

## Overview

The agent runner is the integration layer. The framework owns the tool-calling
loop, the ACP protocol, conversation context, and token budgeting. The recipe
domain owns prompt content and task state. The binary wires them together.

## Framework Components

### `framework/agent/runner.rs`

The tool-calling loop.

```rust
pub struct AgentRunner {
    llm: Arc<LlmClient>,
    tools: Arc<ToolRegistry>,
    context: ContextManager,
    config: Arc<Config>,
}

impl AgentRunner {
    pub fn new(
        llm: LlmClient,
        tools: ToolRegistry,
        config: Config,
    ) -> Self {
        Self {
            llm: Arc::new(llm),
            tools: Arc::new(tools),
            context: ContextManager::new(config.context_budget),
            config: Arc::new(config),
        }
    }

    /// Handle a prompt turn
    pub async fn handle_prompt(
        &mut self,
        prompt: &[ContentBlock],
        cx: &ConnectionTo<Client>,
        session_id: &SessionId,
        cancel: CancellationToken,
    ) -> PromptResponse {
        // 1. Extract user text
        let user_text = extract_text(prompt);

        // 2. Add to context
        self.context.push(HistoryEntry::user(user_text.clone()));

        // 3. Run tool-calling loop
        let stop = self.tool_loop(cx, session_id, cancel).await;

        // 4. Save session state
        self.save_session(session_id);

        PromptResponse::new(stop)
    }

    async fn tool_loop(
        &mut self,
        cx: &ConnectionTo<Client>,
        session_id: &SessionId,
        cancel: CancellationToken,
    ) -> StopReason {
        let mut iterations = 0u32;

        loop {
            if cancel.is_cancelled() {
                return StopReason::Cancelled;
            }
            if iterations >= MAX_TURN_REQUESTS {
                return StopReason::MaxTurnRequests;
            }
            iterations += 1;

            // Get trimmed history
            let messages = self.context.get_trimmed(self.config.message_budget);

            // Call LLM
            let stream = self.llm.chat_stream(
                &self.config.profile,
                &messages,
                &self.tools.tool_definitions(),
                cancel.clone(),
            ).await;

            // Process stream...
            let response = self.process_stream(stream, cx, session_id).await;

            // Handle response
            match response {
                StreamResult::Text(text) => return StopReason::EndTurn,
                StreamResult::ToolCalls(calls) => {
                    for call in calls {
                        self.execute_tool(&call, cx, session_id, cancel.clone()).await;
                    }
                }
                StreamResult::Error(e) => {
                    send_text(cx, session_id, &format!("[error: {}]", e));
                    return StopReason::EndTurn;
                }
            }

            // Check task completion (domain hook)
            if self.is_task_complete() {
                if let Some(msg) = self.task_progress_message() {
                    send_text(cx, session_id, &msg);
                }
                return StopReason::EndTurn;
            }
        }
    }

    async fn execute_tool(
        &mut self,
        call: &ToolCall,
        cx: &ConnectionTo<Client>,
        session_id: &SessionId,
        cancel: CancellationToken,
    ) {
        let name = &call.function.name;
        let args = &call.function.arguments;

        let Some(tool) = self.tools.get(name) else {
            send_tool_error(cx, session_id, &call.id, &format!("Unknown tool: {}", name));
            return;
        };

        // Check permissions (if agent-managed)
        if self.config.profile.permission_mode == PermissionMode::Agent {
            if !self.request_permission(tool.as_ref(), call, cx, session_id, cancel.clone()).await {
                return;
            }
        }

        send_tool_in_progress(cx, session_id, &call.id, name);

        let result = tool.execute(serde_json::from_str(args).unwrap_or(json!({}))).await;

        // Send result to LLM
        self.context.push(HistoryEntry::tool_result(&call.id, &result));
        send_tool_result(cx, session_id, &call.id, &result);
    }
}
```

### `framework/agent/context.rs`

Conversation history with token budgeting.

```rust
pub struct ContextManager {
    history: Vec<HistoryEntry>,
    max_tokens: u64,
    tokenizer: TokenCounter,
}

impl ContextManager {
    pub fn new(max_tokens: u64) -> Self;

    /// Add a message to history
    pub fn push(&mut self, entry: HistoryEntry);

    /// Get history trimmed to fit budget
    pub fn get_trimmed(&self, budget: u64) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        let mut tokens_used = 0u64;

        // Walk backwards from most recent
        for entry in self.history.iter().rev() {
            let entry_tokens = entry.token_count;
            if tokens_used + entry_tokens > budget {
                break;
            }
            tokens_used += entry_tokens;
            messages.push(entry.to_chat_message());
        }

        messages.reverse();
        messages
    }

    /// Summarize old tool results to save tokens
    pub fn summarize_old_results(&mut self, keep_last_n: usize);
}

pub struct HistoryEntry {
    pub role: Role,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_result: Option<ToolResult>,
    pub token_count: u64,
    pub is_summarized: bool,
}

impl HistoryEntry {
    fn summarize(&mut self) {
        if let Some(ref result) = self.tool_result {
            self.content = Some(format!(
                "[tool result: {}]",
                if result.is_success() { "success" } else { "failed" }
            ));
            self.is_summarized = true;
            self.token_count = 5; // Approximate
        }
    }
}
```

### `framework/agent/prompt.rs`

Generic knowledge-context container. The domain fills sections; the framework
budgets and assembles.

```rust
pub struct KnowledgeContext {
    pub sections: Vec<KnowledgeSection>,
    pub token_budget: u64,
    pub estimated_tokens: u64,
}

pub struct KnowledgeSection {
    pub label: String,
    pub json: String,
    pub estimated_tokens: u64,
}

pub trait PromptAssembler {
    fn assemble(&self, ctx: &KnowledgeContext) -> String;
}
```

### `framework/agent/session.rs`

Generic session state holding an opaque domain task.

```rust
pub struct SessionState<T: Serialize + DeserializeOwned> {
    pub id: SessionId,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub domain_task: T,   // opaque to the framework
}
```

## Domain Components

### `domain/recipe/agent/prompt.rs`

`RecipePromptBuilder` decides which knowledge sections to inject and their
labels, using the framework's `KnowledgeContext`.

```rust
pub struct RecipePromptBuilder {
    base_prompt: String,
    knowledge: Arc<RecipeKnowledgeStore>,
}

impl RecipePromptBuilder {
    pub fn new(base_prompt: String, knowledge: Arc<RecipeKnowledgeStore>) -> Self;

    /// Build labeled sections for the framework's KnowledgeContext
    pub async fn build_context(
        &self,
        domain: Option<&str>,
    ) -> Result<KnowledgeContext> {
        let mut sections = Vec::new();

        // Site configurations
        let configs = self.knowledge.load_site_configs(domain)?;
        if !configs.is_empty() {
            sections.push(KnowledgeSection {
                label: "Site Configurations".into(),
                json: serde_json::to_string(&configs)?,
                estimated_tokens: estimate_tokens(&configs),
            });
        }

        // Known patterns
        let patterns = self.knowledge.get_patterns()?;
        if !patterns.success_patterns.is_empty() || !patterns.anti_patterns.is_empty() {
            sections.push(KnowledgeSection {
                label: "Known Patterns".into(),
                json: serde_json::to_string(&patterns)?,
                estimated_tokens: estimate_tokens(&patterns),
            });
        }

        // User preferences
        if let Some(user_model) = self.knowledge.load_user_model()? {
            sections.push(KnowledgeSection {
                label: "User Preferences".into(),
                json: serde_json::to_string(&user_model)?,
                estimated_tokens: estimate_tokens(&user_model),
            });
        }

        Ok(KnowledgeContext {
            sections,
            token_budget: self.config.knowledge_budget,
            estimated_tokens: 0, // computed by assembler
        })
    }
}
```

### `domain/recipe/agent/task.rs`

Recipe task state and tracker. The framework session holds this as an opaque
`DomainTask`.

```rust
pub struct RecipeTaskState {
    pub goal: String,
    pub target_count: Option<u32>,
    pub pending_urls: VecDeque<String>,
    pub in_progress_urls: Vec<String>,
    pub completed_urls: Vec<String>,
    pub failed_urls: Vec<FailedUrl>,
    pub scraped_recipe_ids: Vec<String>,
}

pub struct RecipeTaskTracker {
    state: Option<RecipeTaskState>,
}

impl RecipeTaskTracker {
    pub fn new() -> Self;

    /// Start a new task
    pub fn start(&mut self, goal: String, target: Option<u32>);

    /// Record URLs discovered
    pub fn add_urls(&mut self, urls: Vec<String>);

    /// Record URL processing start
    pub fn start_url(&mut self, url: &str);

    /// Record URL processing complete
    pub fn complete_url(&mut self, url: &str, recipe_id: Option<String>);

    /// Record URL processing failed
    pub fn fail_url(&mut self, url: &str, error: &str);

    /// Check if task is complete
    pub fn is_complete(&self) -> bool;

    /// Get progress info for a user message
    pub fn progress_message(&self) -> Option<String>;
}
```

## Binary

### `bin/recipe_agent.rs`

Wires framework and recipe domain together.

```rust
pub fn create_runner(config: Config) -> AgentRunner {
    // Initialize framework storage
    let config_dir = ConfigDir::resolve().expect("Config directory");
    config_dir.init().expect("Initialize directories");

    // Initialize domain storage
    let recipe_db = Arc::new(RecipeDb::open(config_dir.recipes_dir()).expect("Recipe DB"));
    let signal_log = Arc::new(RecipeSignalLog::open(config_dir.signals_dir()).expect("Signal log"));
    let knowledge = Arc::new(
        RecipeKnowledgeStore::open(config_dir.knowledge_dir()).expect("Knowledge store")
    );

    // Initialize learning domain and runner
    let learning_domain = Arc::new(RecipeLearningDomain::new(knowledge.clone()));
    let runner = Arc::new(CompressionRunner::new(
        learning_domain,
        signal_log.clone(),
        CompressionConfig::default(),
        CheckpointStore::open(config_dir.state_dir().join("compression_checkpoint.json")),
    ));

    // Build tool registry
    let mut tools = ToolRegistry::new();
    register_task_tools(&mut tools, knowledge.clone(), recipe_db.clone());
    register_learning_tools(&mut tools, knowledge.clone(), signal_log.clone(), runner.clone());

    // Initialize LLM client
    let llm = LlmClient::new().expect("LLM client");

    AgentRunner::new(llm, tools, config)
}
```

## Configuration

```toml
[profiles.local-llama]
# ... profile fields ...

context_budget = 100000        # Max tokens for context
message_budget = 80000         # Max tokens for messages
knowledge_budget = 8000        # Max tokens for knowledge sections
compression_interval_days = 7  # Auto-compress weekly
signal_retention_days = 30     # Keep signals for 30 days
```

## Tests

- `ToolRegistry` returns correct definitions
- `ContextManager` trims correctly and summarizes old results
- `KnowledgeContext` budgets sections within the token limit
- `RecipePromptBuilder` builds the expected sections
- `RecipeTaskTracker` updates state and reports progress
- Integration test: full scrape cycle

## Dependencies

- `framework/agent` — runner, context, prompt, session
- `framework/tools` — `ToolRegistry`
- `framework/learning` — `CompressionRunner`
- `domain/recipe` — tools, prompt builder, task tracker, learning domain
- `tokio` — async runtime
- `agent-client-protocol` — ACP types

## Size

Estimate: ~600-800 lines

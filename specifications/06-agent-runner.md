# Agent Runner

> **Placement**: This layer spans the framework/domain boundary (see
> [`00-architecture.md`](./00-architecture.md)).
> - **Framework** (`framework/agent/`): `AgentRunner` (tool-calling loop, ACP),
>   `ContextManager` (history trimming, server-authoritative context
>   budgeting, overflow-driven compaction), the
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
loop, the ACP protocol, and conversation context. **The framework does not
count tokens client-side** — it has no model tokenizer, and a chars-per-token
heuristic under-counts the JSON-heavy tool traffic that dominates an agent
loop, lending false confidence to trim decisions. The server's tokenizer is
the single source of truth: the framework reacts to `ContextOverflow` by
compacting history and retrying, and records the server-reported
`usage.prompt_tokens` to bound warm-turn requests from ground truth. A coarse
byte ceiling is used only as an honest pre-filter to avoid sending
obviously-huge history for a wasted round-trip. The recipe domain owns prompt
content and task state. The binary wires them together.

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

            // Coarse byte pre-filter (correctness is reactive — see Overflow
            // below). Cold turn uses config.byte_budget; warm turns derive the
            // ceiling from the last server usage.prompt_tokens.
            let messages = self.context.get_trimmed(self.config.byte_budget);

            // Call LLM
            let stream = self.llm.chat_stream(
                &self.config.profile,
                &messages,
                &self.tools.tool_definitions(),
                cancel.clone(),
            ).await;

            // Process stream...
            let response = self.process_stream(stream, cx, session_id).await;

            // Handle response. `process_stream` records server `usage`
            // (prompt_tokens) on the terminal chunk via context.record_usage.
            match response {
                StreamResult::Text(text) => return StopReason::EndTurn,
                StreamResult::ToolCalls(calls) => {
                    for call in calls {
                        self.execute_tool(&call, cx, session_id, cancel.clone()).await;
                    }
                }
                // Server rejected the prompt as too long: compact and retry.
                // The server's tokenizer is authoritative — this is the
                // correctness path, not the byte pre-filter.
                StreamResult::Overflow => {
                    if !self.context.compact(self.config.keep_last_tool_results) {
                        send_text(cx, session_id, "[context exhausted]");
                        return StopReason::ContextExhausted;
                    }
                    // loop: retry with compacted history
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

Conversation context management. The framework does **not** count tokens
client-side: it has no model tokenizer, and a chars-per-token heuristic
under-counts the JSON-heavy tool traffic that dominates an agent loop — wrong
in the worst direction for trim decisions. The server's tokenizer is the
single source of truth. `ContextManager` therefore exposes three things:

- `get_trimmed(max_bytes)` — a **coarse byte pre-filter** taking the most
  recent entries under a byte ceiling. Its only job is to avoid sending
  obviously-huge history (e.g. a 50 KB tool result) for a wasted round-trip.
  It never claims exact fit.
- `compact(keep_last_n)` — the **correctness** lever. Summarizes old tool
  results to placeholders, then drops the oldest non-system entry once all
  summarizable tool results (those outside the kept window) are summarized.
  Returns whether it made progress (false ⇒ context exhausted, stop). Idempotent.
- `record_usage(prompt_tokens)` — stores the last server-reported
  `usage.prompt_tokens` so the next request's ceiling is set from ground
  truth on warm turns.

```rust
pub struct ContextManager {
    history: Vec<HistoryEntry>,
    /// Coarse byte ceiling — honest heuristic, not an exact fit.
    max_bytes: u64,
    /// Last server-reported prompt token count (warm-turn ground truth).
    last_prompt_tokens: Option<u64>,
}

impl ContextManager {
    pub fn new(max_bytes: u64) -> Self;

    /// Append an entry.
    pub fn push(&mut self, entry: HistoryEntry);

    /// Most-recent-first entries under `max_bytes` (clamped to `self.max_bytes`).
    /// Coarse pre-filter only; correctness comes from reactive `compact`.
    pub fn get_trimmed(&self, max_bytes: u64) -> Vec<ChatMessage>;

    /// Summarize old tool results (keep last `keep_last_n`), then drop the
    /// oldest non-system entry once all summarizable tool results (those
    /// outside the kept window) are summarized. Returns false when no further
    /// progress is possible (context exhausted).
    pub fn compact(&mut self, keep_last_n: usize) -> bool;

    /// Record the server's last reported prompt token count.
    pub fn record_usage(&mut self, prompt_tokens: u64);
}

pub enum HistoryEntry {
    System { text: String },
    User { text: String },
    Assistant { text: Option<String>, tool_calls: Vec<ToolCall> },
    ToolResult {
        tool_call_id: String,
        content: String,
        success: bool,
        is_summarized: bool,
    },
}

impl HistoryEntry {
    /// Coarse byte size (for the pre-filter only — not a token count).
    pub fn size(&self) -> u64;

    /// Collapse a tool result to a short placeholder: `[tool result:
    /// success|failed]`. No-op for other variants and already-summarized
    /// results. Returns whether it changed.
    fn summarize(&mut self) -> bool;
}
```

> **Spec deviation, recorded:** the original design held a `token_count: u64`
> per entry and trimmed by tokens via a `TokenCounter` trait. Dropped because
> the framework has no tokenizer and a chars-per-token estimate under-counts
> JSON tool traffic, lending false confidence to trim decisions. The server's
> tokenizer is authoritative; `compact` reacts to its `ContextOverflow`
> verdict, and `record_usage` tracks its accounting. (The `CharTokenCounter`
> placeholder in PR #49 is superseded by this design.)

### `framework/agent/prompt.rs`

Generic knowledge-context container. The domain fills sections; the framework
budgets and assembles.

```rust
pub struct KnowledgeContext {
    pub sections: Vec<KnowledgeSection>,
    /// Coarse byte ceiling (pre-filter; the server is authoritative on tokens).
    pub byte_budget: u64,
    pub estimated_bytes: u64,
}

pub struct KnowledgeSection {
    pub label: String,
    pub json: String,
    pub estimated_bytes: u64,
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
                estimated_bytes: estimate_bytes(&configs),
            });
        }

        // Known patterns
        let patterns = self.knowledge.get_patterns()?;
        if !patterns.success_patterns.is_empty() || !patterns.anti_patterns.is_empty() {
            sections.push(KnowledgeSection {
                label: "Known Patterns".into(),
                json: serde_json::to_string(&patterns)?,
                estimated_bytes: estimate_bytes(&patterns),
            });
        }

        // User preferences
        if let Some(user_model) = self.knowledge.load_user_model()? {
            sections.push(KnowledgeSection {
                label: "User Preferences".into(),
                json: serde_json::to_string(&user_model)?,
                estimated_bytes: estimate_bytes(&user_model),
            });
        }

        Ok(KnowledgeContext {
            sections,
            byte_budget: self.config.knowledge_byte_budget,
            estimated_bytes: 0, // computed by assembler
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

# Context management. The framework does NOT count tokens client-side — the
# server's tokenizer is authoritative. `context_window` is the model's real
# window (user-supplied, matches the model spec) and bounds warm-turn requests
# via the last server-reported usage.prompt_tokens. The byte budgets are
# coarse pre-filters only (avoid sending obviously-huge history/knowledge for
# a wasted round-trip); correctness on overflow is reactive (compact + retry).
context_window          = 100000  # Model context window (tokens)
completion_reservation  = 4096    # Tokens reserved for the model's reply
byte_budget            = 400_000  # Coarse byte ceiling for history pre-filter
keep_last_tool_results = 4        # Tool results kept in full during compaction
knowledge_byte_budget  = 32_000   # Coarse byte ceiling for knowledge sections
compression_interval_days = 7     # Auto-compress weekly
signal_retention_days = 30        # Keep signals for 30 days
```

## Tests

- `ToolRegistry` returns correct definitions
- `ContextManager` byte-pre-filters most-recent-first, compacts on overflow
  (summarize old tool results, then drop oldest non-system), and records
  server `usage`; `compact` returns false when context is exhausted
- `AgentRunner` compacts and retries on `ContextOverflow`, stops with
  `ContextExhausted` when `compact` makes no progress
- `KnowledgeContext` byte-pre-filters sections within the byte ceiling
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

# Architecture: Framework / Domain Split

## Purpose

This project is a **reusable agent framework with self-improvement (learning)
capabilities**. Recipe scraping is the first domain implementation, not the
product boundary. The framework provides the learning *mechanism*; each domain
provides its own *knowledge* and *content*.

The codebase separates two layers:

- **Framework** (`src/framework/`) — generic infrastructure every learning agent
  needs: storage primitives, the compression loop, tool registry, context and
  token management, the agent runner. It contains no domain symbols.
- **Domain** (`src/domain/<name>/`) — concrete logic for one use case. The recipe
  domain (`src/domain/recipe/`) is the first. A second domain links against the
  framework without modifying it.

## Architectural Principle

> Extract the learning *mechanism* into the framework. Keep the domain
> *knowledge* concrete. Do not abstract domain internals until a second domain
> proves the abstraction.

Infrastructure — storage primitives, compression orchestration, scheduling,
confidence math, tool registry, token budgeting — is domain-agnostic and lives in
the framework. Domain abstractions — signal shape, knowledge shape, pattern
extraction — remain concrete per domain. The Rule of Three governs the latter:
generalize them only after a second domain exercises the same boundary.

The bridge is a small set of traits the domain implements. The framework never
inspects domain internals; it calls trait methods and passes opaque types around.

## Component Placement

| Concern | Layer | Rationale |
|---------|-------|-----------|
| Storage file I/O (JSONL, JSON, append-only) | Framework | Pure serialization mechanics |
| Config dir, session store, checkpoint store | Framework | Domain-agnostic paths and rotation |
| Compression loop orchestration | Framework | aggregate → extract → update → prune is reusable |
| Compression scheduling (time/count triggers) | Framework | Generic thresholds |
| Confidence scoring (Wilson score) | Framework | Pure statistics |
| Tool registry, `Tool` trait | Framework | Required by any agent |
| Context manager, token budgeting, summarization | Framework | Conversation trimming is universal |
| Knowledge-context container + prompt assembly | Framework (mechanism) | Token-budgeted section assembly is generic |
| Agent runner (tool-calling loop, ACP) | Framework | Protocol and loop, no domain logic |
| `Recipe`, `SiteConfig`, `UserModel`, `Patterns` | Domain | Recipe-shaped by nature |
| `RecipeSignal` / `RecipeSignalType` | Domain | Meaningful events are domain-specific |
| Pattern extraction logic (`infer_action`) | Domain | `no_ingredients → TryAlternativeMethod` is recipe logic |
| Signal aggregation (by method, common errors) | Domain | Aggregation keys are domain concepts |
| Task state (pending/completed URLs, recipe count) | Domain | Scraping workflow is domain-specific |
| Fetch / parse / save / search tools | Domain | Domain operations |
| Recipe prompt content (sections, labels) | Domain | What to inject is domain knowledge |

## Recipe Domain

The recipe domain is the first implementation of the framework's learning
contracts. It provides concrete models, storage built on framework primitives, a
`LearningDomain` implementation, domain tools, and recipe prompt/task logic.

### Models (`domain/recipe/models/`)

All recipe types are concrete. No generic `Knowledge` trait wraps them.

```rust
// domain/recipe/models/recipe.rs
pub struct Recipe { name, ingredients, instructions, times, servings, difficulty, ... }
pub enum Difficulty { Easy, Medium, Hard }
pub enum ParseMethod { SchemaOrg, Microdata, Selectors, Heuristic }

// domain/recipe/models/site_config.rs
pub struct SiteConfig {
    domain: String,
    preferred_method: ParseMethod,
    selectors: Selectors,         // per recipe field (title, ingredients, ...)
    rate_limit_ms: u64,
    requires_js: bool,
    skip_patterns: Vec<String>,
    stats: SiteStats,
}

// domain/recipe/models/user_model.rs
pub struct UserModel {
    user_id: String,
    max_prep_time_minutes: Option<u32>,
    preferred_difficulty: Option<Difficulty>,
    dietary_restrictions: Vec<String>,
    sample_size: u32,
    confidence: f64,
}

// domain/recipe/models/patterns.rs
pub struct RecipePatterns { success_patterns: Vec<SuccessPattern>, anti_patterns: Vec<AntiPattern> }
pub struct AntiPattern { description, sites, failure_rate, action: AntiPatternAction, ... }
pub enum AntiPatternAction { SkipUrl, TryAlternativeMethod, SlowDown, UseHeadlessBrowser, LogWarning }

// domain/recipe/models/signal.rs
pub struct RecipeSignal { id, signal_type: RecipeSignalType, domain, url, recipe_id, ... }
pub enum RecipeSignalType {
    ParseSuccess { method: ParseMethod, time_ms: u64 },
    ParseFailure { method: ParseMethod, error: String, attempted_methods: Vec<ParseMethod> },
    RecipeSaved { recipe_id, has_image, ingredient_count },
    RecipeDeleted { recipe_id, reason },
    ExplicitFeedback { feedback, sentiment },
    RateLimitHit { wait_time_ms },
    ConfigUpdated { domain, field, old_value, new_value },
}
```

### Storage (`domain/recipe/storage/`)

Domain stores define the domain's data types and delegate file I/O to framework
primitives (`DailyLog<T>`, `JsonStore<T>`, `AppendOnlyDb<T>`).

```rust
// domain/recipe/storage/recipe_db.rs
pub struct RecipeDb { inner: AppendOnlyDb<Recipe> }   // + search index
pub struct RecipeSignalLog { inner: DailyLog<RecipeSignal> }

// domain/recipe/storage/knowledge_store.rs
pub struct RecipeKnowledgeStore {
    configs: JsonStore<SiteConfig>,       // keyed by domain
    users: JsonStore<UserModel>,          // keyed by user_id
    patterns: JsonStore<RecipePatterns>,
    cache: RwLock<RecipeKnowledgeCache>,  // TTL + RwLock, domain-owned
}
```

The cache (RwLock + TTL + refresh) is heterogeneous across three knowledge types
and therefore belongs to the domain store, not a generic primitive.

### Learning (`domain/recipe/learning/`)

`RecipeLearningDomain` implements the framework's `LearningDomain` trait. It owns
the three compression stages that are recipe-specific: aggregation, pattern
extraction (including `infer_action`), and knowledge updates.

```rust
// domain/recipe/learning/domain.rs
pub struct RecipeLearningDomain {
    knowledge: Arc<RecipeKnowledgeStore>,
    config: CompressionConfig,   // from framework
}

impl LearningDomain for RecipeLearningDomain {
    type Signal = RecipeSignal;
    type Pattern = RecipePattern;        // success or anti, tagged
    type Stats = RecipeAggregatedStats;

    fn aggregate(&self, signals: &[RecipeSignal]) -> RecipeAggregatedStats {
        // Group by domain and ParseMethod; count successes/failures;
        // track common errors.
    }

    fn extract_patterns(&self, stats: &RecipeAggregatedStats) -> Vec<RecipePattern> {
        // Best method per domain (success_rate >= threshold) -> SuccessPattern.
        // High-failure errors -> AntiPattern with infer_action(error).
        // Confidence via the framework's Wilson-score utility.
    }

    fn update_knowledge(&self, patterns: &[RecipePattern]) -> u64 {
        // Update SiteConfig.preferred_method where confidence is high;
        // merge new patterns into RecipePatterns without duplicates.
    }
}

impl RecipeLearningDomain {
    /// Maps parse errors to actions. Recipe logic; does not live in the framework.
    fn infer_action(&self, error: &str) -> AntiPatternAction {
        match error {
            e if e.contains("timeout")        => AntiPatternAction::SlowDown,
            e if e.contains("no_ingredients") => AntiPatternAction::TryAlternativeMethod,
            e if e.contains("/video")         => AntiPatternAction::SkipUrl,
            e if e.contains("javascript")     => AntiPatternAction::UseHeadlessBrowser,
            _                                  => AntiPatternAction::LogWarning,
        }
    }
}
```

`RecipeAggregatedStats` implements the framework's `DomainStats` trait (exposing
`total_count()`); its rich fields (`by_method`, `common_errors`) are internal to
the domain.

### Tools (`domain/recipe/tools/`)

Task tools (`fetch`, `parse`, `save`, `search`) and learning tools
(`update_config`, `record_signal`, `get_knowledge`, `compress_signals`) implement
the framework's `Tool` trait and contain recipe logic. `CompressSignalsTool`
delegates to a `CompressionRunner<RecipeLearningDomain>` and contains no
compression logic of its own.

### Agent (`domain/recipe/agent/`)

- `RecipePromptBuilder` decides which knowledge sections to inject (site configs,
  patterns, user preferences) and their labels, using the framework's
  `KnowledgeContext` container and `PromptAssembler`.
- `RecipeTaskTracker` tracks scraping progress (pending/in-progress/completed/
  failed URLs, recipe count, target). The framework session holds it as an opaque
  `DomainTask`.

## Framework Layer

The framework is generic and reusable. It contains no recipe symbols.

### Storage (`framework/storage/`)

```rust
pub struct ConfigDir { path: PathBuf }                 // resolve $DO_SOMETHING_CONFIG, init dirs
pub struct DailyLog<T: Serde> { dir: PathBuf }         // append, read_range, prune, count
pub struct JsonStore<T: Serde> { path: PathBuf }       // load/save single object or keyed map
pub struct AppendOnlyDb<T: Serde> { dir: PathBuf }     // append + index
pub struct SessionStore<S: Serde> { dir: PathBuf }     // save/load/list domain session state
pub struct CheckpointStore { path: PathBuf }           // compression checkpoint

pub trait StorageError: Error + Send + Sync { ... }    // generic error type
```

Domain-specific storage traits (`RecipesStorage`, `SignalStorage`,
`KnowledgeStorage`) name domain types (`Recipe`, `Signal`, `SiteConfig`) and
therefore live in the domain, not the framework.

### Learning (`framework/learning/`)

```rust
/// A domain's aggregated statistics (opaque to the framework beyond its count).
pub trait DomainStats {
    fn total_count(&self) -> u64;
}

/// What a domain provides so the framework can run the learning loop.
/// The framework never inspects Signal / Pattern / Stats internals.
pub trait LearningDomain: Send + Sync {
    type Signal: Serialize + DeserializeOwned + Clone + Send + Sync;
    type Pattern: Serialize + DeserializeOwned + Clone + Send + Sync;
    type Stats: DomainStats;

    fn aggregate(&self, signals: &[Self::Signal]) -> Self::Stats;
    fn extract_patterns(&self, stats: &Self::Stats) -> Vec<Self::Pattern>;
    fn update_knowledge(&self, patterns: &[Self::Pattern]) -> u64;
}

/// Generic compression loop. Reusable across domains.
pub struct CompressionRunner<D: LearningDomain> {
    domain: Arc<D>,
    signal_log: Arc<DailyLog<D::Signal>>,
    config: CompressionConfig,
    checkpoint: CheckpointStore,
}

impl<D: LearningDomain> CompressionRunner<D> {
    pub fn run(&self, days: u32) -> Result<CompressionStats> {
        let signals  = self.signal_log.read_range(from, to)?;          // generic
        let stats    = self.domain.aggregate(&signals);                 // domain
        let patterns = self.domain.extract_patterns(&stats);            // domain
        let updated  = self.domain.update_knowledge(&patterns);         // domain
        let pruned   = self.signal_log.prune(self.config.retention)?;   // generic
        self.checkpoint.save_now()?;                                    // generic
        CompressionStats {
            signals_processed: stats.total_count(),
            patterns_extracted: patterns.len(),
            configs_updated: updated,
            signals_pruned: pruned,
            ..
        }
    }

    pub fn should_run(&self) -> Result<bool> { /* time + total-count triggers — generic */ }
}

pub struct CompressionConfig { min_sample_size, confidence_threshold, lookback_days, retention_days, min_success_rate }
pub struct CompressionStats  { signals_processed, patterns_extracted, configs_updated, signals_pruned, time_ms, completed_at }

// Pure statistics, reusable:
pub fn wilson_score(successes: u64, total: u64) -> f64;
pub fn sample_confidence(n: u64) -> f64;
```

The runner owns the loop, scheduling, pruning, checkpointing, and stats. The
domain owns what signals mean, what patterns look like, and how knowledge
updates. This is the most important boundary in the system.

### Tools (`framework/tools/`)

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, args: Value) -> ToolResult;
}
pub struct ToolRegistry { tools: HashMap<String, Arc<dyn Tool>> }
pub struct ToolDef { kind, function: { name, description, parameters } }
pub enum ToolResult { Success(Value), Error(String), PermissionRequired { .. } }
```

### Agent (`framework/agent/`)

```rust
pub struct AgentRunner { llm, tools: Arc<ToolRegistry>, context: ContextManager, config }
pub struct ContextManager { history: Vec<HistoryEntry>, max_tokens }   // trim, summarize — generic
pub struct HistoryEntry { role, content, tool_calls, tool_result, token_count }

// Generic knowledge-context container. The domain fills sections; the framework budgets.
pub struct KnowledgeContext {
    pub sections: Vec<KnowledgeSection>,   // { label, json, estimated_tokens }
    pub token_budget: u64,
    pub estimated_tokens: u64,
}
pub struct KnowledgeSection { label: String, json: String, estimated_tokens: u64 }

pub trait PromptAssembler {
    fn assemble(&self, ctx: &KnowledgeContext) -> String;   // generic layout within budget
}
```

`KnowledgeContext` holds labeled JSON sections, not `site_configs`/`patterns`/
`user_model` fields. The domain's `RecipeKnowledgeStore::load_for_context`
builds sections ("Site Configurations", "Known Patterns", "User Preferences");
the framework assembles them into the prompt within the token budget.

Task state (pending URLs, scraped recipe count) is not in the framework. The
framework session stores an opaque `DomainTask: Serialize + DeserializeOwned`;
the recipe domain supplies `RecipeTaskState`.

## Module Structure

```
src/
├── framework/                      # Generic, reusable
│   ├── storage/
│   │   ├── config_dir.rs
│   │   ├── daily_log.rs            # DailyLog<T>
│   │   ├── json_store.rs           # JsonStore<T>
│   │   ├── append_only_db.rs       # AppendOnlyDb<T>
│   │   ├── session_store.rs        # SessionStore<S>
│   │   ├── checkpoint.rs
│   │   └── error.rs
│   ├── learning/
│   │   ├── domain.rs               # LearningDomain, DomainStats traits
│   │   ├── runner.rs               # CompressionRunner<D>
│   │   ├── scheduler.rs            # should_run
│   │   ├── stats.rs                # CompressionConfig, CompressionStats
│   │   └── confidence.rs           # wilson_score, sample_confidence
│   ├── tools/
│   │   ├── traits.rs               # Tool, ToolResult
│   │   └── registry.rs             # ToolRegistry
│   └── agent/
│       ├── runner.rs
│       ├── context.rs              # ContextManager, HistoryEntry
│       ├── prompt.rs               # KnowledgeContext, PromptAssembler
│       └── session.rs              # SessionState (generic + opaque DomainTask)
│
├── domain/
│   └── recipe/                     # First domain implementation
│       ├── models/
│       │   ├── recipe.rs
│       │   ├── site_config.rs
│       │   ├── user_model.rs
│       │   ├── patterns.rs
│       │   └── signal.rs
│       ├── storage/
│       │   ├── recipe_db.rs        # RecipeDb (over AppendOnlyDb<Recipe>)
│       │   ├── signal_log.rs       # RecipeSignalLog (over DailyLog<RecipeSignal>)
│       │   └── knowledge_store.rs  # RecipeKnowledgeStore (over JsonStore<...>)
│       ├── learning/
│       │   ├── domain.rs           # RecipeLearningDomain: impl LearningDomain
│       │   ├── aggregate.rs
│       │   ├── extract.rs          # infer_action lives here
│       │   └── update.rs
│       ├── tools/
│       │   ├── task/               # fetch, parse, save, search
│       │   └── learning/           # update_config, record_signal, get_knowledge, compress
│       └── agent/
│           ├── prompt.rs           # RecipePromptBuilder (fills KnowledgeContext)
│           └── task.rs             # RecipeTaskState, RecipeTaskTracker
│
└── bin/
    └── recipe_agent.rs             # wires framework + recipe domain together
```

A second domain (e.g. `domain/docs/`) mirrors `domain/recipe/` and reuses all of
`framework/` without modification.

## Build Order

The framework is built bottom-up so each layer is testable before the next.

1. **Framework storage primitives** — `ConfigDir`, `DailyLog<T>`, `JsonStore<T>`,
   `AppendOnlyDb<T>`, `SessionStore<S>`, `CheckpointStore`, error types.
2. **Recipe domain models and storage** — concrete types and domain stores built
   over the framework primitives.
3. **Framework learning contracts** — `LearningDomain`, `DomainStats`,
   `CompressionRunner<D>`, scheduler, `CompressionConfig`/`CompressionStats`,
   confidence utilities. Unit-tested with a toy domain.
4. **Recipe learning domain** — `RecipeLearningDomain` implementing the trait
   (`aggregate`, `extract_patterns` + `infer_action`, `update_knowledge`).
5. **Framework tools and agent** — `Tool` trait, `ToolRegistry`,
   `ContextManager`, `KnowledgeContext`/`PromptAssembler`, `AgentRunner`.
6. **Recipe tools and agent** — task tools, learning tools,
   `RecipePromptBuilder`, `RecipeTaskState`.
7. **Binary** — `bin/recipe_agent.rs` wires framework and recipe domain for an
   end-to-end scrape → signal → compress → knowledge-update cycle.

Do not generalize `Signal`, `Knowledge`, or `Pattern` shapes beyond the
`LearningDomain` associated types until a second domain validates the boundaries.

## Testing Strategy

- **Framework**: unit tests use a toy domain (`ToyLearningDomain` with trivial
  signal/pattern types) to prove the loop, scheduling, and budgeting are reusable
  without recipe types. This is the key test that the split is real.
- **Domain**: unit tests for `RecipeLearningDomain` (aggregation, `infer_action`,
  knowledge updates), recipe tools, and `RecipeKnowledgeStore` caching.
- **Integration**: the binary wires framework and recipe domain for an
  end-to-end scrape → signal → compress → knowledge-update cycle.

## Code Style & Guidelines

1. Test coverage for every public function; prefer unit tests with fakes/mocks.
2. Dependency injection via traits, not concrete types.
3. Test behavior and each branch, not line coverage.
4. `cargo clippy --all-targets --all-features -- -D warnings` before committing.

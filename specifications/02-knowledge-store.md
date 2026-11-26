# Knowledge Store

> **Placement**: Domain component (see [`00-architecture.md`](./00-architecture.md)).
> The store, its cache (RwLock + TTL), `SiteConfig`, `UserModel`, and
> `RecipePatterns` all live in `domain/recipe/` (models + storage). They are
> concrete recipe types — no generic `Knowledge` trait wraps them.
>
> The *mechanism* of context assembly lives in the framework: `KnowledgeContext`
> holds a list of labeled `KnowledgeSection` blobs with a token budget
> (`framework/agent/prompt.rs`). The domain's `load_for_context` fills those
> sections.

## Code Style & Guidelines

1. **Test coverage**: Every public function must have test coverage
2. **Unit tests**: Prefer unit tests over integration tests. Use mocks/fakes for dependencies
3. **Dependency injection**: Accept traits as parameters, not concrete types
4. **Relevant tests**: Test behavior and edge cases, not just line coverage. Test each branch
5. **Linting**: Run `cargo clippy --all-targets --all-features -- -D warnings` before committing

## Overview

The knowledge layer provides read/write access to site configs, user models, and
discovered patterns. It sits between the storage layer and the tools, and owns
the in-memory cache that makes knowledge reads cheap.

## Components

### `domain/recipe/storage/knowledge_store.rs`

The knowledge store wraps the framework `JsonStore` primitives and adds a
read-through cache.

```rust
pub struct RecipeKnowledgeStore {
    configs: JsonStore<SiteConfig>,
    users: JsonStore<UserModel>,
    patterns: JsonStore<RecipePatterns>,
    cache: RwLock<RecipeKnowledgeCache>,
}

#[derive(Default)]
struct RecipeKnowledgeCache {
    site_configs: HashMap<String, SiteConfig>,
    user_models: HashMap<String, UserModel>,
    patterns: Option<RecipePatterns>,
    last_refresh: Option<DateTime<Utc>>,
}

impl RecipeKnowledgeStore {
    pub fn new(
        configs: JsonStore<SiteConfig>,
        users: JsonStore<UserModel>,
        patterns: JsonStore<RecipePatterns>,
    ) -> Self;

    /// Get site config, falling back to default if not found
    pub fn get_site_config(&self, domain: &str) -> Result<SiteConfig>;

    /// Update site config (validates first)
    pub fn update_site_config(&self, config: &SiteConfig) -> Result<()>;

    /// Get user model, creating default if not exists
    pub fn get_user_model(&self, user_id: &str) -> Result<UserModel>;

    /// Update user model
    pub fn update_user_model(&self, model: &UserModel) -> Result<()>;

    /// Get current patterns
    pub fn get_patterns(&self) -> Result<RecipePatterns>;

    /// Update patterns
    pub fn update_patterns(&self, patterns: &RecipePatterns) -> Result<()>;

    /// Load knowledge context for LLM prompt injection
    pub fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext>;

    /// Invalidate cache
    pub fn refresh(&self) -> Result<()>;
}
```

### `domain/recipe/models/site_config.rs`

Site configuration with defaults.

```rust
impl SiteConfig {
    /// Create default config for a domain
    pub fn default_for_domain(domain: &str) -> Self;

    /// Merge with fallback config
    pub fn merge_with_fallback(&self, fallback: &SiteConfig) -> Self;

    /// Validate config structure
    pub fn validate(&self) -> Result<()>;

    /// Get selector for a field, falling back to default
    pub fn get_selector(&self, field: &str) -> Option<&str>;
}

/// Default site configs for known recipe sites
pub mod defaults {
    pub fn allrecipes() -> SiteConfig;
    pub fn foodnetwork() -> SiteConfig;
    pub fn tasty() -> SiteConfig;
    // ... other common sites
}
```

### `domain/recipe/models/user_model.rs`

User preference model.

```rust
impl UserModel {
    /// Create empty model for a user
    pub fn new(user_id: &str) -> Self;

    /// Infer preference from a kept recipe
    pub fn record_recipe_kept(&mut self, recipe: &Recipe);

    /// Infer preference from a deleted recipe
    pub fn record_recipe_deleted(&mut self, recipe: &Recipe);

    /// Update confidence based on sample size
    pub fn update_confidence(&mut self);

    /// Check if recipe matches preferences
    pub fn matches_preferences(&self, recipe: &Recipe) -> f64;
}
```

### `domain/recipe/models/patterns.rs`

Pattern discovery and matching.

```rust
impl RecipePatterns {
    /// Check if URL matches any anti-pattern
    pub fn matches_anti_pattern(&self, url: &str, domain: &str) -> Option<&AntiPattern>;

    /// Get applicable success patterns for a domain
    pub fn get_success_patterns(&self, domain: &str) -> Vec<&SuccessPattern>;

    /// Add a new pattern (called by compression)
    pub fn add_success_pattern(&mut self, pattern: SuccessPattern);

    /// Add a new anti-pattern (called by compression)
    pub fn add_anti_pattern(&mut self, pattern: AntiPattern);
}

impl AntiPattern {
    /// Check if this anti-pattern matches a URL
    pub fn matches(&self, url: &str, domain: &str) -> bool;
}

impl SuccessPattern {
    /// Check if this pattern applies to a situation
    pub fn applies(&self, domain: &str, method: ParseMethod) -> bool;
}
```

### `domain/recipe/agent/prompt.rs`

The domain builds labeled `KnowledgeSection`s; the framework assembles them into
the prompt within the token budget.

```rust
impl RecipeKnowledgeStore {
    /// Build labeled sections for the framework's KnowledgeContext
    pub fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext>;
}
```

## Context Injection Strategy

The knowledge context is injected into the system prompt as labeled sections:

```
You are a recipe scraping agent with the following learned knowledge:

## Site Configurations
<JSON of relevant site configs>

## Known Patterns
<JSON of success/anti-patterns>

## User Preferences
<JSON of user model>
```

The token budget is managed by:
1. Loading most relevant site config (if domain specified)
2. Loading user preferences (always)
3. Loading patterns that apply (by domain)
4. Truncating if over budget

## Thread Safety

`RecipeKnowledgeStore` uses `RwLock` for caching:
- Reads don't block each other
- Writes are rare (only on learning)
- Cache invalidation on write

## Tests

- `RecipeKnowledgeStore::get_site_config()` returns fallback for unknown domains
- `UserModel::record_recipe_kept()` updates preferences
- `RecipePatterns::matches_anti_pattern()` detects matching patterns
- `load_for_context()` respects the token budget

## Dependencies

- `framework/storage` — `JsonStore<T>` primitive
- `framework/agent` — `KnowledgeContext`, `KnowledgeSection`
- `domain/recipe/models` — `SiteConfig`, `UserModel`, `RecipePatterns`
- `parking_lot` or `std::sync::RwLock` for caching
- `chrono` for timestamps

## Size

Estimate: ~600-800 lines

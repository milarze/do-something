# Storage Layer

> **Placement**: This layer spans the framework/domain boundary (see
> [`00-architecture.md`](./00-architecture.md)).
> - **Framework** (`framework/storage/`): `ConfigDir`, `DailyLog<T>`,
>   `JsonStore<T>`, `AppendOnlyDb<T>`, `SessionStore<S>`, `CheckpointStore`,
>   error types. File I/O mechanics only — no domain types.
> - **Domain** (`domain/recipe/storage/`): `RecipeDb`, `RecipeSignalLog`,
>   `RecipeKnowledgeStore`, and the `RecipesStorage` / `SignalStorage` /
>   `KnowledgeStorage` traits. Domain stores are thin wrappers over the framework
>   primitives.

## Code Style & Guidelines

1. **Test coverage**: Every public function must have test coverage
2. **Unit tests**: Prefer unit tests over integration tests. Use mocks/fakes for dependencies
3. **Dependency injection**: Accept traits as parameters, not concrete types
4. **Relevant tests**: Test behavior and edge cases, not just line coverage. Test each branch
5. **Linting**: Run `cargo clippy --all-targets --all-features -- -D warnings` before committing

## Overview

The storage layer provides file-based persistence and manages the
`~/.do-something` directory structure. Generic serialization mechanics live in
the framework; recipe-typed stores live in the domain and delegate I/O to the
framework primitives.

## Directory Structure

```
~/.do-something/
├── config.json                    # Runtime config (model, api endpoint)
├── knowledge/
│   ├── site_configs/
│   │   ├── allrecipes.com.json
│   │   ├── foodnetwork.com.json
│   │   └── _default.json
│   ├── user_models/
│   │   └── default.json
│   └── patterns/
│       ├── anti_patterns.json
│       └── success_patterns.json
├── recipes/
│   ├── recipes.jsonl
│   └── index.json
├── signals/
│   ├── 2026-06-30.jsonl
│   └── 2026-07-01.jsonl
└── state/
    ├── sessions/
    │   └── sess_abc123.json
    └── compression_checkpoint.json
```

## Framework Components

### `framework/storage/config_dir.rs`

Resolves the config directory path and initializes the directory structure.

```rust
pub struct ConfigDir {
    path: PathBuf,
}

impl ConfigDir {
    /// Resolve config directory: $DO_SOMETHING_CONFIG or ~/.do-something
    pub fn resolve() -> Result<Self>;

    /// Ensure directory structure exists
    pub fn init(&self) -> Result<()>;

    /// Get subdirectory paths
    pub fn knowledge_dir(&self) -> PathBuf;
    pub fn recipes_dir(&self) -> PathBuf;
    pub fn signals_dir(&self) -> PathBuf;
    pub fn state_dir(&self) -> PathBuf;
}
```

### `framework/storage/daily_log.rs`

Generic append-only daily JSONL log, parameterized by record type.

```rust
pub struct DailyLog<T: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> DailyLog<T> {
    pub fn open(dir: PathBuf) -> Result<Self>;

    /// Append a record to today's log
    pub fn append(&self, record: &T) -> Result<()>;

    /// Read records from a date range
    pub fn read_range(&self, from: NaiveDate, to: NaiveDate) -> Result<Vec<T>>;

    /// Count records for a domain key in the last N days
    pub fn count_for_key(&self, key: &str, days: u32) -> Result<u64>;

    /// Prune logs older than retention period
    pub fn prune(&self, older_than_days: u32) -> Result<u64>;

    /// Read records for a specific date
    pub fn read_date(&self, date: NaiveDate) -> Result<Vec<T>>;

    /// List available log dates
    pub fn available_dates(&self) -> Result<Vec<NaiveDate>>;
}
```

### `framework/storage/json_store.rs`

Generic JSON file store for a single object or a keyed map of objects.

```rust
pub struct JsonStore<T: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> JsonStore<T> {
    pub fn open(dir: PathBuf) -> Result<Self>;

    /// Load a single object (e.g. patterns file)
    pub fn load(&self) -> Result<T>;

    /// Save a single object
    pub fn save(&self, value: &T) -> Result<()>;

    /// Load a keyed object by name
    pub fn get(&self, key: &str) -> Result<Option<T>>;

    /// Save a keyed object by name
    pub fn put(&self, key: &str, value: &T) -> Result<()>;

    /// Delete a keyed object
    pub fn delete(&self, key: &str) -> Result<()>;

    /// List all keys
    pub fn list(&self) -> Result<Vec<String>>;
}
```

### `framework/storage/append_only_db.rs`

Generic append-only database with an index, parameterized by record type.

```rust
pub struct AppendOnlyDb<T: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> AppendOnlyDb<T> {
    pub fn open(dir: PathBuf) -> Result<Self>;

    /// Append a record, return its ID
    pub fn insert(&self, record: &T) -> Result<String>;

    /// Get a record by ID
    pub fn get(&self, id: &str) -> Result<Option<T>>;

    /// Count total records
    pub fn count(&self) -> Result<u64>;

    /// Iterate all records
    pub fn all(&self) -> Result<Vec<T>>;

    /// Rebuild the index
    pub fn rebuild_index(&self) -> Result<()>;
}
```

### `framework/storage/session_store.rs`

Generic session state persistence, parameterized by session type.

```rust
pub struct SessionStore<S: Serialize + DeserializeOwned> {
    dir: PathBuf,
    _marker: PhantomData<S>,
}

impl<S: Serialize + DeserializeOwned> SessionStore<S> {
    pub fn open(dir: PathBuf) -> Result<Self>;

    pub fn save(&self, session: &S) -> Result<()>;
    pub fn load(&self, id: &str) -> Result<Option<S>>;
    pub fn list(&self) -> Result<Vec<String>>;
    pub fn delete(&self, id: &str) -> Result<()>;
    pub fn exists(&self, id: &str) -> Result<bool>;
    pub fn latest(&self) -> Result<Option<S>>;
    pub fn count(&self) -> Result<usize>;
    pub fn clear(&self) -> Result<u64>;
}
```

### `framework/storage/checkpoint.rs`

Compression checkpoint store.

```rust
pub struct CheckpointStore {
    path: PathBuf,
}

impl CheckpointStore {
    pub fn open(path: PathBuf) -> Result<Self>;
    pub fn load(&self) -> Result<Checkpoint>;
    pub fn save(&self, checkpoint: &Checkpoint) -> Result<()>;
}

pub struct Checkpoint {
    pub last_run: DateTime<Utc>,
    pub last_domain: Option<String>,
    pub last_stats: Option<CompressionStats>,
}
```

### `framework/storage/error.rs`

Generic storage error type.

```rust
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Config directory not found")]
    NotFound,

    #[error("Invalid path: {0}")]
    InvalidPath(String),
}
```

## Domain Components

### `domain/recipe/storage/traits.rs`

Recipe-typed storage traits. These name domain types and therefore live in the
domain, not the framework.

```rust
pub trait RecipesStorage: Send + Sync {
    fn insert(&self, recipe: &Recipe) -> Result<RecipeId>;
    fn get(&self, id: &RecipeId) -> Result<Option<Recipe>>;
    fn exists_by_url(&self, url: &str) -> Result<bool>;
    fn search(&self, query: &str) -> Result<Vec<RecipeId>>;
    fn count(&self) -> Result<u64>;
    fn all(&self) -> Result<Vec<Recipe>>;
}

pub trait SignalStorage: Send + Sync {
    fn append(&self, signal: &RecipeSignal) -> Result<()>;
    fn read_range(&self, from: NaiveDate, to: NaiveDate) -> Result<Vec<RecipeSignal>>;
    fn count_for_domain(&self, domain: &str, days: u32) -> Result<u64>;
    fn prune(&self, older_than_days: u32) -> Result<u64>;
    fn read_date(&self, date: NaiveDate) -> Result<Vec<RecipeSignal>>;
    fn available_dates(&self) -> Result<Vec<NaiveDate>>;
}

pub trait KnowledgeStorage: Send + Sync {
    fn get_site_config(&self, domain: &str) -> Result<Option<SiteConfig>>;
    fn save_site_config(&self, config: &SiteConfig) -> Result<()>;
    fn list_site_configs(&self) -> Result<Vec<String>>;
    fn delete_site_config(&self, domain: &str) -> Result<()>;
    fn get_user_model(&self, user_id: &str) -> Result<Option<UserModel>>;
    fn save_user_model(&self, model: &UserModel) -> Result<()>;
    fn list_user_models(&self) -> Result<Vec<String>>;
    fn get_patterns(&self) -> Result<RecipePatterns>;
    fn save_patterns(&self, patterns: &RecipePatterns) -> Result<()>;
    fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext>;
}
```

### `domain/recipe/storage/recipe_db.rs`

Append-only recipe storage with a search index, built over `AppendOnlyDb<Recipe>`.

```rust
pub struct RecipeDb {
    inner: AppendOnlyDb<Recipe>,
}

impl RecipeDb {
    pub fn open(dir: PathBuf) -> Result<Self>;

    /// Append a recipe, return its ID
    pub fn insert(&self, recipe: &Recipe) -> Result<RecipeId>;

    /// Get recipe by ID
    pub fn get(&self, id: &RecipeId) -> Result<Option<Recipe>>;

    /// Check if URL already scraped (by content_hash)
    pub fn exists_by_url(&self, url: &str) -> Result<bool>;

    /// Full-text search across recipes
    pub fn search(&self, query: &str) -> Result<Vec<RecipeId>>;

    /// Rebuild search index
    pub fn rebuild_index(&self) -> Result<()>;
}
```

### `domain/recipe/storage/signal_log.rs`

Daily JSONL signal log, built over `DailyLog<RecipeSignal>`.

```rust
pub struct RecipeSignalLog {
    inner: DailyLog<RecipeSignal>,
}

impl RecipeSignalLog {
    pub fn open(dir: PathBuf) -> Result<Self>;

    /// Append a signal to today's log
    pub fn append(&self, signal: &RecipeSignal) -> Result<()>;

    /// Read signals from a date range
    pub fn read_range(&self, from: NaiveDate, to: NaiveDate) -> Result<Vec<RecipeSignal>>;

    /// Count signals for a domain in date range
    pub fn count_for_domain(&self, domain: &str, days: u32) -> Result<u64>;

    /// Prune signals older than retention period
    pub fn prune(&self, older_than_days: u32) -> Result<u64>;
}
```

### `domain/recipe/storage/knowledge_store.rs`

Persistence for site configs, user models, and patterns, built over
`JsonStore<...>`.

```rust
pub struct RecipeKnowledgeStore {
    configs: JsonStore<SiteConfig>,       // keyed by domain
    users: JsonStore<UserModel>,          // keyed by user_id
    patterns: JsonStore<RecipePatterns>,
    cache: RwLock<RecipeKnowledgeCache>,
}

impl RecipeKnowledgeStore {
    pub fn open(dir: PathBuf) -> Result<Self>;

    // Site configs
    pub fn get_site_config(&self, domain: &str) -> Result<Option<SiteConfig>>;
    pub fn save_site_config(&self, config: &SiteConfig) -> Result<()>;
    pub fn list_site_configs(&self) -> Result<Vec<String>>;

    // User models
    pub fn get_user_model(&self, user_id: &str) -> Result<Option<UserModel>>;
    pub fn save_user_model(&self, model: &UserModel) -> Result<()>;

    // Patterns
    pub fn get_patterns(&self) -> Result<RecipePatterns>;
    pub fn save_patterns(&self, patterns: &RecipePatterns) -> Result<()>;

    /// Load all knowledge relevant to a domain for context injection
    pub fn load_for_context(&self, domain: Option<&str>) -> Result<KnowledgeContext>;
}
```

## Tests

- `ConfigDir::resolve()` honors `$DO_SOMETHING_CONFIG`
- `DailyLog<T>` creates daily files and reads across a range
- `JsonStore<T>` round-trips single objects and keyed maps
- `AppendOnlyDb<T>` generates unique IDs and rebuilds its index
- `RecipeDb::insert()` generates unique IDs
- `RecipeDb::exists_by_url()` detects duplicates
- `RecipeSignalLog::append()` creates daily files
- `RecipeKnowledgeStore` round-trips all knowledge types

## Dependencies

- `serde`, `serde_json` — serialization
- `chrono` — timestamps and dates
- `uuid` — ID generation
- `thiserror` — error types

## Size

Estimate: ~500-700 lines

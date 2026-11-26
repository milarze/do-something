# Compression Pipeline

> **Placement**: This layer spans the framework/domain boundary and is the most
> important boundary in the system (see [`00-architecture.md`](./00-architecture.md)).
> - **Framework** (`framework/learning/`): the `LearningDomain` and `DomainStats`
>   traits, `CompressionRunner<D>` (loop orchestration), scheduler (`should_run`),
>   `CompressionConfig` / `CompressionStats`, and confidence utilities
>   (`wilson_score`, `sample_confidence`). The framework never inspects domain
>   signal/pattern/stats internals.
> - **Domain** (`domain/recipe/learning/`): `RecipeLearningDomain` implements
>   `LearningDomain` and owns `aggregate`, `extract_patterns` (including the
>   recipe-specific `infer_action`), and `update_knowledge`.

## Code Style & Guidelines

1. **Test coverage**: Every public function must have test coverage
2. **Unit tests**: Prefer unit tests over integration tests. Use mocks/fakes for dependencies
3. **Dependency injection**: Accept traits as parameters, not concrete types
4. **Relevant tests**: Test behavior and edge cases, not just line coverage. Test each branch
5. **Linting**: Run `cargo clippy --all-targets --all-features -- -D warnings` before committing

## Overview

The compression pipeline distills many noisy signals into reliable knowledge. It
runs a four-stage loop — aggregate, extract patterns, update knowledge, prune —
where the loop mechanics are generic and the three domain stages are
recipe-specific.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPRESSION RUNNER <D>                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│   │  AGGREGATE   │    │   EXTRACT    │    │    UPDATE    │           │
│   │  (domain)    │ →  │  PATTERNS    │ →  │  KNOWLEDGE   │           │
│   │              │    │  (domain)    │    │  (domain)    │           │
│   │  signals →   │    │              │    │              │           │
│   │  stats       │    │ patterns     │    │ configs      │           │
│   └──────────────┘    └──────────────┘    └──────────────┘           │
│          │                  │                    │                  │
│          └──────────────────┴────────────────────┘                  │
│                            ↓                                         │
│                    ┌──────────────┐                                 │
│                    │    PRUNE     │  (framework)                    │
│                    │   old data   │                                 │
│                    └──────────────┘                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

The runner drives the loop, scheduling, pruning, checkpointing, and stats. The
domain owns what signals mean, what patterns look like, and how knowledge
updates.

## Framework Components

### `framework/learning/domain.rs`

The contracts a domain implements so the framework can run the loop.

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
```

### `framework/learning/stats.rs`

Configuration and result types.

```rust
#[derive(Clone)]
pub struct CompressionConfig {
    /// Minimum signals required before extraction
    pub min_sample_size: u64,

    /// Confidence threshold for pattern persistence
    pub confidence_threshold: f64,

    /// Days of signals to process
    pub lookback_days: u32,

    /// Retention period for processed signals
    pub retention_days: u32,

    /// Minimum success rate to recommend a method
    pub min_success_rate: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 10,
            confidence_threshold: 0.7,
            lookback_days: 7,
            retention_days: 30,
            min_success_rate: 0.8,
        }
    }
}

pub struct CompressionStats {
    pub signals_processed: u64,
    pub patterns_extracted: usize,
    pub configs_updated: u64,
    pub signals_pruned: u64,
    pub time_ms: u64,
    pub completed_at: DateTime<Utc>,
}
```

### `framework/learning/runner.rs`

Generic compression loop.

```rust
pub struct CompressionRunner<D: LearningDomain> {
    domain: Arc<D>,
    signal_log: Arc<DailyLog<D::Signal>>,
    config: CompressionConfig,
    checkpoint: CheckpointStore,
}

impl<D: LearningDomain> CompressionRunner<D> {
    pub fn new(
        domain: Arc<D>,
        signal_log: Arc<DailyLog<D::Signal>>,
        config: CompressionConfig,
        checkpoint: CheckpointStore,
    ) -> Self;

    /// Run full compression pipeline
    pub fn run(&self, domain: Option<&str>, days: u32) -> Result<CompressionStats> {
        let start = std::time::Instant::now();

        // Stage 1: Aggregate (domain)
        let stats = self.domain.aggregate_from_log(domain, days, &self.signal_log)?;

        // Stage 2: Extract patterns (domain)
        let patterns = self.domain.extract_patterns(&stats)?;

        // Stage 3: Update knowledge (domain)
        let configs_updated = self.domain.update_knowledge(&patterns)?;

        // Stage 4: Prune old signals (framework)
        let pruned = self.signal_log.prune(self.config.retention_days)?;

        Ok(CompressionStats {
            signals_processed: stats.total_count(),
            patterns_extracted: patterns.len(),
            configs_updated,
            signals_pruned: pruned,
            time_ms: start.elapsed().as_millis() as u64,
            completed_at: Utc::now(),
        })
    }
}
```

### `framework/learning/scheduler.rs`

Trigger conditions.

```rust
impl<D: LearningDomain> CompressionRunner<D> {
    pub fn should_run(&self, domain: &str) -> Result<bool> {
        // Check time since last run
        let checkpoint = self.checkpoint.load()?;
        if Utc::now() - checkpoint.last_run > chrono::Duration::days(7) {
            return Ok(true);
        }

        // Check signal count
        let count = self.signal_log.count_for_key(domain, 7)?;
        if count >= 100 {
            return Ok(true);
        }

        Ok(false)
    }
}
```

Compression runs when any condition is met:

| Condition | Threshold |
|-----------|-----------|
| Signal count for domain | > 100 |
| Time since last run | > 7 days |
| Explicit tool call | `compress_signals()` |
| Repeated failure pattern | Detected |

### `framework/learning/confidence.rs`

Pure statistics utilities.

```rust
/// Wilson score interval lower bound for a binomial proportion.
pub fn wilson_score(successes: u64, total: u64) -> f64;

/// Sample-size confidence factor in [0, 1].
pub fn sample_confidence(n: u64) -> f64;
```

## Domain Components

### `domain/recipe/learning/domain.rs`

`RecipeLearningDomain` implements `LearningDomain`.

```rust
pub struct RecipeLearningDomain {
    knowledge: Arc<RecipeKnowledgeStore>,
    config: CompressionConfig,
}

impl LearningDomain for RecipeLearningDomain {
    type Signal = RecipeSignal;
    type Pattern = RecipePattern;
    type Stats = RecipeAggregatedStats;

    fn aggregate(&self, signals: &[RecipeSignal]) -> RecipeAggregatedStats {
        // Group by domain and ParseMethod; count successes/failures;
        // track common errors.
    }

    fn extract_patterns(&self, stats: &RecipeAggregatedStats) -> Vec<RecipePattern> {
        // Best method per domain (success_rate >= threshold) -> SuccessPattern.
        // High-failure errors -> AntiPattern with infer_action(error).
        // Confidence via the framework's wilson_score / sample_confidence.
    }

    fn update_knowledge(&self, patterns: &[RecipePattern]) -> u64 {
        // Update SiteConfig.preferred_method where confidence is high;
        // merge new patterns into RecipePatterns without duplicates.
    }
}
```

### `domain/recipe/learning/aggregate.rs`

Recipe signal aggregation.

```rust
pub struct RecipeAggregatedStats {
    pub domain: String,
    pub by_method: HashMap<ParseMethod, MethodStats>,
    pub total_successes: u64,
    pub total_failures: u64,
    pub common_errors: Vec<ErrorFrequency>,
    pub time_period: (DateTime<Utc>, DateTime<Utc>),
}

impl DomainStats for RecipeAggregatedStats {
    fn total_count(&self) -> u64 { self.total_successes + self.total_failures }
}

pub struct MethodStats {
    pub successes: u64,
    pub failures: u64,
    pub avg_time_ms: f64,
    pub errors: HashMap<String, u64>,
}

pub struct ErrorFrequency {
    pub error: String,
    pub count: u64,
    pub percentage: f64,
}

impl RecipeLearningDomain {
    pub fn aggregate(&self, signals: &[RecipeSignal]) -> RecipeAggregatedStats {
        let mut by_domain: HashMap<String, RecipeAggregatedStats> = HashMap::new();

        for signal in signals {
            let entry = by_domain.entry(signal.domain.clone()).or_default();

            match &signal.signal_type {
                RecipeSignalType::ParseSuccess { method, time_ms } => {
                    entry.record_success(*method, *time_ms);
                }
                RecipeSignalType::ParseFailure { method, error, .. } => {
                    entry.record_failure(*method, error);
                }
                // ... other signal types
            }
        }

        by_domain.into_values().next().unwrap_or_default()
    }
}
```

### `domain/recipe/learning/extract.rs`

Pattern extraction, including `infer_action`.

```rust
pub struct ExtractedPatterns {
    pub success_patterns: Vec<SuccessPattern>,
    pub anti_patterns: Vec<AntiPattern>,
}

impl RecipeLearningDomain {
    pub fn extract_patterns(
        &self,
        stats: &RecipeAggregatedStats,
    ) -> Result<ExtractedPatterns> {
        let mut success_patterns = Vec::new();
        let mut anti_patterns = Vec::new();

        // Skip if not enough data
        if stats.total_count() < self.config.min_sample_size {
            return Ok(ExtractedPatterns::default());
        }

        // Find best parsing method
        if let Some((method, success_rate)) = stats.best_method() {
            if success_rate >= self.config.min_success_rate {
                success_patterns.push(SuccessPattern {
                    description: format!(
                        "{} parsing works well for {}",
                        method, stats.domain
                    ),
                    sites: vec![stats.domain.clone()],
                    success_rate,
                    sample_size: stats.total_count(),
                    confidence: self.compute_confidence(stats),
                });
            }
        }

        // Find anti-patterns (high failure rates)
        for (error, count) in &stats.common_errors {
            if *count > 5 && stats.failure_rate() > 0.5 {
                anti_patterns.push(AntiPattern {
                    description: error.clone(),
                    sites: vec![stats.domain.clone()],
                    failure_rate: stats.failure_rate(),
                    action: self.infer_action(error),
                    sample_size: *count,
                    confidence: self.compute_confidence(stats),
                });
            }
        }

        Ok(ExtractedPatterns { success_patterns, anti_patterns })
    }

    fn compute_confidence(&self, stats: &RecipeAggregatedStats) -> f64 {
        let n = stats.total_count() as f64;
        let p = stats.success_rate();
        let sample_conf = sample_confidence(n as u64);
        sample_conf * self.config.confidence_threshold
    }

    /// Maps parse errors to actions. Recipe logic; lives in the domain.
    fn infer_action(&self, error: &str) -> AntiPatternAction {
        match error {
            e if e.contains("timeout") => AntiPatternAction::SlowDown,
            e if e.contains("no_ingredients") => AntiPatternAction::TryAlternativeMethod,
            e if e.contains("/video") => AntiPatternAction::SkipUrl,
            e if e.contains("javascript") => AntiPatternAction::UseHeadlessBrowser,
            _ => AntiPatternAction::LogWarning,
        }
    }
}
```

### `domain/recipe/learning/update.rs`

Knowledge store updates.

```rust
impl RecipeLearningDomain {
    pub fn update_knowledge(&self, patterns: &ExtractedPatterns) -> Result<u64> {
        let mut configs_updated = 0u64;

        // Update success patterns
        for pattern in &patterns.success_patterns {
            for site in &pattern.sites {
                let mut config = self.knowledge.get_site_config(site)?;

                if pattern.confidence >= self.config.confidence_threshold {
                    if let Some(method) = self.extract_method(&pattern.description) {
                        config.preferred_method = method;
                        config.stats.success_count = pattern.sample_size;
                        self.knowledge.update_site_config(&config)?;
                        configs_updated += 1;
                    }
                }
            }
        }

        // Merge new patterns without duplicates
        let mut existing = self.knowledge.get_patterns()?;

        for pattern in &patterns.success_patterns {
            if !existing.success_patterns.iter().any(|p|
                p.description == pattern.description
            ) {
                existing.success_patterns.push(pattern.clone());
            }
        }

        for pattern in &patterns.anti_patterns {
            if !existing.anti_patterns.iter().any(|p|
                p.description == pattern.description
            ) {
                existing.anti_patterns.push(pattern.clone());
            }
        }

        existing.version += 1;
        self.knowledge.update_patterns(&existing)?;

        Ok(configs_updated)
    }
}
```

## Thread Safety

Compression is expensive, so:
- Runs in a background task
- Uses RwLock on the knowledge store
- Logs progress for observability

## Tests

- **Framework**: `CompressionRunner` runs the loop end-to-end with a
  `ToyLearningDomain` (trivial signal/pattern types), proving the loop,
  scheduling, pruning, and checkpointing are reusable without recipe types.
- **Domain**: `aggregate()` groups signals correctly; `extract_patterns()` finds
  valid patterns; `infer_action()` maps errors to actions; `update_knowledge()`
  merges without duplicates; confidence calculation is correct.

## Dependencies

- `framework/learning` — `LearningDomain`, `CompressionRunner<D>`, confidence utils
- `framework/storage` — `DailyLog<T>`, `CheckpointStore`
- `domain/recipe` — knowledge store, models
- `chrono` — time handling
- `tokio` — async runtime

## Size

Estimate: ~700-900 lines

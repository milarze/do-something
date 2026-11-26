# Implementation Specifications

This directory specifies a **reusable self-improving agent framework** with recipe
scraping as the first domain implementation. Each specification prescribes one
layer of the system.

## Architecture

[`00-architecture.md`](./00-architecture.md) defines the framework/domain split
and is the architectural foundation. Every other specification elaborates a
component of that architecture.

- **Framework** (`src/framework/`) — generic infrastructure any learning agent
  needs: storage primitives, the compression loop, tool registry, context and
  token management, the agent runner.
- **Domain** (`src/domain/recipe/`) — concrete recipe logic: models, recipe
  storage, `RecipeLearningDomain`, recipe tools, recipe prompt and task state.

The rule: extract the learning *mechanism* into the framework; keep the domain
*knowledge* concrete. Domain internals (`Signal` shape, `Knowledge` shape,
`infer_action`) are not abstracted until a second domain proves the abstraction.

## Specifications

| # | Specification | Layer | Prescribes |
|---|---------------|-------|------------|
| 00 | [Architecture](./00-architecture.md) | Both | Framework/domain split, contracts, module structure, build order |
| 01 | [Storage Layer](./01-storage-layer.md) | Both | Generic storage primitives + recipe stores |
| 02 | [Knowledge Store](./02-knowledge-store.md) | Domain | Recipe knowledge: site configs, user models, patterns |
| 03 | [Task Tools](./03-task-tools.md) | Both | `Tool` trait/registry + fetch/parse/save/search |
| 04 | [Learning Tools](./04-learning-tools.md) | Domain | update_config, record_signal, get_knowledge, compress |
| 05 | [Compression Pipeline](./05-compression-pipeline.md) | Both | `CompressionRunner<D>` + `RecipeLearningDomain` |
| 06 | [Agent Runner](./06-agent-runner.md) | Both | `AgentRunner`, context, prompt, task state |

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
│       ├── models/                 # Recipe, SiteConfig, UserModel, Patterns, Signals
│       ├── storage/                # RecipeDb, RecipeSignalLog, RecipeKnowledgeStore
│       ├── learning/               # RecipeLearningDomain: impl LearningDomain
│       │                           #   (aggregate, extract, update, infer_action)
│       ├── tools/
│       │   ├── task/               # fetch, parse, save, search
│       │   └── learning/           # update_config, record_signal, get_knowledge, compress
│       └── agent/                  # RecipePromptBuilder, RecipeTaskState
│
└── bin/
    └── recipe_agent.rs             # wires framework + recipe domain together
```

A second domain (e.g. `domain/docs/`) mirrors `domain/recipe/` and reuses all of
`framework/` without modification.

## Prerequisites

- Rust 2024 edition
- Dependencies from `Cargo.toml`

## Testing Strategy

- **Framework**: unit tests use a toy domain (`ToyLearningDomain`) to prove the
  loop, scheduling, and budgeting are reusable without recipe types.
- **Domain**: unit tests for `RecipeLearningDomain`, recipe tools, and
  `RecipeKnowledgeStore` caching.
- **Integration**: end-to-end scrape → signal → compress → knowledge-update cycle.

Each layer has unit tests, integration tests with the layers below it, and test
fixtures (sample HTML, signals, configs).

## Code Style & Guidelines

1. **Test coverage**: every change has proper test coverage
2. **Unit tests**: prefer unit tests over integration tests; use mocks/fakes for
   dependencies
3. **Dependency injection**: accept traits as parameters, not concrete types
4. **Relevant tests**: test behavior and each branch, not line coverage
5. **Linting**: `cargo clippy --all-targets --all-features -- -D warnings`

## Notes

- Build one specification at a time, in [build order](./00-architecture.md#build-order)
- Run `cargo clippy --all-targets --all-features -- -D warnings` after each step

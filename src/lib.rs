//! Do-something: A self-improving agent framework.
//!
//! This project separates two layers:
//!
//! - **Framework** (`src/framework/`) — generic infrastructure every learning agent
//!   needs: storage primitives, compression loop, tool registry, context management.
//!   Contains no domain symbols.
//!
//! - **Domain** (`src/domain/`) — concrete logic for one use case.
//!   The recipe domain (`src/domain/recipe/`) is the first implementation.
//!
//! # Architecture
//!
//! The framework provides generic storage primitives (`DailyLog<T>`, `JsonStore<T>`,
//! `AppendOnlyDb<T>`, `SessionStore<S>`). Domain stores wrap these and add
//! domain-specific functionality.

pub mod domain;
pub mod framework;

// Re-export commonly used types for convenience
pub use domain::recipe::{
    ParseMethod, Recipe, RecipeAggregatedStats, RecipeDb, RecipeId, RecipeKnowledgeStore,
    RecipeLearningDomain, RecipePattern, RecipePatterns, RecipeSignal, RecipeSignalLog,
    RecipeSignalType, SiteConfig, UserModel,
};
pub use framework::learning::{CompressionConfig, CompressionRunner, DomainStats, LearningDomain};
pub use framework::storage::{ConfigDir, StorageError};

// Legacy modules - will be migrated or removed
pub mod agent;
pub mod config;
pub mod llm;
pub mod tools;

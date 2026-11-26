//! Recipe domain implementation.
//!
//! This is the first domain implementation of the framework's
//! learning contracts. It provides:
//!
//! - Concrete models (`Recipe`, `SiteConfig`, `UserModel`, `RecipePatterns`)
//! - Domain stores wrapping framework primitives
//! - Domain-specific signal types
//! - Task tools: fetch, parse, save, search

pub mod learning;
pub mod models;
pub mod paths;
pub mod storage;
pub mod tools;

pub use learning::{RecipeAggregatedStats, RecipeLearningDomain, RecipePattern};
pub use models::*;
pub use paths::RecipePaths;
pub use storage::*;
pub use tools::learning::{
    CompressSignalsTool, CompressionRunnerPort, ConfigBackupStore, FileConfigBackupStore,
    GetKnowledgeTool, RecordSignalTool, UpdateConfigTool,
};
pub use tools::task::{FetchHtmlTool, ParseRecipeTool, RateLimiter, SaveRecipeTool, SearchTool};

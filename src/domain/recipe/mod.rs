//! Recipe domain implementation.
//!
//! This is the first domain implementation of the framework's
//! learning contracts. It provides:
//!
//! - Concrete models (`Recipe`, `SiteConfig`, `UserModel`, `RecipePatterns`)
//! - Domain stores wrapping framework primitives
//! - Domain-specific signal types

pub mod learning;
pub mod models;
pub mod paths;
pub mod storage;

pub use learning::{RecipeAggregatedStats, RecipeLearningDomain, RecipePattern};
pub use models::*;
pub use paths::RecipePaths;
pub use storage::*;

//! Recipe learning domain.
//!
//! Implements the framework's [`LearningDomain`] trait for recipe scraping.
//! Owns signal aggregation, pattern extraction (including `infer_action`),
//! and knowledge updates.

pub mod aggregate;
pub mod domain;
pub mod extract;
pub mod update;

pub use aggregate::{ErrorFrequency, MethodStats, RecipeAggregatedStats};
pub use domain::RecipeLearningDomain;
pub use extract::RecipePattern;

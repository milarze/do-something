//! Knowledge management layer for the recipe scraping agent.
//!
//! This module provides read/write access to learned knowledge:
//! - Site-specific parsing configurations
//! - User preference models
//! - Discovered patterns (success and anti-patterns)
//!
//! # Architecture
//!
//! The knowledge layer sits between the storage layer and tools:
//! - Caches frequently accessed data in memory
//! - Validates changes before persisting
//! - Assembles knowledge for LLM prompt injection
//!
//! # Thread Safety
//!
//! `KnowledgeStore` uses `RwLock` for caching:
//! - Multiple readers don't block each other
//! - Writers are rare (only on learning updates)
//! - Cache invalidation happens on writes

pub mod context;
pub mod patterns;
pub mod site_config;
pub mod store;
pub mod user_model;

pub use context::KnowledgeContextAssembler;
pub use patterns::PatternMatcher;
pub use site_config::{SiteConfigManager, defaults};
pub use store::KnowledgeStore;
pub use user_model::UserModelManager;

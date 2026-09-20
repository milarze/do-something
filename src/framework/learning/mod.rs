//! Framework learning layer: generic compression loop and contracts.
//!
//! The framework owns the loop mechanics (aggregate → extract → update →
//! prune), scheduling, checkpointing, and confidence utilities. Domain
//! stages are supplied via the [`LearningDomain`] trait.

pub mod confidence;
pub mod domain;
pub mod runner;
pub mod scheduler;
pub mod stats;

pub use confidence::{sample_confidence, wilson_score};
pub use domain::{DomainStats, LearningDomain};
pub use runner::CompressionRunner;
pub use stats::CompressionConfig;

//! Data models for recipe scraping and self-improving agent.
//!
//! This module defines all serializable structures used for:
//! - Recipe data and storage
//! - Knowledge store (site configs, user models, patterns)
//! - Signal logging and compression
//! - Agent state persistence

use serde::{Deserialize, Serialize};

pub mod recipe;
pub mod knowledge;
pub mod signal;
pub mod agent_state;

/// Method used for parsing recipe content from HTML.
///
/// This is defined centrally to avoid duplication between knowledge and signal models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseMethod {
    #[default]
    SchemaOrg,
    Microdata,
    Selectors,
    Heuristic,
}

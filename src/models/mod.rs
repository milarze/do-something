//! Data models for recipe scraping and self-improving agent.
//!
//! This module defines all serializable structures used for:
//! - Recipe data and storage
//! - Knowledge store (site configs, user models, patterns)
//! - Signal logging and compression
//! - Agent state persistence

pub mod recipe;
pub mod knowledge;
pub mod signal;
pub mod agent_state;

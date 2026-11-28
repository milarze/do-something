//! Framework agent layer: conversation context, prompt assembly, session
//! state, and the tool-calling runner.
//!
//! This module implements the framework half of
//! [`06-agent-runner.md`](../../../specifications/06-agent-runner.md). It is
//! generic over domains and contains no recipe symbols.

pub mod context;

pub use context::{ContextManager, HistoryEntry};

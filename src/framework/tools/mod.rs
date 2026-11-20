//! Framework tools: generic `Tool` trait and `ToolRegistry`.
//!
//! The framework defines the generic tool interface and registry. Domain
//! tools (e.g. recipe task tools) implement [`Tool`] and register in a
//! [`ToolRegistry`]. The framework never inspects a tool's internals beyond
//! the trait methods.

pub mod registry;
pub mod traits;

pub use registry::{ToolDef, ToolDefFunction, ToolRegistry};
pub use traits::{PermissionRequest, Tool, ToolResult};

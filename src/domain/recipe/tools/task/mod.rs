//! Recipe task tools.
//!
//! Concrete domain tools implementing the framework's [`Tool`] trait:
//! [`FetchHtmlTool`], [`ParseRecipeTool`], [`SaveRecipeTool`], and
//! [`SearchTool`]. These are the operations the recipe agent invokes to do
//! its work.

pub mod fetch_html;
pub mod parse_recipe;
pub mod save_recipe;
pub mod search;

pub use fetch_html::{FetchHtmlTool, RateLimiter};
pub use parse_recipe::ParseRecipeTool;
pub use save_recipe::SaveRecipeTool;
pub use search::SearchTool;

use std::sync::Arc;

use crate::domain::recipe::storage::{RecipeDb, RecipeKnowledgeStore};
use crate::framework::tools::ToolRegistry;

/// Register all recipe task tools with a registry.
///
/// Wires the domain's concrete dependencies (knowledge store, recipe db)
/// into the framework registry.
pub fn register_task_tools(
    registry: &mut ToolRegistry,
    knowledge: Arc<RecipeKnowledgeStore>,
    db: Arc<RecipeDb>,
) {
    registry.register(FetchHtmlTool::new());
    registry.register(ParseRecipeTool::new(knowledge));
    registry.register(SaveRecipeTool::new(db));
    registry.register(SearchTool::new());
}

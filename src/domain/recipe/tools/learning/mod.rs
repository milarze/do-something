//! Recipe learning tools.
//!
//! These tools let the agent modify its own knowledge base in response to
//! feedback signals: record signals, inspect knowledge, update site configs,
//! and trigger compression. Concrete domain tools implementing the
//! framework's [`Tool`] trait:
//! [`UpdateConfigTool`], [`RecordSignalTool`], [`GetKnowledgeTool`], and
//! [`CompressSignalsTool`].

pub mod compress_signals;
pub mod get_knowledge;
pub mod record_signal;
pub mod update_config;

pub use compress_signals::{CompressSignalsTool, CompressionRunnerPort};
pub use get_knowledge::GetKnowledgeTool;
pub use record_signal::RecordSignalTool;
pub use update_config::{ConfigBackupStore, FileConfigBackupStore, UpdateConfigTool};

use std::sync::Arc;

use crate::domain::recipe::learning::RecipeLearningDomain;
use crate::domain::recipe::storage::{RecipeKnowledgeStore, RecipeSignalLog};
use crate::framework::learning::runner::CompressionRunner;
use crate::framework::tools::ToolRegistry;

/// Register all recipe learning tools with a registry.
///
/// Wires the domain's concrete dependencies (knowledge store, signal log,
/// compression runner, backup store) into the framework registry.
pub fn register_learning_tools(
    registry: &mut ToolRegistry,
    knowledge: Arc<RecipeKnowledgeStore>,
    signal_log: Arc<RecipeSignalLog>,
    runner: Arc<CompressionRunner<RecipeLearningDomain>>,
    backups: Arc<dyn ConfigBackupStore>,
) {
    registry.register(UpdateConfigTool::new(
        knowledge.clone(),
        signal_log.clone(),
        backups,
    ));
    registry.register(RecordSignalTool::new(signal_log));
    registry.register(GetKnowledgeTool::new(knowledge));
    registry.register(CompressSignalsTool::new(runner));
}

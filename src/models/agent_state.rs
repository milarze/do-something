//! Agent state persistence models.
//!
//! These models handle:
//! - Session state for resumability
//! - Task progress tracking
//! - Context window management

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Unique identifier for a session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn generate() -> Self {
        Self(format!("sess_{}", uuid::Uuid::new_v4().simple()))
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Persistable session state for resumability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Session identifier.
    pub id: SessionId,

    /// Conversation history (for context window).
    #[serde(default)]
    pub history: Vec<HistoryEntry>,

    /// Current task state (if any).
    #[serde(default)]
    pub task: Option<TaskState>,

    /// Knowledge context loaded for this session.
    #[serde(default)]
    pub knowledge_context: KnowledgeContext,

    /// When this session was created.
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,

    /// When this session was last active.
    #[serde(default = "Utc::now")]
    pub last_active_at: DateTime<Utc>,

    /// Profile name in use.
    #[serde(default)]
    pub profile_name: String,

    /// Total tokens used in this session.
    #[serde(default)]
    pub total_tokens_used: u64,
}

impl SessionState {
    pub fn new(id: SessionId, profile_name: impl Into<String>) -> Self {
        Self {
            id,
            history: Vec::new(),
            task: None,
            knowledge_context: KnowledgeContext::default(),
            created_at: Utc::now(),
            last_active_at: Utc::now(),
            profile_name: profile_name.into(),
            total_tokens_used: 0,
        }
    }

    pub fn touch(&mut self) {
        self.last_active_at = Utc::now();
    }
}

/// A single entry in conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Role (system, user, assistant, tool).
    pub role: String,

    /// Text content (if applicable).
    #[serde(default)]
    pub content: Option<String>,

    /// Tool calls (if this is an assistant message with tools).
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,

    /// Tool result (if this is a tool response).
    #[serde(default)]
    pub tool_result: Option<ToolResultRecord>,

    /// Estimated token count for this entry.
    #[serde(default)]
    pub token_count: u64,

    /// When this entry was created.
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,

    /// Whether this entry has been summarized.
    #[serde(default)]
    pub is_summarized: bool,
}

/// Record of a tool call for history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Tool call ID.
    pub id: String,

    /// Tool name.
    pub name: String,

    /// Arguments as JSON string.
    pub arguments: String,
}

/// Record of a tool result for history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultRecord {
    /// Corresponding tool call ID.
    pub tool_call_id: String,

    /// Result content.
    pub content: String,

    /// Whether execution succeeded.
    pub success: bool,
}

/// Current task progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskState {
    /// What the user asked for.
    pub goal: String,

    /// URLs discovered but not yet processed.
    #[serde(default)]
    pub pending_urls: VecDeque<String>,

    /// URLs currently being processed.
    #[serde(default)]
    pub in_progress_urls: Vec<String>,

    /// URLs that have been processed.
    #[serde(default)]
    pub completed_urls: Vec<String>,

    /// URLs that failed.
    #[serde(default)]
    pub failed_urls: Vec<FailedUrl>,

    /// Recipe IDs successfully scraped.
    #[serde(default)]
    pub scraped_recipe_ids: Vec<String>,

    /// Target count (if user specified).
    #[serde(default)]
    pub target_count: Option<u32>,

    /// When this task started.
    #[serde(default = "Utc::now")]
    pub started_at: DateTime<Utc>,
}

impl TaskState {
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            pending_urls: VecDeque::new(),
            in_progress_urls: Vec::new(),
            completed_urls: Vec::new(),
            failed_urls: Vec::new(),
            scraped_recipe_ids: Vec::new(),
            target_count: None,
            started_at: Utc::now(),
        }
    }

    pub fn is_complete(&self) -> bool {
        if let Some(target) = self.target_count {
            self.scraped_recipe_ids.len() >= target as usize
        } else {
            self.pending_urls.is_empty() && self.in_progress_urls.is_empty()
        }
    }

    pub fn progress_percent(&self) -> Option<f64> {
        self.target_count.map(|target| {
            (self.scraped_recipe_ids.len() as f64 / target as f64) * 100.0
        })
    }
}

/// Record of a failed URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedUrl {
    /// The URL that failed.
    pub url: String,

    /// Error message.
    pub error: String,

    /// Number of retry attempts.
    #[serde(default)]
    pub retry_count: u32,

    /// Whether this URL should be permanently skipped.
    #[serde(default)]
    pub skip_permanently: bool,
}

/// Knowledge context loaded for a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeContext {
    /// Site configs loaded for current domains.
    #[serde(default)]
    pub site_configs: Vec<String>, // JSON strings

    /// Relevant patterns.
    #[serde(default)]
    pub patterns: Vec<String>, // JSON strings

    /// User model summary.
    #[serde(default)]
    pub user_model: Option<String>, // JSON string

    /// Token budget allocated for knowledge.
    #[serde(default)]
    pub token_budget: u64,

    /// Estimated tokens used.
    #[serde(default)]
    pub estimated_tokens: u64,
}

impl KnowledgeContext {
    pub fn is_within_budget(&self) -> bool {
        self.estimated_tokens <= self.token_budget
    }
}

/// Checkpoint for compression runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionCheckpoint {
    /// When the last compression ran.
    #[serde(default = "Utc::now")]
    pub last_run: DateTime<Utc>,

    /// Number of signals at last compression.
    #[serde(default)]
    pub signal_count: u64,

    /// Version of knowledge at last compression.
    #[serde(default)]
    pub knowledge_version: u32,
}

impl Default for CompressionCheckpoint {
    fn default() -> Self {
        Self {
            last_run: Utc::now(),
            signal_count: 0,
            knowledge_version: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_state_creation() {
        let session = SessionState::new(SessionId::generate(), "test-profile");
        assert!(!session.id.0.is_empty());
        assert!(session.history.is_empty());
        assert!(session.task.is_none());
    }

    #[test]
    fn task_state_progress() {
        let mut task = TaskState::new("Scrape 5 recipes");
        task.target_count = Some(5);
        
        assert!(!task.is_complete());
        assert_eq!(task.progress_percent(), Some(0.0));

        task.scraped_recipe_ids.push("rc_1".to_string());
        task.scraped_recipe_ids.push("rc_2".to_string());

        assert!(!task.is_complete());
        assert_eq!(task.progress_percent(), Some(40.0));
    }

    #[test]
    fn task_state_complete() {
        let mut task = TaskState::new("Scrape 3 recipes");
        task.target_count = Some(3);
        
        task.scraped_recipe_ids.push("rc_1".to_string());
        task.scraped_recipe_ids.push("rc_2".to_string());
        task.scraped_recipe_ids.push("rc_3".to_string());

        assert!(task.is_complete());
        assert_eq!(task.progress_percent(), Some(100.0));
    }

    #[test]
    fn history_entry_serialization() {
        let entry = HistoryEntry {
            role: "assistant".to_string(),
            content: Some("I'll scrape that recipe.".to_string()),
            tool_calls: vec![ToolCallRecord {
                id: "call_123".to_string(),
                name: "fetch_html".to_string(),
                arguments: r#"{"url":"https://example.com"}"#.to_string(),
            }],
            tool_result: None,
            token_count: 25,
            created_at: Utc::now(),
            is_summarized: false,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: HistoryEntry = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.role, "assistant");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "fetch_html");
    }

    #[test]
    fn failed_url_recording() {
        let failed = FailedUrl {
            url: "https://example.com/recipe/123".to_string(),
            error: "Timeout".to_string(),
            retry_count: 2,
            skip_permanently: false,
        };

        assert_eq!(failed.retry_count, 2);
        assert!(!failed.skip_permanently);
    }

    #[test]
    fn knowledge_context_budget() {
        let mut ctx = KnowledgeContext {
            token_budget: 5000,
            estimated_tokens: 4500,
            ..Default::default()
        };
        assert!(ctx.is_within_budget());

        ctx.estimated_tokens = 6000;
        assert!(!ctx.is_within_budget());
    }
}

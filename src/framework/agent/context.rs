//! Conversation context management.
//!
//! [`ContextManager`] holds the running conversation as [`HistoryEntry`]s and
//! produces a token-budgeted slice of [`ChatMessage`]s for the next LLM
//! request. Token counting is delegated to a [`TokenCounter`] (dependency
//! injection) so tests use deterministic counts and a real tokenizer can be
//! swapped in later.
//!
//! > **Spec:** [`06-agent-runner.md`](../../../specifications/06-agent-runner.md)
//! > — `framework/agent/context.rs`. `HistoryEntry` is modelled as an enum
//! > (one variant per message shape) rather than the spec's single struct with
//! > optional fields: it is type-safe (a user message cannot carry a tool
//! > result), makes [`HistoryEntry::to_chat_message`] a total function, and
//! > avoids a cluster of always-empty fields per variant.

use std::sync::Arc;

use crate::llm::{ChatMessage, ToolCall};

use super::token::{CharTokenCounter, TokenCounter};

/// A single message in the conversation history.
///
/// Each variant maps to exactly one [`ChatMessage`] via
/// [`to_chat_message`](HistoryEntry::to_chat_message). `token_count` is left at
/// zero by the constructors and recomputed by [`ContextManager::push`] using
/// the manager's [`TokenCounter`].
#[derive(Debug, Clone)]
pub enum HistoryEntry {
    /// A system instruction.
    System {
        text: String,
        token_count: u64,
    },
    /// A user message.
    User {
        text: String,
        token_count: u64,
    },
    /// An assistant turn, optionally with tool calls it requested.
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        token_count: u64,
    },
    /// A tool result returned to the model.
    ///
    /// `success` records whether the tool succeeded so
    /// [`summarize`](HistoryEntry::summarize) can emit a pass/fail placeholder
    /// without re-deriving it.
    ToolResult {
        tool_call_id: String,
        content: String,
        success: bool,
        is_summarized: bool,
        token_count: u64,
    },
}

impl HistoryEntry {
    pub fn system(text: impl Into<String>) -> Self {
        Self::System {
            text: text.into(),
            token_count: 0,
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            text: text.into(),
            token_count: 0,
        }
    }

    pub fn assistant(text: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            text,
            tool_calls,
            token_count: 0,
        }
    }

    /// A tool-result entry. `success` records whether the tool succeeded.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        success: bool,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            success,
            is_summarized: false,
            token_count: 0,
        }
    }

    /// Convert to the message shape the LLM client expects.
    pub fn to_chat_message(&self) -> ChatMessage {
        match self {
            Self::System { text, .. } => ChatMessage::system(text.clone()),
            Self::User { text, .. } => ChatMessage::user(text.clone()),
            Self::Assistant { text, tool_calls, .. } => {
                ChatMessage::assistant(text.clone(), tool_calls.clone())
            }
            Self::ToolResult {
                tool_call_id,
                content,
                ..
            } => ChatMessage::tool_result(tool_call_id.clone(), content.clone()),
        }
    }

    /// Stored token estimate (zero until the manager computes it).
    pub fn token_count(&self) -> u64 {
        match self {
            Self::System { token_count, .. }
            | Self::User { token_count, .. }
            | Self::Assistant { token_count, .. }
            | Self::ToolResult { token_count, .. } => *token_count,
        }
    }

    /// Approximate token usage of this entry under `counter`.
    fn estimate_tokens(&self, counter: &dyn TokenCounter) -> u64 {
        match self {
            Self::System { text, .. } | Self::User { text, .. } => counter.count(text),
            Self::Assistant {
                text, tool_calls, ..
            } => {
                let mut tokens = text.as_deref().map_or(0, |t| counter.count(t));
                for call in tool_calls {
                    tokens += counter.count(&call.function.name);
                    tokens += counter.count(&call.function.arguments);
                }
                tokens
            }
            Self::ToolResult {
                tool_call_id,
                content,
                ..
            } => counter.count(content) + counter.count(tool_call_id),
        }
    }

    /// Recompute and store the token estimate.
    fn set_token_count(&mut self, counter: &dyn TokenCounter) {
        let count = self.estimate_tokens(counter);
        match self {
            Self::System { token_count, .. }
            | Self::User { token_count, .. }
            | Self::Assistant { token_count, .. }
            | Self::ToolResult { token_count, .. } => *token_count = count,
        }
    }

    /// Collapse a tool result to a short pass/fail placeholder. No-op for
    /// other variants and for already-summarised tool results. Returns whether
    /// the entry changed.
    fn summarize(&mut self, counter: &dyn TokenCounter) -> bool {
        let Self::ToolResult {
            content,
            success,
            is_summarized,
            token_count,
            ..
        } = self
        else {
            return false;
        };
        if *is_summarized {
            return false;
        }
        let placeholder = format!(
            "[tool result: {}]",
            if *success { "success" } else { "failed" }
        );
        *token_count = counter.count(&placeholder);
        *content = placeholder;
        *is_summarized = true;
        true
    }
}

/// Conversation history with token budgeting.
///
/// History is held as [`HistoryEntry`]s; [`get_trimmed`](Self::get_trimmed)
/// returns the most recent entries that fit a token budget as
/// [`ChatMessage`]s. Old tool results can be collapsed to short placeholders
/// via [`summarize_old_results`](Self::summarize_old_results) to reclaim budget.
pub struct ContextManager {
    history: Vec<HistoryEntry>,
    max_tokens: u64,
    counter: Arc<dyn TokenCounter>,
}

impl ContextManager {
    /// New manager with the given hard context window and the default
    /// [`CharTokenCounter`].
    pub fn new(max_tokens: u64) -> Self {
        Self::new_with_counter(max_tokens, Arc::new(CharTokenCounter::default()))
    }

    /// New manager with an explicit token counter (for tests / real tokenizers).
    pub fn new_with_counter(max_tokens: u64, counter: Arc<dyn TokenCounter>) -> Self {
        Self {
            history: Vec::new(),
            max_tokens,
            counter,
        }
    }

    /// Append an entry, computing its token count with the manager's counter.
    pub fn push(&mut self, mut entry: HistoryEntry) {
        entry.set_token_count(&*self.counter);
        self.history.push(entry);
    }

    /// All history entries, in push order (oldest first).
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Return history as LLM messages, taking the most recent entries that fit
    /// within `budget` (clamped to the context window [`max_tokens`](Self::new)).
    ///
    /// Walks backwards from the most recent entry and stops at the first entry
    /// that would exceed the budget, so a single entry larger than the budget
    /// yields an empty result. Returned in chronological order.
    pub fn get_trimmed(&self, budget: u64) -> Vec<ChatMessage> {
        let budget = budget.min(self.max_tokens);
        let mut messages = Vec::new();
        let mut used = 0u64;
        for entry in self.history.iter().rev() {
            let tokens = entry.token_count();
            if used + tokens > budget {
                break;
            }
            used += tokens;
            messages.push(entry.to_chat_message());
        }
        messages.reverse();
        messages
    }

    /// Collapse old tool results to short placeholders, keeping the last
    /// `keep_last_n` tool results in full. No-op for non-tool entries and for
    /// already-summarised results, so repeated calls are idempotent.
    pub fn summarize_old_results(&mut self, keep_last_n: usize) {
        // Indices of tool-result entries, in chronological order.
        let tool_indices: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, HistoryEntry::ToolResult { .. }))
            .map(|(i, _)| i)
            .collect();

        // Summarise all but the last `keep_last_n`.
        let keep_from = tool_indices.len().saturating_sub(keep_last_n);
        for &i in &tool_indices[..keep_from] {
            self.history[i].summarize(&*self.counter);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, Role, ToolCall, ToolCallFunction};
    use std::sync::Arc;

    /// A manager using a 1-char-per-token counter, so token arithmetic is exact.
    fn mgr(max_tokens: u64) -> ContextManager {
        ContextManager::new_with_counter(max_tokens, Arc::new(CharTokenCounter::with_ratio(1)))
    }

    /// The text content of a message, panicking if it isn't a plain string.
    fn content_of(msg: &ChatMessage) -> String {
        match &msg.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            other => panic!("expected string content, got {other:?}"),
        }
    }

    fn assert_user(msg: &ChatMessage, expected: &str) {
        assert!(matches!(msg.role, Role::User), "expected User role");
        assert_eq!(content_of(msg), expected);
    }

    fn assert_tool_summarized(entry: &HistoryEntry, success: bool) {
        match entry {
            HistoryEntry::ToolResult {
                content,
                success: s,
                is_summarized,
                ..
            } => {
                assert!(*is_summarized, "expected summarized");
                assert_eq!(*s, success);
                let expected = if success {
                    "[tool result: success]"
                } else {
                    "[tool result: failed]"
                };
                assert_eq!(content, expected);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    fn assert_tool_full(entry: &HistoryEntry, content: &str) {
        match entry {
            HistoryEntry::ToolResult {
                content: c,
                is_summarized,
                ..
            } => {
                assert!(!*is_summarized, "expected not summarized");
                assert_eq!(c, content);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    // --- HistoryEntry::to_chat_message -------------------------------------

    #[test]
    fn to_chat_message_system() {
        let m = HistoryEntry::system("you are helpful").to_chat_message();
        assert!(matches!(m.role, Role::System));
        assert_eq!(content_of(&m), "you are helpful");
    }

    #[test]
    fn to_chat_message_user() {
        let m = HistoryEntry::user("hi").to_chat_message();
        assert!(matches!(m.role, Role::User));
        assert_eq!(content_of(&m), "hi");
    }

    #[test]
    fn to_chat_message_assistant_with_text_and_calls() {
        let call = ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "f".into(),
                arguments: "{}".into(),
            },
        };
        let m = HistoryEntry::assistant(Some("running".into()), vec![call]).to_chat_message();
        assert!(matches!(m.role, Role::Assistant));
        assert_eq!(content_of(&m), "running");
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].id, "1");
    }

    #[test]
    fn to_chat_message_assistant_without_text() {
        let m = HistoryEntry::assistant(None, vec![]).to_chat_message();
        assert!(matches!(m.role, Role::Assistant));
        assert!(m.content.is_none());
        assert!(m.tool_calls.is_empty());
    }

    #[test]
    fn to_chat_message_tool_result() {
        let m = HistoryEntry::tool_result("call_9", "done", true).to_chat_message();
        assert!(matches!(m.role, Role::Tool));
        assert_eq!(m.tool_call_id.as_deref(), Some("call_9"));
        assert_eq!(content_of(&m), "done");
    }

    // --- token counting -----------------------------------------------------

    #[test]
    fn constructors_leave_token_count_zero() {
        assert_eq!(HistoryEntry::user("hello").token_count(), 0);
    }

    #[test]
    fn push_computes_token_count() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("hello")); // 5 chars → 5 tokens
        assert_eq!(cm.history()[0].token_count(), 5);
    }

    #[test]
    fn assistant_token_count_includes_tool_calls() {
        let mut cm = mgr(1000);
        let args = "{\"url\":\"http://x\"}";
        let call = ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "fetch".into(),
                arguments: args.into(),
            },
        };
        cm.push(HistoryEntry::assistant(Some("hi".into()), vec![call]));
        // text "hi"(2) + name "fetch"(5) + args(len)
        let expected = 2 + 5 + args.chars().count() as u64;
        assert_eq!(cm.history()[0].token_count(), expected);
    }

    #[test]
    fn tool_result_token_count_includes_id() {
        let mut cm = mgr(1000);
        cm.push(HistoryEntry::tool_result("c1", "hello", true));
        // content "hello"(5) + id "c1"(2) = 7
        assert_eq!(cm.history()[0].token_count(), 7);
    }

    // --- get_trimmed --------------------------------------------------------

    #[test]
    fn get_trimmed_empty_history() {
        assert!(mgr(100).get_trimmed(50).is_empty());
    }

    #[test]
    fn get_trimmed_single_entry_within_budget() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("hello")); // 5 tokens
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 1);
        assert_user(&msgs[0], "hello");
    }

    #[test]
    fn get_trimmed_single_entry_over_budget_returns_empty() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("hello")); // 5 tokens
        assert!(cm.get_trimmed(4).is_empty());
    }

    #[test]
    fn get_trimmed_takes_most_recent_within_budget_in_order() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("aaaa")); // 4
        cm.push(HistoryEntry::user("bb")); // 2
        cm.push(HistoryEntry::user("ccc")); // 3
        // budget 5: ccc(3) + bb(2) = 5, stop before aaaa(4)
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 2);
        assert_user(&msgs[0], "bb");
        assert_user(&msgs[1], "ccc");
    }

    #[test]
    fn get_trimmed_exact_budget_boundary() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("aaa")); // 3
        cm.push(HistoryEntry::user("bb")); // 2
        // budget 5: 3 + 2 = 5, both included
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn get_trimmed_clamped_to_max_tokens() {
        let mut cm = mgr(3); // window of 3
        cm.push(HistoryEntry::user("aaaa")); // 4 > window
        cm.push(HistoryEntry::user("bb")); // 2 <= window
        // budget 100 clamped to 3: take bb(2), stop before aaaa(4)
        let msgs = cm.get_trimmed(100);
        assert_eq!(msgs.len(), 1);
        assert_user(&msgs[0], "bb");
    }

    // --- summarize_old_results ---------------------------------------------

    #[test]
    fn summarize_old_results_noop_without_tool_results() {
        let mut cm = mgr(1000);
        cm.push(HistoryEntry::user("hi"));
        cm.push(HistoryEntry::assistant(Some("yo".into()), vec![]));
        cm.summarize_old_results(1);
        assert_eq!(cm.history().len(), 2);
        assert_user(&cm.history()[0].to_chat_message(), "hi");
    }

    #[test]
    fn summarize_old_results_keeps_last_n() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true)); // summarize
        cm.push(HistoryEntry::tool_result("c2", &big, false)); // summarize
        cm.push(HistoryEntry::tool_result("c3", &big, true)); // keep
        cm.summarize_old_results(1);
        let h = cm.history();
        assert_tool_summarized(&h[0], true);
        assert_tool_summarized(&h[1], false);
        assert_tool_full(&h[2], &big);
    }

    #[test]
    fn summarize_old_results_keep_zero_summarizes_all() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.push(HistoryEntry::tool_result("c2", &big, false));
        cm.summarize_old_results(0);
        assert_tool_summarized(&cm.history()[0], true);
        assert_tool_summarized(&cm.history()[1], false);
    }

    #[test]
    fn summarize_old_results_keep_more_than_count_summarizes_none() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.summarize_old_results(5);
        assert_tool_full(&cm.history()[0], &big);
    }

    #[test]
    fn summarize_old_results_idempotent() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.summarize_old_results(0);
        let after_first = content_of(&cm.history()[0].to_chat_message());
        cm.summarize_old_results(0); // again — should be a no-op
        let after_second = content_of(&cm.history()[0].to_chat_message());
        assert_eq!(after_first, after_second);
    }

    #[test]
    fn summarize_old_results_reduces_token_count() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100); // 100 + id "c1"(2) = 102
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        let before = cm.history()[0].token_count();
        cm.summarize_old_results(0);
        let after = cm.history()[0].token_count();
        assert!(after < before, "after {after} should be < before {before}");
    }

    #[test]
    fn summarize_old_results_ignores_non_tool_entries() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::user("hello"));
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.push(HistoryEntry::assistant(Some("ok".into()), vec![]));
        cm.push(HistoryEntry::tool_result("c2", &big, false));
        cm.summarize_old_results(1); // keep only c2
        let h = cm.history();
        assert_user(&h[0].to_chat_message(), "hello");
        assert_tool_summarized(&h[1], true);
        assert!(matches!(h[2], HistoryEntry::Assistant { .. }));
        assert_tool_full(&h[3], &big);
    }
}

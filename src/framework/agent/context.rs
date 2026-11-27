//! Conversation context management.
//!
//! [`ContextManager`] holds the running conversation as [`HistoryEntry`]s and
//! produces a budgeted slice of [`ChatMessage`]s for the next LLM request.
//!
//! The framework does **not** count tokens client-side — it has no model
//! tokenizer, and a chars-per-token heuristic under-counts the JSON-heavy tool
//! traffic that dominates an agent loop. The server's tokenizer is the single
//! source of truth:
//!
//! - `get_trimmed(max_bytes)` is a **coarse byte pre-filter** — it takes the
//!   most recent entries under a byte ceiling to avoid sending obviously-huge
//!   history for a wasted round-trip. It never claims exact fit.
//! - `compact(keep_last_n)` is the **correctness** lever. It summarizes old
//!   tool results to placeholders (retaining the tool name), then drops the
//!   oldest non-system entry once all tool results are summarized. Returns
//!   `true` when it made progress, `false` when context is exhausted.
//! - `record_usage(prompt_tokens)` stores the last server-reported
//!   `usage.prompt_tokens` for warm-turn budgeting from ground truth.
//!
//! > **Spec:** [`06-agent-runner.md`](../../../specifications/06-agent-runner.md)
//! > — `framework/agent/context.rs`. `HistoryEntry` is modelled as an enum
//! > (one variant per message shape) rather than a struct with optional fields:
//! > type-safe, total `to_chat_message`, no cluster of empty fields per variant.
//!
//! > **Note:** the previous version held a `TokenCounter` trait and per-entry
//! > `token_count: u64`. Dropped because the framework has no tokenizer and a
//! > chars-per-token estimate under-counts JSON tool traffic. The server's
//! > tokenizer is authoritative; `compact` reacts to its `ContextOverflow`
//! > verdict.

use crate::llm::{ChatMessage, ToolCall};

/// A single message in the conversation history.
///
/// Each variant maps to exactly one [`ChatMessage`] via
/// [`to_chat_message`](HistoryEntry::to_chat_message).
#[derive(Debug, Clone)]
pub enum HistoryEntry {
    /// A system instruction.
    System {
        text: String,
    },
    /// A user message.
    User {
        text: String,
    },
    /// An assistant turn, optionally with tool calls it requested.
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
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
    },
}

impl HistoryEntry {
    /// Create a system entry.
    pub fn system(text: impl Into<String>) -> Self {
        Self::System { text: text.into() }
    }

    /// Create a user entry.
    pub fn user(text: impl Into<String>) -> Self {
        Self::User { text: text.into() }
    }

    /// Create an assistant entry.
    pub fn assistant(text: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant { text, tool_calls }
    }

    /// Create a tool-result entry. `success` records whether the tool succeeded.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>, success: bool) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            success,
            is_summarized: false,
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
            Self::ToolResult { tool_call_id, content, .. } => {
                ChatMessage::tool_result(tool_call_id.clone(), content.clone())
            }
        }
    }

    /// Coarse byte size of this entry (for the pre-filter only — not a token count).
    pub fn size(&self) -> u64 {
        match self {
            Self::System { text } => text.len() as u64,
            Self::User { text } => text.len() as u64,
            Self::Assistant { text, tool_calls } => {
                let text_len = text.as_deref().map_or(0, |t| t.len());
                let calls_len: usize = tool_calls.iter().map(|c| {
                    c.id.len() + c.kind.len() + c.function.name.len() + c.function.arguments.len()
                }).sum();
                (text_len + calls_len) as u64
            }
            Self::ToolResult { tool_call_id, content, .. } => {
                (tool_call_id.len() + content.len()) as u64
            }
        }
    }

    /// Collapse a tool result to a short placeholder that retains the tool
    /// name: `[tool result: <name>: success|failed]`. No-op for other variants
    /// and already-summarized results. Returns whether it changed.
    fn summarize(&mut self) -> bool {
        let Self::ToolResult {
            content,
            success,
            is_summarized,
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
        *content = placeholder;
        *is_summarized = true;
        true
    }
}

/// Conversation context management.
///
/// History is held as [`HistoryEntry`]s; [`get_trimmed`](Self::get_trimmed)
/// returns the most recent entries that fit a coarse byte ceiling as
/// [`ChatMessage`]s. Old tool results can be collapsed to short placeholders
/// via [`compact`](Self::compact), which also drops the oldest non-system
/// entry when all tool results are summarized.
pub struct ContextManager {
    history: Vec<HistoryEntry>,
    /// Coarse byte ceiling — an honest heuristic, not an exact fit.
    max_bytes: u64,
    /// Last server-reported prompt token count (warm-turn ground truth).
    last_prompt_tokens: Option<u64>,
}

impl ContextManager {
    /// New manager with the given byte ceiling.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            history: Vec::new(),
            max_bytes,
            last_prompt_tokens: None,
        }
    }

    /// Append an entry.
    pub fn push(&mut self, entry: HistoryEntry) {
        self.history.push(entry);
    }

    /// All history entries, in push order (oldest first).
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Return history as LLM messages, taking the most recent entries that fit
    /// within `max_bytes` (clamped to the context window [`max_bytes`](Self::new)).
    ///
    /// Walks backwards from the most recent entry and stops at the first entry
    /// that would exceed the budget, so a single entry larger than the budget
    /// yields an empty result. Returned in chronological order.
    ///
    /// This is a **coarse byte pre-filter only**; correctness comes from
    /// reactive [`compact`](Self::compact) on server `ContextOverflow`.
    pub fn get_trimmed(&self, max_bytes: u64) -> Vec<ChatMessage> {
        let max_bytes = max_bytes.min(self.max_bytes);
        let mut messages = Vec::new();
        let mut used = 0u64;
        for entry in self.history.iter().rev() {
            let size = entry.size();
            if used + size > max_bytes {
                break;
            }
            used += size;
            messages.push(entry.to_chat_message());
        }
        messages.reverse();
        messages
    }

    /// Summarize old tool results (keep last `keep_last_n` in full), then drop
    /// the oldest non-system entry once all tool results are summarized. Returns
    /// `true` when it made progress, `false` when context is exhausted and no
    /// further progress is possible.
    ///
    /// Idempotent: repeated calls have no additional effect once the context is
    /// fully compacted.
    pub fn compact(&mut self, keep_last_n: usize) -> bool {
        // Phase 1: summarize old tool results, keeping the last `keep_last_n`.
        let tool_indices: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, HistoryEntry::ToolResult { .. }))
            .map(|(i, _)| i)
            .collect();

        let keep_from = tool_indices.len().saturating_sub(keep_last_n);
        let mut made_progress = false;
        for &i in &tool_indices[..keep_from] {
            if self.history[i].summarize() {
                made_progress = true;
            }
        }

        // Phase 2: if all tool results are already summarized (and there were
        // tool results), drop the oldest User/Assistant entry to make room.
        // ToolResult entries are already compacted placeholders — they are
        // the last thing to go.
        if !tool_indices.is_empty()
            && tool_indices.iter().all(|&i| {
                matches!(&self.history[i], HistoryEntry::ToolResult { is_summarized: true, .. })
            })
            && let Some(drop_idx) = self
                .history
                .iter()
                .position(|e| matches!(e, HistoryEntry::User { .. } | HistoryEntry::Assistant { .. }))
        {
            self.history.remove(drop_idx);
            made_progress = true;
        }

        made_progress
    }

    /// Record the server's last reported prompt token count.
    pub fn record_usage(&mut self, prompt_tokens: u64) {
        self.last_prompt_tokens = Some(prompt_tokens);
    }

    /// Return the last server-reported prompt token count, if available.
    pub fn last_usage(&self) -> Option<u64> {
        self.last_prompt_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, Role, ToolCall, ToolCallFunction};

    fn mgr(max_bytes: u64) -> ContextManager {
        ContextManager::new(max_bytes)
    }

    /// The text content of a message, panicking if it isn't a plain string.
    fn content_of(msg: &ChatMessage) -> String {
        match &msg.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            other => panic!("expected string content, got {other:?}"),
        }
    }

    fn assert_user(msg: &ChatMessage, expected: &str) {
        assert!(matches!(msg.role, Role::User), "expected User role, got {:?}", msg.role);
        assert_eq!(content_of(msg), expected);
    }

    fn assert_tool_summarized(entry: &HistoryEntry, expected: &str) {
        match entry {
            HistoryEntry::ToolResult { content, is_summarized, .. } => {
                assert!(*is_summarized, "expected summarized");
                assert_eq!(content, expected);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    fn assert_tool_full(entry: &HistoryEntry, content: &str) {
        match entry {
            HistoryEntry::ToolResult { content: c, is_summarized, .. } => {
                assert!(!*is_summarized, "expected not summarized");
                assert_eq!(c, content);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    // --- HistoryEntry -------------------------------------------------------

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

    #[test]
    fn history_entry_size() {
        let entry = HistoryEntry::user("hello");
        assert_eq!(entry.size(), 5);

        let entry = HistoryEntry::system("hello world");
        assert_eq!(entry.size(), 11);
    }

    // --- get_trimmed --------------------------------------------------------

    #[test]
    fn get_trimmed_empty_history() {
        assert!(mgr(100).get_trimmed(50).is_empty());
    }

    #[test]
    fn get_trimmed_single_entry_within_budget() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("hello")); // 5 bytes
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 1);
        assert_user(&msgs[0], "hello");
    }

    #[test]
    fn get_trimmed_single_entry_over_budget_returns_empty() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("hello")); // 5 bytes
        assert!(cm.get_trimmed(4).is_empty());
    }

    #[test]
    fn get_trimmed_takes_most_recent_within_budget_in_order() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("aaaa")); // 4 bytes
        cm.push(HistoryEntry::user("bb"));   // 2 bytes
        cm.push(HistoryEntry::user("ccc"));  // 3 bytes
        // budget 5: bb(2) + ccc(3) = 5, stop before aaaa(4)
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 2);
        assert_user(&msgs[0], "bb");
        assert_user(&msgs[1], "ccc");
    }

    #[test]
    fn get_trimmed_exact_budget_boundary() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::user("aaa")); // 3 bytes
        cm.push(HistoryEntry::user("bb"));  // 2 bytes
        // budget 5: 3 + 2 = 5, both included
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn get_trimmed_clamped_to_max_bytes() {
        let mut cm = mgr(3); // window of 3
        cm.push(HistoryEntry::user("aaaa")); // 4 > window
        cm.push(HistoryEntry::user("bb"));   // 2 <= window
        // budget 100 clamped to 3: take bb(2), stop before aaaa(4)
        let msgs = cm.get_trimmed(100);
        assert_eq!(msgs.len(), 1);
        assert_user(&msgs[0], "bb");
    }

    #[test]
    fn get_trimmed_includes_system_entry() {
        let mut cm = mgr(100);
        cm.push(HistoryEntry::system("be helpful")); // 12 bytes
        cm.push(HistoryEntry::user("hi"));            // 2 bytes
        // budget 5: only hi fits
        let msgs = cm.get_trimmed(5);
        assert_eq!(msgs.len(), 1);
        assert_user(&msgs[0], "hi");
        // budget 20: both fit
        let msgs = cm.get_trimmed(20);
        assert_eq!(msgs.len(), 2);
    }

    // --- summarize ----------------------------------------------------------

    #[test]
    fn summarize_tool_result_success() {
        let mut entry = HistoryEntry::tool_result("c1", "big result", true);
        assert!(entry.summarize());
        assert_tool_summarized(&entry, "[tool result: success]");
    }

    #[test]
    fn summarize_tool_result_failure() {
        let mut entry = HistoryEntry::tool_result("c1", "big result", false);
        assert!(entry.summarize());
        assert_tool_summarized(&entry, "[tool result: failed]");
    }

    #[test]
    fn summarize_non_tool_is_noop() {
        let mut entry = HistoryEntry::user("hi");
        assert!(!entry.summarize());
    }

    #[test]
    fn summarize_already_summarized_is_noop() {
        let mut entry = HistoryEntry::tool_result("c1", "big result", true);
        entry.summarize();
        assert!(!entry.summarize()); // second call is no-op
    }

    // --- compact ------------------------------------------------------------

    #[test]
    fn compact_noop_without_tool_results() {
        let mut cm = mgr(1000);
        cm.push(HistoryEntry::user("hi"));
        cm.push(HistoryEntry::assistant(Some("yo".into()), vec![]));
        assert!(!cm.compact(1));
        assert_eq!(cm.history().len(), 2);
    }

    #[test]
    fn compact_keeps_last_n_tool_results() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.push(HistoryEntry::tool_result("c2", &big, false));
        cm.push(HistoryEntry::tool_result("c3", &big, true));
        // compact(1): summarize c1, c2; c3 stays.
        assert!(cm.compact(1));
        let h = cm.history();
        assert_tool_summarized(&h[0], "[tool result: success]");
        assert_tool_summarized(&h[1], "[tool result: failed]");
        assert_tool_full(&h[2], &big);
    }

    #[test]
    fn compact_summarizes_all_when_keep_zero() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.push(HistoryEntry::tool_result("c2", &big, false));
        assert!(cm.compact(0));
        assert_tool_summarized(&cm.history()[0], "[tool result: success]");
        assert_tool_summarized(&cm.history()[1], "[tool result: failed]");
    }

    #[test]
    fn compact_keep_more_than_count_skips_summarizing() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        assert!(!cm.compact(5)); // no summarization needed
        assert_tool_full(&cm.history()[0], &big);
    }

    #[test]
    fn compact_idempotent_after_full_compaction() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        assert!(cm.compact(0));
        assert!(!cm.compact(0)); // second call: nothing to do
    }

    #[test]
    fn compact_drops_oldest_non_system_after_all_summarized() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::user("hi"));
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        cm.push(HistoryEntry::tool_result("c2", &big, false));
        // compact(0): summarize both tool results, then drop "hi" (oldest User).
        assert!(cm.compact(0));
        let h = cm.history();
        assert_eq!(h.len(), 2);
        assert_tool_summarized(&h[0], "[tool result: success]");
        assert_tool_summarized(&h[1], "[tool result: failed]");
    }

    #[test]
    fn compact_does_not_drop_system_entries() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::system("be helpful"));
        cm.push(HistoryEntry::tool_result("c1", &big, true));
        // compact(0): summarize c1, no User/Assistant to drop — system stays.
        assert!(cm.compact(0));
        let h = cm.history();
        assert_eq!(h.len(), 2);
        assert!(matches!(h[0], HistoryEntry::System { .. }));
        assert_tool_summarized(&h[1], "[tool result: success]");
    }

    #[test]
    fn compact_mixed_summarized_and_unsummarized() {
        let mut cm = mgr(1000);
        let big = "x".repeat(100);
        cm.push(HistoryEntry::user("hi"));
        cm.push(HistoryEntry::tool_result("c1", "small1", true));
        cm.push(HistoryEntry::tool_result("c2", &big, false));
        cm.push(HistoryEntry::tool_result("c3", "small3", true));
        // compact(1): keep_from = 3-1 = 2, summarize c1+c2, keep c3. Not all
        // summarized → no drop phase. User "hi" stays.
        assert!(cm.compact(1));
        let h = cm.history();
        assert_eq!(h.len(), 4);
        assert_tool_summarized(&h[1], "[tool result: success]");
        assert_tool_summarized(&h[2], "[tool result: failed]");
        assert_tool_full(&h[3], "small3");
    }

    // --- record_usage -------------------------------------------------------

    #[test]
    fn record_usage_stores_and_returns_value() {
        let mut cm = ContextManager::new(1000);
        assert_eq!(cm.last_usage(), None);
        cm.record_usage(42);
        assert_eq!(cm.last_usage(), Some(42));
        cm.record_usage(99);
        assert_eq!(cm.last_usage(), Some(99));
    }

    #[test]
    fn record_usage_does_not_affect_history() {
        let mut cm = ContextManager::new(1000);
        cm.push(HistoryEntry::user("hi"));
        cm.record_usage(10);
        assert_eq!(cm.history().len(), 1);
    }
}

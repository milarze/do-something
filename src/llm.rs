//! OpenAI-compatible chat completions client with SSE streaming and tool calls.

use crate::config::Profile;
use anyhow::{Context, Result, anyhow};
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments (string).
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// Either a plain string (most messages) or an array of OpenAI content
    /// parts (multimodal user messages). Stored as raw JSON so we don't
    /// commit to a fixed shape — different OpenAI-compatible servers tolerate
    /// slightly different part schemas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(serde_json::Value::String(text.into())),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(serde_json::Value::String(text.into())),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }
    /// Multimodal user message. `parts` is the OpenAI `content` array, e.g.
    /// `[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]`.
    pub fn user_multipart(parts: Vec<serde_json::Value>) -> Self {
        Self {
            role: Role::User,
            content: Some(serde_json::Value::Array(parts)),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }
    pub fn assistant(text: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: text.map(serde_json::Value::String),
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(serde_json::Value::String(content.into())),
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }
}

/// Tool definition for the OpenAI `tools` array.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: &'static str, // "function"
    pub function: ToolDefFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChunkChoiceDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChunkChoice {
    delta: ChunkChoiceDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

/// Optional error envelope that can appear either in a non-2xx HTTP body or
/// inline as a streaming chunk. Both llama.cpp and OpenAI use this shape.
#[derive(Debug, Clone, Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Debug, Clone, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    message: Option<String>,
    #[allow(dead_code)]
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    code: Option<serde_json::Value>,
}

#[derive(Debug, Default, Clone)]
struct ToolCallAccum {
    id: Option<String>,
    name: Option<String>,
    args: String,
    announced: bool,
}

pub struct LlmClient {
    http: reqwest::Client,
}

#[derive(Debug)]
pub enum StreamEvent {
    /// A streaming text token from the assistant.
    TextDelta(String),
    /// First time we see a given tool call, with the announced id+name.
    /// Args may still be incomplete; full args arrive in `ToolCallDone`.
    ToolCallStart {
        id: String,
        name: String,
    },
    /// Final assembled tool calls when the stream finishes.
    Done {
        finish_reason: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    /// The server rejected the request because the prompt + reservation
    /// exceeded the available context window. The agent should treat this as
    /// `StopReason::MaxTokens` and let the client compact history.
    ContextOverflow(String),
    Error(String),
    Cancelled,
}

impl LlmClient {
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("building reqwest client")?;
        Ok(Self { http })
    }

    /// Best-effort preflight against an OpenAI-compatible (and ideally
    /// llama.cpp) server. Every probe is tolerant: we log and continue on
    /// failure. Never returns Err — callers just inspect the report.
    pub async fn preflight(&self, profile: &Profile) -> PreflightReport {
        let timeout = std::time::Duration::from_millis(1500);
        let v1 = profile.base_url.trim_end_matches('/').to_string();
        let root = strip_v1_suffix(&v1);

        let mut report = PreflightReport::default();

        // /health — llama.cpp specific (also exposed at /v1/health on llama).
        let health_url = format!("{root}/health");
        match self.http.get(&health_url).timeout(timeout).send().await {
            Ok(resp) => {
                report.health = Some(HealthState {
                    status: resp.status().as_u16(),
                });
            }
            Err(e) => {
                tracing::debug!("preflight: GET {health_url} failed: {e}");
            }
        }

        // /props — llama.cpp specific.
        let props_url = format!("{root}/props");
        match self.http.get(&props_url).timeout(timeout).send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<PropsResponse>().await {
                Ok(p) => report.props = Some(p),
                Err(e) => tracing::debug!("preflight: parsing {props_url}: {e}"),
            },
            Ok(resp) => {
                tracing::debug!("preflight: GET {props_url} -> HTTP {}", resp.status());
            }
            Err(e) => tracing::debug!("preflight: GET {props_url} failed: {e}"),
        }

        // /v1/models — OpenAI standard.
        let models_url = format!("{v1}/models");
        match self.http.get(&models_url).timeout(timeout).send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<ModelsResponse>().await {
                Ok(m) => {
                    report.models = m.data.into_iter().map(|d| d.id).collect();
                }
                Err(e) => tracing::debug!("preflight: parsing {models_url}: {e}"),
            },
            Ok(resp) => {
                tracing::debug!("preflight: GET {models_url} -> HTTP {}", resp.status());
            }
            Err(e) => tracing::debug!("preflight: GET {models_url} failed: {e}"),
        }

        report
    }

    pub async fn chat_stream(
        &self,
        profile: &Profile,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        cancel: CancellationToken,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>> {
        let url = format!("{}/chat/completions", profile.base_url.trim_end_matches('/'));

        let mut body = serde_json::json!({
            "model": profile.model,
            "messages": messages,
            "stream": true,
        });
        let obj = body.as_object_mut().expect("object");
        if !tools.is_empty() {
            obj.insert("tools".to_string(), serde_json::to_value(tools)?);
        }
        if let Some(t) = profile.temperature {
            obj.insert("temperature".to_string(), serde_json::json!(t));
        }
        if let Some(m) = profile.max_tokens {
            obj.insert("max_tokens".to_string(), serde_json::json!(m));
        }
        for (k, v) in profile.extra_body.iter() {
            obj.insert(k.clone(), toml_to_json(v.clone()));
        }

        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = profile.api_key() {
            req = req.bearer_auth(key);
        }

        let resp = tokio::select! {
            r = req.send() => r,
            _ = cancel.cancelled() => {
                return Ok(Box::pin(futures::stream::once(async { StreamEvent::Cancelled })));
            }
        };
        let resp = resp.context("sending chat/completions request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let snippet = text.chars().take(500).collect::<String>();
            if is_context_overflow(status.as_u16(), &text) {
                tracing::warn!(
                    "chat/completions HTTP {status}: context overflow detected -> StopReason::MaxTokens"
                );
                let msg = snippet.clone();
                return Ok(Box::pin(futures::stream::once(async move {
                    StreamEvent::ContextOverflow(msg)
                })));
            }
            return Err(anyhow!("chat/completions HTTP {status}: {snippet}"));
        }

        let byte_stream = resp.bytes_stream();
        let sse = byte_stream.eventsource();

        let stream = async_stream::stream! {
            futures::pin_mut!(sse);
            let mut tool_buf: BTreeMap<u32, ToolCallAccum> = BTreeMap::new();
            let mut last_finish: Option<String> = None;

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        yield StreamEvent::Cancelled;
                        return;
                    }
                    next = sse.next() => {
                        let Some(item) = next else { break; };
                        match item {
                            Err(e) => {
                                yield StreamEvent::Error(format!("sse error: {e}"));
                                return;
                            }
                            Ok(ev) => {
                                if ev.data.is_empty() { continue; }
                                if ev.data == "[DONE]" { break; }

                                // Check for an error envelope first — llama.cpp
                                // and OpenAI-compatible servers can send
                                // `data: {"error": {...}}` mid-stream.
                                if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(&ev.data) {
                                    if let Some(err) = env.error {
                                        let msg = err.message.unwrap_or_else(|| ev.data.clone());
                                        if is_context_overflow_message(&msg) {
                                            tracing::warn!(
                                                "stream error indicates context overflow: {msg}"
                                            );
                                            yield StreamEvent::ContextOverflow(msg);
                                        } else {
                                            yield StreamEvent::Error(msg);
                                        }
                                        return;
                                    }
                                }

                                let chunk = match serde_json::from_str::<ChatChunk>(&ev.data) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        tracing::warn!("malformed chunk: {e}; data={}", ev.data);
                                        continue;
                                    }
                                };
                                for c in chunk.choices {
                                    if let Some(t) = c.delta.content {
                                        if !t.is_empty() {
                                            yield StreamEvent::TextDelta(t);
                                        }
                                    }
                                    if let Some(tcs) = c.delta.tool_calls {
                                        for d in tcs {
                                            let entry = tool_buf.entry(d.index).or_default();
                                            if let Some(id) = d.id { entry.id = Some(id); }
                                            if let Some(f) = d.function {
                                                if let Some(n) = f.name {
                                                    match entry.name.as_mut() {
                                                        Some(existing) => existing.push_str(&n),
                                                        None => entry.name = Some(n),
                                                    }
                                                }
                                                if let Some(a) = f.arguments {
                                                    entry.args.push_str(&a);
                                                }
                                            }
                                            if !entry.announced {
                                                if let (Some(id), Some(name)) = (entry.id.clone(), entry.name.clone()) {
                                                    entry.announced = true;
                                                    yield StreamEvent::ToolCallStart { id, name };
                                                }
                                            }
                                        }
                                    }
                                    if let Some(reason) = c.finish_reason {
                                        last_finish = Some(reason);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Assemble final tool calls.
            let mut tool_calls = Vec::new();
            for (_idx, accum) in tool_buf {
                if let (Some(id), Some(name)) = (accum.id, accum.name) {
                    tool_calls.push(ToolCall {
                        id,
                        kind: "function".into(),
                        function: ToolCallFunction {
                            name,
                            arguments: if accum.args.is_empty() { "{}".into() } else { accum.args },
                        },
                    });
                }
            }
            yield StreamEvent::Done { finish_reason: last_finish, tool_calls };
        };

        Ok(Box::pin(stream))
    }
}

fn toml_to_json(v: toml::Value) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        toml::Value::String(s) => J::String(s),
        toml::Value::Integer(i) => J::Number(i.into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(f).map(J::Number).unwrap_or(J::Null),
        toml::Value::Boolean(b) => J::Bool(b),
        toml::Value::Datetime(d) => J::String(d.to_string()),
        toml::Value::Array(a) => J::Array(a.into_iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => J::Object(
            t.into_iter()
                .map(|(k, v)| (k, toml_to_json(v)))
                .collect::<serde_json::Map<_, _>>(),
        ),
    }
}

// ===== Preflight types =====

#[derive(Debug, Default, Clone)]
pub struct PreflightReport {
    pub health: Option<HealthState>,
    pub props: Option<PropsResponse>,
    pub models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HealthState {
    pub status: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PropsResponse {
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub build_info: Option<String>,
    #[serde(default)]
    pub total_slots: Option<u32>,
    /// Present when the server was started with `--jinja`. We don't try to
    /// interpret the contents — its mere presence (and non-empty caps) is a
    /// reliable signal that native tool-calls work.
    #[serde(default)]
    pub chat_template_caps: Option<serde_json::Value>,
    /// Reserved for future vision/audio support; currently unused.
    #[allow(dead_code)]
    #[serde(default)]
    pub modalities: Option<serde_json::Value>,
}

#[allow(dead_code)]
impl PropsResponse {
    /// Heuristic: llama.cpp populates `chat_template_caps` only when `--jinja`
    /// is enabled. An empty object/array means "no caps detected" — treat as
    /// false to be safe.
    pub fn jinja_enabled(&self) -> bool {
        match &self.chat_template_caps {
            Some(serde_json::Value::Object(m)) => !m.is_empty(),
            Some(serde_json::Value::Array(a)) => !a.is_empty(),
            Some(_) => true,
            None => false,
        }
    }

    pub fn supports_tools(&self) -> bool {
        match &self.chat_template_caps {
            Some(serde_json::Value::Object(m)) => m
                .get("supports_tools")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelEntry {
    id: String,
}

/// Strip a trailing `/v1` (with or without trailing slash) from a base URL to
/// get the server root, where llama.cpp's `/health` and `/props` live.
fn strip_v1_suffix(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Heuristic detection of context-overflow errors from a non-2xx HTTP body.
/// Tries to parse an `{"error": {...}}` envelope first; falls back to a raw
/// substring scan if the body is plain text.
fn is_context_overflow(status: u16, body: &str) -> bool {
    // llama.cpp returns 500 for context overflow; some proxies use 400/413.
    if !matches!(status, 400 | 413 | 500 | 503) {
        return false;
    }
    if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(body) {
        if let Some(msg) = env.error.and_then(|e| e.message) {
            return is_context_overflow_message(&msg);
        }
    }
    is_context_overflow_message(body)
}

/// Substring match against the various phrasings llama.cpp / OpenAI use for
/// "your prompt is too long".
fn is_context_overflow_message(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    // llama.cpp: "the request exceeds the available context size" /
    //            "input is too large to process. increase the physical batch size"
    // openai:    "this model's maximum context length is N tokens" /
    //            "context_length_exceeded"
    m.contains("context size")
        || m.contains("context length")
        || m.contains("context_length_exceeded")
        || m.contains("exceeds the available context")
        || m.contains("input is too large")
        || m.contains("maximum context length")
        || m.contains("too many tokens")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_llama_cpp_overflow() {
        let body = r#"{"error":{"code":500,"message":"the request exceeds the available context size. try increasing the context size or enable context shift","type":"server_error"}}"#;
        assert!(is_context_overflow(500, body));
    }

    #[test]
    fn detects_openai_overflow() {
        let body = r#"{"error":{"message":"This model's maximum context length is 8192 tokens. However, you requested 9000 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}"#;
        assert!(is_context_overflow(400, body));
    }

    #[test]
    fn ignores_unrelated_500() {
        let body = r#"{"error":{"message":"internal server error","type":"server_error"}}"#;
        assert!(!is_context_overflow(500, body));
    }

    #[test]
    fn ignores_2xx() {
        // Defensive: status check should keep success codes from being flagged.
        let body = "context length exceeded";
        assert!(!is_context_overflow(200, body));
    }

    #[test]
    fn strips_v1_suffix_correctly() {
        assert_eq!(strip_v1_suffix("http://x:8080/v1"), "http://x:8080");
        assert_eq!(strip_v1_suffix("http://x:8080/v1/"), "http://x:8080");
        assert_eq!(strip_v1_suffix("http://x:8080"), "http://x:8080");
        assert_eq!(strip_v1_suffix("http://x/api/v1"), "http://x/api");
    }
}

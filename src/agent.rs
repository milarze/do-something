//! ACP Agent implementation: handles initialize, session/new, session/prompt,
//! session/cancel, with tool-call loop dispatched through the client.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification,
    ClientCapabilities, ContentBlock, ContentChunk, CurrentModeUpdate, EmbeddedResourceResource,
    Implementation, InitializeRequest, InitializeResponse, NewSessionRequest, NewSessionResponse,
    PermissionOption, PermissionOptionId, PermissionOptionKind, PromptCapabilities, PromptRequest,
    PromptResponse, RequestPermissionOutcome, RequestPermissionRequest, SessionId, SessionMode,
    SessionModeId, SessionModeState, SessionNotification, SessionUpdate, SetSessionModeRequest,
    SetSessionModeResponse, StopReason, TextContent, ToolCall as AcpToolCall, ToolCallContent,
    ToolCallId, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, PermissionMode};
use crate::llm::{ChatMessage, LlmClient, StreamEvent, ToolCall as LlmToolCall};
use crate::tools::{self, ToolOutcome};

/// Maximum number of LLM round-trips per prompt turn before we bail with
/// `StopReason::MaxTurnRequests`. Each tool batch counts as one request.
const MAX_TURN_REQUESTS: u32 = 16;

/// Per-session state.
struct SessionState {
    history: Vec<ChatMessage>,
    profile_name: String,
    cancel: CancellationToken,
}

#[derive(Clone)]
pub struct AgentState {
    pub config: Arc<Config>,
    pub llm: Arc<LlmClient>,
    /// Captured during `initialize`. Defaults to "no capabilities" until set.
    client_caps: Arc<Mutex<ClientCapabilities>>,
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
}

impl AgentState {
    pub fn new(config: Config, llm: LlmClient) -> Self {
        Self {
            config: Arc::new(config),
            llm: Arc::new(llm),
            client_caps: Arc::new(Mutex::new(ClientCapabilities::default())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn caps(&self) -> ClientCapabilities {
        self.client_caps.lock().unwrap().clone()
    }
}

pub async fn handle_initialize(
    state: AgentState,
    req: InitializeRequest,
) -> InitializeResponse {
    tracing::info!(
        "initialize: protocol_version={:?} client_caps={:?}",
        req.protocol_version,
        req.client_capabilities
    );
    *state.client_caps.lock().unwrap() = req.client_capabilities.clone();

    let prompt_caps = PromptCapabilities::new().embedded_context(true);
    let agent_caps = AgentCapabilities::new()
        .prompt_capabilities(prompt_caps)
        .load_session(false);

    InitializeResponse::new(req.protocol_version)
        .agent_capabilities(agent_caps)
        .agent_info(
            Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
                .title("do-something ACP agent"),
        )
}

pub async fn handle_new_session(
    state: AgentState,
    cx: ConnectionTo<Client>,
    req: NewSessionRequest,
) -> NewSessionResponse {
    let id = format!("sess_{}", uuid::Uuid::new_v4().simple());
    tracing::info!("session/new: id={} cwd={}", id, req.cwd.display());
    let (profile_name, profile) = state.config.active_profile();
    let mut history = Vec::new();
    if let Some(sys) = &profile.system_prompt {
        history.push(ChatMessage::system(sys.clone()));
    }
    let session = SessionState {
        history,
        profile_name: profile_name.to_string(),
        cancel: CancellationToken::new(),
    };
    let session_id = SessionId::new(id.clone());
    state
        .sessions
        .lock()
        .unwrap()
        .insert(id.clone(), session);

    // Build a SessionModeState that exposes each configured profile as a mode.
    // The currently active profile is the current mode.
    let modes = build_mode_state(&state, profile_name);

    // Best-effort: announce slash commands for this session. The notification
    // is a one-shot at session-open; we don't refresh it later.
    let cmds = AvailableCommandsUpdate::new(slash_commands());
    let _ = cx.send_notification(SessionNotification::new(
        session_id.clone(),
        SessionUpdate::AvailableCommandsUpdate(cmds),
    ));

    NewSessionResponse::new(session_id).modes(modes)
}

/// Handle `session/set_mode`: switch the active profile for this session.
/// Profiles are exposed as ACP "modes". Returns an error response if the
/// requested mode_id does not match a configured profile.
pub async fn handle_set_mode(
    state: AgentState,
    cx: ConnectionTo<Client>,
    req: SetSessionModeRequest,
) -> Result<SetSessionModeResponse, agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.to_string();
    let mode_id = req.mode_id.0.to_string();
    tracing::info!("session/set_mode: id={} mode={}", session_id_str, mode_id);

    // Validate the requested mode matches a known profile.
    if state.config.profile(&mode_id).is_none() {
        return Err(agent_client_protocol::Error::invalid_params()
            .data(format!("unknown mode/profile: {mode_id}")));
    }

    // Apply.
    {
        let mut sessions = state.sessions.lock().unwrap();
        let Some(s) = sessions.get_mut(&session_id_str) else {
            return Err(agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session: {session_id_str}")));
        };
        s.profile_name = mode_id.clone();
    }

    // Notify the client that the current mode has changed.
    let upd = CurrentModeUpdate::new(SessionModeId::new(mode_id));
    let _ = cx.send_notification(SessionNotification::new(
        req.session_id.clone(),
        SessionUpdate::CurrentModeUpdate(upd),
    ));

    Ok(SetSessionModeResponse::new())
}

fn build_mode_state(state: &AgentState, current: &str) -> SessionModeState {
    let available_modes: Vec<SessionMode> = state
        .config
        .profiles
        .iter()
        .map(|(name, p)| {
            let mut desc = format!("{} @ {}", p.model, p.base_url);
            if name == current {
                desc.push_str(" (active)");
            }
            SessionMode::new(SessionModeId::new(name.clone()), name.clone()).description(desc)
        })
        .collect();
    SessionModeState::new(SessionModeId::new(current.to_string()), available_modes)
}

fn slash_commands() -> Vec<AvailableCommand> {
    vec![
        AvailableCommand::new("help", "Show available slash commands"),
        AvailableCommand::new("profiles", "List configured LLM profiles and the active one"),
    ]
}

pub async fn handle_cancel(state: AgentState, notif: CancelNotification) {
    let id = notif.session_id.0.as_ref();
    tracing::info!("session/cancel: id={}", id);
    if let Some(s) = state.sessions.lock().unwrap().get(id) {
        s.cancel.cancel();
    }
}

/// The main prompt turn handler.
pub async fn handle_prompt(
    state: AgentState,
    cx: ConnectionTo<Client>,
    req: PromptRequest,
) -> PromptResponse {
    let session_id_str = req.session_id.0.to_string();
    tracing::info!("session/prompt: id={}", session_id_str);

    // Capture per-turn state and append the user message to history.
    let user_text = extract_text(&req.prompt);

    // Slash-command short-circuit: if the user typed `/help` or `/profiles`
    // (and only that — no extra prose preceding), we handle it locally and
    // skip the LLM round-trip entirely. Slash commands are advisory under
    // ACP — clients may strip them, pass them through verbatim, or pre-handle
    // them. Treat unknown slash inputs as regular prompts.
    if let Some(reply) = handle_slash_command(&state, &user_text, &req.session_id) {
        send_text(&cx, &req.session_id, &reply);
        return PromptResponse::new(StopReason::EndTurn);
    }

    let (profile, cancel) = {
        let mut sessions = state.sessions.lock().unwrap();
        let Some(s) = sessions.get_mut(&session_id_str) else {
            tracing::error!("unknown session id: {}", session_id_str);
            return PromptResponse::new(StopReason::EndTurn);
        };
        // Reset cancel token for this turn.
        s.cancel = CancellationToken::new();
        let cancel = s.cancel.clone();
        let profile = state
            .config
            .profile(&s.profile_name)
            .cloned()
            .expect("profile exists");
        // Build the user message: multimodal if the profile supports vision
        // and the prompt has at least one image; plain text otherwise.
        let user_msg = build_user_message(&req.prompt, &user_text, &profile);
        s.history.push(user_msg);
        (profile, cancel)
    };

    let caps = state.caps();
    let tool_defs = tools::tool_defs(&caps);
    let session_id = req.session_id.clone();

    // Tool-calling loop: keep asking the LLM until it stops emitting tool calls
    // (or hits the iteration cap, errors out, or is cancelled).
    let mut iterations: u32 = 0;
    let stop = loop {
        if cancel.is_cancelled() {
            break StopReason::Cancelled;
        }
        if iterations >= MAX_TURN_REQUESTS {
            tracing::warn!("max_turn_requests ({MAX_TURN_REQUESTS}) reached");
            break StopReason::MaxTurnRequests;
        }
        iterations += 1;

        // Snapshot history for this round.
        let history = {
            let sessions = state.sessions.lock().unwrap();
            let Some(s) = sessions.get(&session_id_str) else {
                return PromptResponse::new(StopReason::EndTurn);
            };
            s.history.clone()
        };

        // Open the streaming chat completion.
        let stream_res = state
            .llm
            .chat_stream(&profile, &history, &tool_defs, cancel.clone())
            .await;
        let mut stream = match stream_res {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("chat stream error: {e:#}");
                send_text(&cx, &session_id, &format!("\n[error: {e}]\n"));
                break StopReason::EndTurn;
            }
        };

        let mut assistant_text = String::new();
        let mut announced: HashMap<String, ToolCallId> = HashMap::new();
        let mut final_tool_calls: Vec<LlmToolCall> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut had_error = false;

        while let Some(ev) = stream.next().await {
            if cancel.is_cancelled() {
                break;
            }
            match ev {
                StreamEvent::TextDelta(t) => {
                    assistant_text.push_str(&t);
                    send_text(&cx, &session_id, &t);
                }
                StreamEvent::ToolCallStart { id, name } => {
                    let tc_id = ToolCallId::new(id.clone());
                    announced.insert(id.clone(), tc_id.clone());
                    let kind = tool_kind_for(&name);
                    let title = format!("{name}");
                    let acp_tc = AcpToolCall::new(tc_id, title)
                        .kind(kind)
                        .status(ToolCallStatus::Pending);
                    let _ = cx.send_notification(SessionNotification::new(
                        session_id.clone(),
                        SessionUpdate::ToolCall(acp_tc),
                    ));
                }
                StreamEvent::Done {
                    finish_reason: fr,
                    tool_calls,
                } => {
                    finish_reason = fr;
                    final_tool_calls = tool_calls;
                    break;
                }
                StreamEvent::Cancelled => {
                    break;
                }
                StreamEvent::ContextOverflow(msg) => {
                    tracing::warn!("context overflow: {msg}");
                    send_text(
                        &cx,
                        &session_id,
                        "\n[context window exceeded — stopping turn]\n",
                    );
                    finish_reason = Some("length".to_string());
                    break;
                }
                StreamEvent::Error(e) => {
                    tracing::error!("stream error: {e}");
                    send_text(&cx, &session_id, &format!("\n[stream error: {e}]\n"));
                    had_error = true;
                    break;
                }
            }
        }

        if cancel.is_cancelled() {
            // Mark any pending tool calls as cancelled before returning.
            for (_id_str, tc_id) in announced.iter() {
                let upd = ToolCallUpdate::new(
                    tc_id.clone(),
                    ToolCallUpdateFields::new().status(ToolCallStatus::Failed),
                );
                let _ = cx.send_notification(SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::ToolCallUpdate(upd),
                ));
            }
            break StopReason::Cancelled;
        }
        if had_error {
            break StopReason::EndTurn;
        }

        // Push the assistant turn (with any tool calls) into history.
        {
            let mut sessions = state.sessions.lock().unwrap();
            if let Some(s) = sessions.get_mut(&session_id_str) {
                let content = if assistant_text.is_empty() {
                    None
                } else {
                    Some(assistant_text.clone())
                };
                s.history
                    .push(ChatMessage::assistant(content, final_tool_calls.clone()));
            }
        }

        // No tool calls => we're done.
        if final_tool_calls.is_empty() {
            if finish_reason.as_deref() == Some("length") {
                break StopReason::MaxTokens;
            }
            break StopReason::EndTurn;
        }

        // Execute each tool call in order.
        for tc in &final_tool_calls {
            if cancel.is_cancelled() {
                break;
            }
            let tc_id = announced
                .get(&tc.id)
                .cloned()
                .unwrap_or_else(|| ToolCallId::new(tc.id.clone()));
            let name = &tc.function.name;
            let args = &tc.function.arguments;

            // If the tool wasn't announced via ToolCallStart, send a ToolCall now.
            if !announced.contains_key(&tc.id) {
                let kind = tool_kind_for(name);
                let acp_tc = AcpToolCall::new(tc_id.clone(), name.clone())
                    .kind(kind)
                    .status(ToolCallStatus::Pending)
                    .raw_input(serde_json::from_str::<serde_json::Value>(args).ok());
                let _ = cx.send_notification(SessionNotification::new(
                    session_id.clone(),
                    SessionUpdate::ToolCall(acp_tc),
                ));
            }

            // Capability check.
            if !tools::tool_supported(name, &caps) {
                let msg = format!("tool `{name}` not supported by client capabilities");
                send_tool_update_failed(&cx, &session_id, &tc_id, &msg);
                append_tool_result(&state, &session_id_str, &tc.id, &msg);
                continue;
            }

            // Permission gate for sensitive tools — only when this profile
            // opts into agent-managed permissions. In client-managed mode
            // (the default) we just make the call and let the client enforce
            // its own policy; a rejection comes back as an error response and
            // is reported as a Failed tool result.
            if profile.permission_mode == PermissionMode::Agent
                && tools::tool_requires_permission(name)
            {
                let perm_req = RequestPermissionRequest::new(
                    session_id.clone(),
                    ToolCallUpdate::new(
                        tc_id.clone(),
                        ToolCallUpdateFields::new()
                            .title(name.clone())
                            .kind(tool_kind_for(name))
                            .status(ToolCallStatus::Pending)
                            .raw_input(serde_json::from_str::<serde_json::Value>(args).ok()),
                    ),
                    vec![
                        PermissionOption::new(
                            PermissionOptionId::new("allow_once"),
                            "Allow once",
                            PermissionOptionKind::AllowOnce,
                        ),
                        PermissionOption::new(
                            PermissionOptionId::new("reject_once"),
                            "Reject",
                            PermissionOptionKind::RejectOnce,
                        ),
                    ],
                );
                let perm_res = tokio::select! {
                    r = cx.send_request(perm_req).block_task() => r,
                    _ = cancel.cancelled() => {
                        send_tool_update_failed(&cx, &session_id, &tc_id, "cancelled");
                        append_tool_result(&state, &session_id_str, &tc.id, "cancelled");
                        continue;
                    }
                };
                let allowed = match perm_res {
                    Ok(resp) => match resp.outcome {
                        RequestPermissionOutcome::Selected(sel) => {
                            sel.option_id.0.as_ref() == "allow_once"
                        }
                        RequestPermissionOutcome::Cancelled => false,
                        _ => false,
                    },
                    Err(e) => {
                        tracing::warn!("permission request failed: {e}");
                        false
                    }
                };
                if !allowed {
                    let msg = "user rejected tool call";
                    send_tool_update_failed(&cx, &session_id, &tc_id, msg);
                    append_tool_result(&state, &session_id_str, &tc.id, msg);
                    continue;
                }
            }

            // Mark in_progress.
            let upd = ToolCallUpdate::new(
                tc_id.clone(),
                ToolCallUpdateFields::new().status(ToolCallStatus::InProgress),
            );
            let _ = cx.send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::ToolCallUpdate(upd),
            ));

            // Invoke.
            let outcome: ToolOutcome =
                tools::invoke_tool(name, args, &cx, &session_id, cancel.clone()).await;

            // Report result.
            let status = if outcome.ok {
                ToolCallStatus::Completed
            } else {
                ToolCallStatus::Failed
            };
            let content_block = ContentBlock::Text(TextContent::new(outcome.text.clone()));
            let upd = ToolCallUpdate::new(
                tc_id.clone(),
                ToolCallUpdateFields::new()
                    .status(status)
                    .content(vec![ToolCallContent::Content(
                        agent_client_protocol::schema::Content::new(content_block),
                    )]),
            );
            let _ = cx.send_notification(SessionNotification::new(
                session_id.clone(),
                SessionUpdate::ToolCallUpdate(upd),
            ));

            // Feed result back to the model.
            append_tool_result(&state, &session_id_str, &tc.id, &outcome.text);
        }

        // Loop again — let the model react to the tool results.
    };

    PromptResponse::new(stop)
}

fn tool_kind_for(name: &str) -> ToolKind {
    match name {
        tools::TOOL_READ_FILE | tools::TOOL_LIST_FILES => ToolKind::Read,
        tools::TOOL_WRITE_FILE => ToolKind::Edit,
        tools::TOOL_RUN_SHELL => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

fn append_tool_result(state: &AgentState, session_id: &str, tool_call_id: &str, text: &str) {
    let mut sessions = state.sessions.lock().unwrap();
    if let Some(s) = sessions.get_mut(session_id) {
        s.history
            .push(ChatMessage::tool_result(tool_call_id, text.to_string()));
    }
}

fn send_tool_update_failed(
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    tc_id: &ToolCallId,
    msg: &str,
) {
    let upd = ToolCallUpdate::new(
        tc_id.clone(),
        ToolCallUpdateFields::new()
            .status(ToolCallStatus::Failed)
            .content(vec![ToolCallContent::Content(
                agent_client_protocol::schema::Content::new(ContentBlock::Text(TextContent::new(
                    msg.to_string(),
                ))),
            )]),
    );
    let _ = cx.send_notification(SessionNotification::new(
        session_id.clone(),
        SessionUpdate::ToolCallUpdate(upd),
    ));
}

/// Build the user-side `ChatMessage` for one prompt turn. If the profile
/// declares `supports_vision = true` and the prompt has at least one image
/// block, we send an OpenAI multimodal `content` array (text + `image_url`
/// data URIs). Otherwise we send the plain text extraction, with images
/// summarised as `[image: <mime>]` placeholders inside that text.
fn build_user_message(
    blocks: &[ContentBlock],
    plain_text: &str,
    profile: &crate::config::Profile,
) -> ChatMessage {
    let has_image = blocks.iter().any(|b| matches!(b, ContentBlock::Image(_)));
    if !profile.supports_vision || !has_image {
        return ChatMessage::user(plain_text.to_string());
    }

    let mut parts: Vec<serde_json::Value> = Vec::new();
    for b in blocks {
        match b {
            ContentBlock::Text(t) => {
                parts.push(serde_json::json!({"type": "text", "text": t.text}));
            }
            ContentBlock::Image(img) => {
                // OpenAI expects either a public https URL or a data URI.
                // ACP delivers base64 in `data` plus the `mime_type`; build
                // a data URI. Some servers also accept `{"url": "..."}` as
                // a bare object; we follow the documented shape here.
                let data_uri = format!("data:{};base64,{}", img.mime_type, img.data);
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": data_uri }
                }));
            }
            ContentBlock::ResourceLink(rl) => {
                parts.push(serde_json::json!({
                    "type": "text",
                    "text": format!("[resource: {}]", rl.uri)
                }));
            }
            ContentBlock::Resource(er) => {
                if let EmbeddedResourceResource::TextResourceContents(tr) = &er.resource {
                    let mime = tr.mime_type.as_deref().unwrap_or("text/plain");
                    parts.push(serde_json::json!({
                        "type": "text",
                        "text": format!("[resource: {} ({mime})]\n{}", tr.uri, tr.text),
                    }));
                }
            }
            _ => {}
        }
    }
    ChatMessage::user_multipart(parts)
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    let mut buf = String::new();
    for b in blocks {
        match b {
            ContentBlock::Text(t) => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&t.text);
            }
            ContentBlock::ResourceLink(rl) => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&format!("[resource: {}]", rl.uri));
            }
            ContentBlock::Image(img) => {
                // Profile is non-vision (or this path is hit before profile
                // resolution). Summarise so the model at least knows an image
                // was attached.
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&format!("[image: {}]", img.mime_type));
            }
            ContentBlock::Resource(er) => {
                // Pull the actual text for text-typed resources, with a small
                // header noting uri + mime so the model has provenance.
                // Blob resources are skipped (binary content not modelled here).
                if let EmbeddedResourceResource::TextResourceContents(tr) = &er.resource {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    let mime = tr.mime_type.as_deref().unwrap_or("text/plain");
                    buf.push_str(&format!(
                        "[resource: {} ({mime})]\n{}",
                        tr.uri, tr.text
                    ));
                }
            }
            _ => {}
        }
    }
    buf
}

/// Detect and handle a leading-`/` slash command. Returns the reply text to
/// send back to the user, or `None` if the input is not a recognised command
/// (in which case it should be passed through to the LLM as a normal prompt).
fn handle_slash_command(
    state: &AgentState,
    user_text: &str,
    _session_id: &SessionId,
) -> Option<String> {
    let trimmed = user_text.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    // Split into command and (ignored) tail. We only honour single-word commands.
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let cmd = parts.next()?;
    match cmd {
        "/help" => Some(format!(
            "Available slash commands:\n\
             - /help — show this help\n\
             - /profiles — list configured LLM profiles\n\
             \n\
             Tip: switch profile via the `session/set_mode` ACP method \
             (your client may expose this as a mode picker).\n"
        )),
        "/profiles" => {
            let (active, _) = state.config.active_profile();
            // Active profile per *session* may differ from config default if the
            // client called set_mode. We don't have session_id-level lookup here
            // for the active mode without locking — this command is informational
            // only, so we report the configured set and the config-level default.
            let mut out = String::from("Configured profiles:\n");
            for (name, p) in state.config.profiles.iter() {
                let marker = if name == active { " (default)" } else { "" };
                out.push_str(&format!(
                    "- {name}{marker}: model={} base_url={}\n",
                    p.model, p.base_url
                ));
            }
            Some(out)
        }
        _ => None,
    }
}

fn send_text(cx: &ConnectionTo<Client>, session_id: &SessionId, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let notif = SessionNotification::new(
        session_id.clone(),
        SessionUpdate::AgentMessageChunk(chunk),
    );
    if let Err(e) = cx.send_notification(notif) {
        tracing::warn!("send_notification failed: {e}");
    }
}

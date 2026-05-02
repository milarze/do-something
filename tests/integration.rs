//! End-to-end integration tests that spawn the `do-something` binary, point
//! it at a mock OpenAI-compatible HTTP server, and drive it as an ACP client.
//!
//! These exercise the wire protocol on both sides without any real LLM. The
//! mock server handles `/health`, `/props`, `/v1/models` (preflight) and
//! `POST /v1/chat/completions` with scripted SSE responses.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion,
    SessionNotification, SessionUpdate, SetSessionModeRequest, TextContent,
};
use agent_client_protocol::{Agent, Client, ConnectionTo};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

#[path = "common/mock_server.rs"]
mod mock_server;
use mock_server::{MockServer, ScriptedResponse};

/// Spawn the agent binary with the given env vars, returning its child handle
/// and stdio. The binary path is set at compile time by Cargo.
fn spawn_agent(
    env: &[(&str, &str)],
) -> (
    tokio::process::Child,
    tokio::process::ChildStdin,
    tokio::process::ChildStdout,
) {
    let bin = env!("CARGO_BIN_EXE_do-something");
    let mut cmd = tokio::process::Command::new(bin);
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    // Clear inherited config to avoid picking up the user's real config.
    cmd.env_remove("DO_SOMETHING_CONFIG");
    cmd.env_remove("DO_SOMETHING_BASE_URL");
    cmd.env_remove("DO_SOMETHING_MODEL");
    cmd.env_remove("DO_SOMETHING_PROFILE");
    cmd.env_remove("DO_SOMETHING_API_KEY");
    cmd.env("DO_SOMETHING_LOG", "warn"); // quiet but still surface errors
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn agent");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    (child, stdin, stdout)
}

/// Collected `SessionNotification`s from the agent, shared between the
/// client's notification handler and the test body.
type Notifications = Arc<Mutex<Vec<SessionUpdate>>>;

fn new_notifications() -> Notifications {
    Arc::new(Mutex::new(Vec::new()))
}

/// Build a minimal client that records all session notifications and never
/// asks for permissions (we run with the default `client` permission mode).
fn record_notifications(notifs: Notifications, update: SessionUpdate) {
    notifs.lock().unwrap().push(update);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn smoke_text_prompt_round_trip() {
    let server = MockServer::start().await;
    server
        .push_chat(ScriptedResponse::text_only(
            ["Hello, ", "world!"].into_iter().map(String::from).collect(),
        ))
        .await;

    let base_url = format!("http://127.0.0.1:{}/v1", server.port());
    let bogus_cfg = "/dev/null"; // forces builtin defaults; env overrides apply
    let (mut child, stdin, stdout) = spawn_agent(&[
        ("DO_SOMETHING_CONFIG", bogus_cfg),
        ("DO_SOMETHING_BASE_URL", &base_url),
        ("DO_SOMETHING_MODEL", "mock-model"),
    ]);

    let notifs = new_notifications();
    let notifs_handler = notifs.clone();

    let transport =
        agent_client_protocol::ByteStreams::new(stdin.compat_write(), stdout.compat());

    let result: Result<(), Box<dyn std::error::Error + Send + Sync>> = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                record_notifications(notifs_handler.clone(), n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(transport, |conn: ConnectionTo<Agent>| async move {
            let init = conn
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            assert!(init.agent_info.is_some(), "agent_info should be set");

            let new_sess = conn
                .send_request(NewSessionRequest::new(PathBuf::from("/")))
                .block_task()
                .await?;
            assert!(
                new_sess.modes.is_some(),
                "session/new should advertise modes (profiles)"
            );

            let prompt = conn
                .send_request(PromptRequest::new(
                    new_sess.session_id.clone(),
                    vec![ContentBlock::Text(TextContent::new("hi".to_string()))],
                ))
                .block_task()
                .await?;

            assert_eq!(
                format!("{:?}", prompt.stop_reason),
                "EndTurn",
                "expected clean EndTurn after a text-only response"
            );
            Ok(())
        })
        .await
        .map_err(|e| e.into());

    // Surface client errors before child cleanup.
    if let Err(e) = result {
        let _ = child.kill().await;
        panic!("client driver failed: {e}");
    }

    // The agent should have streamed our two text deltas as agent_message_chunks.
    let updates = notifs.lock().unwrap().clone();
    let chunks: Vec<String> = updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(c) => match &c.content {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let combined = chunks.join("");
    assert!(
        combined.contains("Hello, world!"),
        "expected streamed text to include scripted response, got: {combined:?}"
    );

    let _ = child.kill().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slash_command_help_short_circuits() {
    let server = MockServer::start().await;
    // Don't push any scripted chat response — if the agent contacts the LLM,
    // the mock will reject and the test will fail.

    let base_url = format!("http://127.0.0.1:{}/v1", server.port());
    let (mut child, stdin, stdout) = spawn_agent(&[
        ("DO_SOMETHING_BASE_URL", &base_url),
        ("DO_SOMETHING_MODEL", "mock-model"),
    ]);

    let notifs = new_notifications();
    let notifs_handler = notifs.clone();

    let transport =
        agent_client_protocol::ByteStreams::new(stdin.compat_write(), stdout.compat());

    let result: Result<(), Box<dyn std::error::Error + Send + Sync>> = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                record_notifications(notifs_handler.clone(), n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(transport, |conn: ConnectionTo<Agent>| async move {
            conn.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_sess = conn
                .send_request(NewSessionRequest::new(PathBuf::from("/")))
                .block_task()
                .await?;

            let resp = conn
                .send_request(PromptRequest::new(
                    new_sess.session_id.clone(),
                    vec![ContentBlock::Text(TextContent::new("/help".to_string()))],
                ))
                .block_task()
                .await?;

            assert_eq!(format!("{:?}", resp.stop_reason), "EndTurn");
            Ok(())
        })
        .await
        .map_err(|e| e.into());

    if let Err(e) = result {
        let _ = child.kill().await;
        panic!("client driver failed: {e}");
    }

    let updates = notifs.lock().unwrap().clone();
    let combined: String = updates
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(c) => match &c.content {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(
        combined.contains("Available slash commands"),
        "expected /help reply, got: {combined:?}"
    );

    // Critical: the agent must not have contacted the mock LLM for slash cmds.
    let chat_hits = server.chat_hits().await;
    assert_eq!(
        chat_hits, 0,
        "slash command should short-circuit before LLM call (got {chat_hits} hits)"
    );

    let _ = child.kill().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_mode_switches_active_profile() {
    let server = MockServer::start().await;

    // Two profiles with the same mock backend, distinguishable by model name.
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg_path = dir.path().join("config.toml");
    let cfg = format!(
        r#"
default_profile = "first"
[profiles.first]
base_url = "http://127.0.0.1:{port}/v1"
model = "first-model"
[profiles.second]
base_url = "http://127.0.0.1:{port}/v1"
model = "second-model"
"#,
        port = server.port()
    );
    std::fs::write(&cfg_path, cfg).unwrap();

    let (mut child, stdin, stdout) = spawn_agent(&[(
        "DO_SOMETHING_CONFIG",
        cfg_path.to_str().unwrap(),
    )]);

    let notifs = new_notifications();
    let notifs_handler = notifs.clone();

    let transport =
        agent_client_protocol::ByteStreams::new(stdin.compat_write(), stdout.compat());

    let result: Result<(), Box<dyn std::error::Error + Send + Sync>> = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                record_notifications(notifs_handler.clone(), n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(transport, |conn: ConnectionTo<Agent>| async move {
            conn.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let new_sess = conn
                .send_request(NewSessionRequest::new(PathBuf::from("/")))
                .block_task()
                .await?;
            let modes = new_sess.modes.expect("modes advertised");
            assert_eq!(
                modes.available_modes.len(),
                2,
                "expected both profiles as modes, got {:?}",
                modes.available_modes
            );
            assert_eq!(modes.current_mode_id.0.as_ref(), "first");

            // Switch to "second".
            conn.send_request(SetSessionModeRequest::new(
                new_sess.session_id.clone(),
                agent_client_protocol::schema::SessionModeId::new("second"),
            ))
            .block_task()
            .await?;

            // Give the notification a moment to propagate.
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(())
        })
        .await
        .map_err(|e| e.into());

    if let Err(e) = result {
        let _ = child.kill().await;
        panic!("client driver failed: {e}");
    }

    let updates = notifs.lock().unwrap().clone();
    let saw_mode_update = updates.iter().any(|u| {
        matches!(u, SessionUpdate::CurrentModeUpdate(m) if m.current_mode_id.0.as_ref() == "second")
    });
    assert!(
        saw_mode_update,
        "expected CurrentModeUpdate(second), got: {updates:?}"
    );

    let _ = child.kill().await;
}

// Suppress unused-import warnings if the module shrinks later.

mod agent;
mod config;
mod llm;
mod tools;

use agent_client_protocol::schema::{
    CancelNotification, InitializeRequest, NewSessionRequest, PromptRequest, SetSessionModeRequest,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Dispatch, Result};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::agent::{
    AgentState, handle_cancel, handle_initialize, handle_new_session, handle_prompt,
    handle_set_mode,
};
use crate::config::{Config, Profile};
use crate::llm::{LlmClient, PreflightReport};

#[tokio::main]
async fn main() -> Result<()> {
    // CRITICAL: log to stderr only; stdout is the JSON-RPC channel.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("DO_SOMETHING_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to load config: {e:#}");
            std::process::exit(2);
        }
    };
    let (active_name, active_profile) = cfg.active_profile();
    tracing::info!(
        "starting do-something agent: profile={} model={} base_url={}",
        active_name,
        active_profile.model,
        active_profile.base_url
    );

    let llm = LlmClient::new().expect("LlmClient init");

    // Best-effort preflight against the active profile. Logs to stderr.
    // We don't fail startup if the server is offline — the agent should still
    // come up so the client can talk to it (and surface errors per-prompt).
    {
        let report = llm.preflight(active_profile).await;
        log_preflight(active_name, active_profile, &report);
    }

    let state = AgentState::new(cfg, llm);

    let s_init = state.clone();
    let s_new = state.clone();
    let s_prompt = state.clone();
    let s_cancel = state.clone();
    let s_set_mode = state.clone();

    Agent
        .builder()
        .name("do-something")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx: ConnectionTo<Client>| {
                let resp = handle_initialize(s_init.clone(), req).await;
                responder.respond(resp)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                let resp = handle_new_session(s_new.clone(), cx, req).await;
                responder.respond(resp)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                let resp = handle_prompt(s_prompt.clone(), cx, req).await;
                responder.respond(resp)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: SetSessionModeRequest, responder, cx: ConnectionTo<Client>| {
                match handle_set_mode(s_set_mode.clone(), cx, req).await {
                    Ok(resp) => responder.respond(resp),
                    Err(e) => responder.respond_with_error(e),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notif: CancelNotification, _cx: ConnectionTo<Client>| {
                handle_cancel(s_cancel.clone(), notif).await;
                Ok::<(), agent_client_protocol::Error>(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_dispatch(
            async move |message: Dispatch, cx: ConnectionTo<Client>| {
                tracing::debug!("unhandled message");
                message.respond_with_error(
                    agent_client_protocol::util::internal_error("method not implemented"),
                    cx,
                )
            },
            agent_client_protocol::on_receive_dispatch!(),
        )
        .connect_to(agent_client_protocol::ByteStreams::new(
            tokio::io::stdout().compat_write(),
            tokio::io::stdin().compat(),
        ))
        .await
}

fn log_preflight(profile_name: &str, profile: &Profile, report: &PreflightReport) {
    let any_signal =
        report.health.is_some() || report.props.is_some() || !report.models.is_empty();
    if !any_signal {
        tracing::warn!(
            "preflight: no response from server at {} — agent will start but prompts will fail until it is reachable",
            profile.base_url
        );
        return;
    }

    if let Some(h) = &report.health {
        match h.status {
            200 => tracing::info!("preflight: /health OK"),
            503 => tracing::warn!("preflight: /health 503 — model still loading"),
            s => tracing::warn!("preflight: /health unexpected status {s}"),
        }
    }

    if let Some(p) = &report.props {
        let model_path = p.model_path.as_deref().unwrap_or("<unknown>");
        let build = p.build_info.as_deref().unwrap_or("<unknown>");
        let slots = p.total_slots.unwrap_or(0);
        tracing::info!(
            "preflight: llama.cpp build={build} model_path={model_path} slots={slots}"
        );
        if slots <= 1 {
            tracing::warn!(
                "preflight: server has only {slots} slot — concurrent requests will block. Consider --parallel N"
            );
        }
        if p.jinja_enabled() {
            if p.supports_tools() {
                tracing::info!(
                    "preflight: --jinja enabled and chat template advertises tool-call support"
                );
            } else {
                tracing::warn!(
                    "preflight: --jinja enabled but chat template does not advertise tool-call support — tools may not work"
                );
            }
        } else {
            tracing::warn!(
                "preflight: chat_template_caps not detected — start llama-server with --jinja for native tool calls"
            );
        }
    } else {
        tracing::debug!(
            "preflight: /props not available (non-llama.cpp server or older build) — skipping jinja/tools detection"
        );
    }

    if !report.models.is_empty() {
        tracing::info!(
            "preflight: /v1/models returned {} model(s); first id = {}",
            report.models.len(),
            report.models[0]
        );
        // Soft check: warn if configured profile.model doesn't appear in the
        // list (purely informational — llama.cpp ignores the field).
        let configured = &profile.model;
        let matches_any = report
            .models
            .iter()
            .any(|m| m == configured || m.ends_with(configured));
        if !matches_any {
            tracing::debug!(
                "preflight: configured model `{configured}` does not match server-reported ids (this is usually fine for llama.cpp)"
            );
        }
    }

    tracing::debug!("preflight summary for profile `{profile_name}`: {report:?}");
}

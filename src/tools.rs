//! Tool registry. Every tool executes by sending requests to the client.
//! No native filesystem or process access is performed by the agent.
//!
//! Tools are advertised to the LLM only when the corresponding client
//! capability was negotiated during `initialize`.

use agent_client_protocol::schema::{
    ClientCapabilities, CreateTerminalRequest, KillTerminalRequest, ReadTextFileRequest,
    ReleaseTerminalRequest, SessionId, TerminalOutputRequest, WaitForTerminalExitRequest,
    WriteTextFileRequest,
};
use agent_client_protocol::{Client, ConnectionTo};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

use crate::llm::{ToolDef, ToolDefFunction};

/// Static tool names — also used as keys when dispatching.
pub const TOOL_READ_FILE: &str = "read_file";
pub const TOOL_WRITE_FILE: &str = "write_file";
pub const TOOL_LIST_FILES: &str = "list_files";
pub const TOOL_RUN_SHELL: &str = "run_shell";

/// Build the OpenAI-format tool list based on what the client supports.
/// Returns an empty Vec when the client advertised no relevant capabilities.
pub fn tool_defs(caps: &ClientCapabilities) -> Vec<ToolDef> {
    let mut out = Vec::new();

    if caps.fs.read_text_file {
        out.push(ToolDef {
            kind: "function",
            function: ToolDefFunction {
                name: TOOL_READ_FILE.into(),
                description:
                    "Read the contents of a text file from the user's workspace. Path must be absolute.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute file path" },
                        "line": { "type": "integer", "description": "1-based start line (optional)" },
                        "limit": { "type": "integer", "description": "Maximum lines to read (optional)" }
                    },
                    "required": ["path"]
                }),
            },
        });
    }

    if caps.fs.write_text_file {
        out.push(ToolDef {
            kind: "function",
            function: ToolDefFunction {
                name: TOOL_WRITE_FILE.into(),
                description:
                    "Write the given text content to a file in the user's workspace, creating it if needed. Path must be absolute. Requires user permission.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute file path" },
                        "content": { "type": "string", "description": "Full new file contents" }
                    },
                    "required": ["path", "content"]
                }),
            },
        });
    }

    if caps.terminal {
        out.push(ToolDef {
            kind: "function",
            function: ToolDefFunction {
                name: TOOL_LIST_FILES.into(),
                description:
                    "List the files in a directory of the user's workspace. Path must be absolute.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute directory path" }
                    },
                    "required": ["path"]
                }),
            },
        });
        out.push(ToolDef {
            kind: "function",
            function: ToolDefFunction {
                name: TOOL_RUN_SHELL.into(),
                description:
                    "Run a shell command in the user's workspace and return its output. Requires user permission.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Command to execute (e.g. /bin/sh)" },
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Command arguments"
                        },
                        "cwd": { "type": "string", "description": "Absolute working directory (optional)" }
                    },
                    "required": ["command"]
                }),
            },
        });
    }

    out
}

/// Whether the model is allowed to invoke this tool given current capabilities.
pub fn tool_supported(name: &str, caps: &ClientCapabilities) -> bool {
    match name {
        TOOL_READ_FILE => caps.fs.read_text_file,
        TOOL_WRITE_FILE => caps.fs.write_text_file,
        TOOL_LIST_FILES | TOOL_RUN_SHELL => caps.terminal,
        _ => false,
    }
}

/// Whether this tool requires explicit user permission before invocation.
pub fn tool_requires_permission(name: &str) -> bool {
    matches!(name, TOOL_WRITE_FILE | TOOL_RUN_SHELL)
}

/// Result of a tool execution to be reported back to the LLM and the client.
pub struct ToolOutcome {
    /// String content for the model.
    pub text: String,
    /// True if execution was successful (false marks ToolCallStatus::Failed).
    pub ok: bool,
}

impl ToolOutcome {
    pub fn ok(text: impl Into<String>) -> Self {
        Self { text: text.into(), ok: true }
    }
    pub fn err(text: impl Into<String>) -> Self {
        Self { text: text.into(), ok: false }
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    #[serde(default)]
    line: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct ListFilesArgs {
    path: String,
}

#[derive(Deserialize)]
struct RunShellArgs {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<String>,
}

/// Dispatch a tool call. All paths to client come through `cx`.
pub async fn invoke_tool(
    name: &str,
    args_json: &str,
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    cancel: CancellationToken,
) -> ToolOutcome {
    let res = match name {
        TOOL_READ_FILE => invoke_read_file(args_json, cx, session_id, cancel).await,
        TOOL_WRITE_FILE => invoke_write_file(args_json, cx, session_id, cancel).await,
        TOOL_LIST_FILES => invoke_list_files(args_json, cx, session_id, cancel).await,
        TOOL_RUN_SHELL => invoke_run_shell(args_json, cx, session_id, cancel).await,
        _ => Err(anyhow!("unknown tool: {name}")),
    };
    match res {
        Ok(text) => ToolOutcome::ok(text),
        Err(e) => ToolOutcome::err(format!("error: {e:#}")),
    }
}

async fn invoke_read_file(
    args_json: &str,
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    cancel: CancellationToken,
) -> Result<String> {
    let a: ReadFileArgs = serde_json::from_str(args_json)
        .map_err(|e| anyhow!("invalid arguments: {e}"))?;
    let path = absolute_path(&a.path)?;
    let mut req = ReadTextFileRequest::new(session_id.clone(), path);
    req.line = a.line;
    req.limit = a.limit;
    let resp = await_with_cancel(cx.send_request(req).block_task(), cancel).await?;
    Ok(resp.content)
}

async fn invoke_write_file(
    args_json: &str,
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    cancel: CancellationToken,
) -> Result<String> {
    let a: WriteFileArgs = serde_json::from_str(args_json)
        .map_err(|e| anyhow!("invalid arguments: {e}"))?;
    let path = absolute_path(&a.path)?;
    let bytes = a.content.len();
    let req = WriteTextFileRequest::new(session_id.clone(), path.clone(), a.content);
    let _ = await_with_cancel(cx.send_request(req).block_task(), cancel).await?;
    Ok(format!("wrote {bytes} bytes to {}", path.display()))
}

async fn invoke_list_files(
    args_json: &str,
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    cancel: CancellationToken,
) -> Result<String> {
    let a: ListFilesArgs = serde_json::from_str(args_json)
        .map_err(|e| anyhow!("invalid arguments: {e}"))?;
    let _ = absolute_path(&a.path)?;
    // No native fs: shell out via terminal/* (also gated on terminal capability).
    run_shell_via_client(
        cx,
        session_id,
        cancel,
        "/bin/sh".into(),
        vec!["-c".into(), format!("ls -la {}", shell_escape(&a.path))],
        None,
    )
    .await
}

async fn invoke_run_shell(
    args_json: &str,
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    cancel: CancellationToken,
) -> Result<String> {
    let a: RunShellArgs = serde_json::from_str(args_json)
        .map_err(|e| anyhow!("invalid arguments: {e}"))?;
    run_shell_via_client(cx, session_id, cancel, a.command, a.args, a.cwd).await
}

async fn run_shell_via_client(
    cx: &ConnectionTo<Client>,
    session_id: &SessionId,
    cancel: CancellationToken,
    command: String,
    args: Vec<String>,
    cwd: Option<String>,
) -> Result<String> {
    let mut req = CreateTerminalRequest::new(session_id.clone(), command).args(args);
    if let Some(c) = cwd {
        req = req.cwd(PathBuf::from(c));
    }
    req = req.output_byte_limit(64 * 1024u64);

    let create_resp = await_with_cancel(cx.send_request(req).block_task(), cancel.clone()).await?;
    let term_id = create_resp.terminal_id.clone();

    // Wait for exit (cancellable).
    let wait_req = WaitForTerminalExitRequest::new(session_id.clone(), term_id.clone());
    let wait_fut = cx.send_request(wait_req).block_task();
    let exit_status = match await_with_cancel(wait_fut, cancel.clone()).await {
        Ok(r) => r.exit_status,
        Err(e) => {
            // Cancelled: do best-effort kill, then release.
            let _ = cx
                .send_request(KillTerminalRequest::new(
                    session_id.clone(),
                    term_id.clone(),
                ))
                .block_task()
                .await;
            let _ = cx
                .send_request(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    term_id.clone(),
                ))
                .block_task()
                .await;
            return Err(e);
        }
    };

    // Fetch final output, then release.
    let out_resp = cx
        .send_request(TerminalOutputRequest::new(
            session_id.clone(),
            term_id.clone(),
        ))
        .block_task()
        .await
        .map_err(|e| anyhow!("terminal/output failed: {e}"))?;
    let _ = cx
        .send_request(ReleaseTerminalRequest::new(
            session_id.clone(),
            term_id.clone(),
        ))
        .block_task()
        .await;

    let mut out = String::new();
    if let Some(code) = exit_status.exit_code {
        out.push_str(&format!("exit_code: {code}\n"));
    }
    if let Some(sig) = &exit_status.signal {
        out.push_str(&format!("signal: {sig}\n"));
    }
    if out_resp.truncated {
        out.push_str("(output truncated)\n");
    }
    out.push_str("---\n");
    out.push_str(&out_resp.output);
    Ok(out)
}

fn absolute_path(p: &str) -> Result<PathBuf> {
    let pb = PathBuf::from(p);
    if !pb.is_absolute() {
        return Err(anyhow!("path must be absolute (got `{p}`)"));
    }
    Ok(pb)
}

fn shell_escape(s: &str) -> String {
    // POSIX-safe single-quoted escape: 'foo'\''bar' for embedded single quotes.
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

async fn await_with_cancel<F, T>(fut: F, cancel: CancellationToken) -> Result<T>
where
    F: std::future::Future<Output = std::result::Result<T, agent_client_protocol::Error>>,
{
    tokio::select! {
        r = fut => r.map_err(|e| anyhow!("client request failed: {e}")),
        _ = cancel.cancelled() => Err(anyhow!("cancelled")),
    }
}

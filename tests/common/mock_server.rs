//! Tiny hand-rolled HTTP/1.1 mock for the OpenAI / llama.cpp endpoints the
//! agent uses. Bound to 127.0.0.1 on an OS-assigned port. Each test gets
//! its own server.
//!
//! Routes:
//! - `GET  /health`              → `200 {"status":"ok"}`
//! - `GET  /props`               → `200` with a llama.cpp-shaped props body
//!                                  declaring `--jinja` and tool support.
//! - `GET  /v1/models`           → `200 {"object":"list","data":[...]}`
//! - `POST /v1/chat/completions` → consumes the next scripted response from
//!                                  the queue (FIFO). If the queue is empty
//!                                  returns 500 — useful for asserting that
//!                                  no LLM call was made.
//!
//! Scripted responses are sent as Server-Sent Events terminated by
//! `data: [DONE]`. Only text deltas are supported in this round (no tool
//! calls); add later as needed.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

pub struct ScriptedResponse {
    /// Sequence of text deltas to stream as SSE chunks. Each becomes one
    /// `data: {...}\n\n` event.
    pub text_deltas: Vec<String>,
}

impl ScriptedResponse {
    pub fn text_only(text_deltas: Vec<String>) -> Self {
        Self { text_deltas }
    }
}

#[derive(Clone)]
pub struct MockServer {
    inner: Arc<Inner>,
}

struct Inner {
    port: u16,
    chat_queue: Mutex<VecDeque<ScriptedResponse>>,
    chat_hits: Mutex<u64>,
}

impl MockServer {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().unwrap().port();
        let inner = Arc::new(Inner {
            port,
            chat_queue: Mutex::new(VecDeque::new()),
            chat_hits: Mutex::new(0),
        });
        let inner_clone = inner.clone();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let inner = inner_clone.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_conn(sock, inner).await {
                        eprintln!("[mock] conn error: {e}");
                    }
                });
            }
        });
        Self { inner }
    }

    pub fn port(&self) -> u16 {
        self.inner.port
    }

    pub async fn push_chat(&self, resp: ScriptedResponse) {
        self.inner.chat_queue.lock().await.push_back(resp);
    }

    pub async fn chat_hits(&self) -> u64 {
        *self.inner.chat_hits.lock().await
    }
}

async fn handle_conn(
    mut sock: TcpStream,
    inner: Arc<Inner>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (read_half, mut write_half) = sock.split();
    let mut reader = BufReader::new(read_half);

    // Parse request line.
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }
    let parts: Vec<&str> = line.trim_end().split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Read headers; capture content-length.
    let mut content_length: usize = 0;
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h).await?;
        if n == 0 {
            break;
        }
        let trimmed = h.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    // Drain body.
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).await?;
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            write_simple_json(&mut write_half, 200, br#"{"status":"ok"}"#).await?;
        }
        ("GET", "/props") => {
            // llama.cpp-shaped: include chat_template_caps to advertise tool
            // support; preflight will log "tool-call support" on the agent side.
            let body = br#"{
                "model_path": "/tmp/mock.gguf",
                "build_info": "mock-build",
                "total_slots": 4,
                "chat_template_caps": { "supports_tools": true }
            }"#;
            write_simple_json(&mut write_half, 200, body).await?;
        }
        ("GET", "/v1/models") => {
            let body = br#"{"object":"list","data":[{"id":"mock-model","object":"model","created":0,"owned_by":"mock"}]}"#;
            write_simple_json(&mut write_half, 200, body).await?;
        }
        ("POST", "/v1/chat/completions") => {
            *inner.chat_hits.lock().await += 1;
            let next = inner.chat_queue.lock().await.pop_front();
            match next {
                None => {
                    write_simple_json(
                        &mut write_half,
                        500,
                        br#"{"error":{"message":"mock: no scripted response queued"}}"#,
                    )
                    .await?;
                }
                Some(resp) => {
                    write_sse_response(&mut write_half, &resp).await?;
                }
            }
        }
        _ => {
            write_simple_json(&mut write_half, 404, br#"{"error":"not found"}"#).await?;
        }
    }
    write_half.flush().await?;
    write_half.shutdown().await.ok();
    Ok(())
}

async fn write_simple_json(
    w: &mut (impl AsyncWriteExt + Unpin),
    status: u16,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );
    w.write_all(head.as_bytes()).await?;
    w.write_all(body).await?;
    Ok(())
}

async fn write_sse_response(
    w: &mut (impl AsyncWriteExt + Unpin),
    script: &ScriptedResponse,
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\n\
                Connection: close\r\n\
                \r\n";
    w.write_all(head.as_bytes()).await?;

    // Stream each text delta as a chat.completion.chunk.
    for delta in &script.text_deltas {
        let chunk = serde_json::json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": delta },
                "finish_reason": null
            }]
        });
        let line = format!("data: {}\n\n", chunk);
        w.write_all(line.as_bytes()).await?;
    }
    // Final chunk with finish_reason: stop.
    let final_chunk = serde_json::json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    });
    w.write_all(format!("data: {}\n\n", final_chunk).as_bytes())
        .await?;
    w.write_all(b"data: [DONE]\n\n").await?;
    Ok(())
}

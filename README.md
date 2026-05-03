# do-something

A minimal [ACP](https://agentclientprotocol.com)-compatible coding agent in
Rust that talks to any OpenAI-compatible chat-completions endpoint —
primarily targeting `llama-server` from llama.cpp, but also working against
OpenAI, vLLM, TGI, and similar backends.

The agent runs as a JSON-RPC subprocess over stdin/stdout and is designed to
be launched by an ACP client (e.g. Zed). All filesystem and shell I/O is
delegated back to the client through the protocol — the agent itself does
not touch the user's filesystem directly.

## Features

- **ACP surface:** `initialize`, `session/new`, `session/prompt`,
  `session/cancel`, `session/set_mode`
- **Streaming:** SSE deltas from the upstream model are forwarded as
  `agent_message_chunk` notifications; cancellation aborts the in-flight
  HTTP stream and returns `StopReason::Cancelled`
- **Tool calling** (delegated to the client via ACP):
  - `read_file` (with optional `line` / `limit`)
  - `write_file`
  - `list_files` (via terminal)
  - `run_shell` (via terminal, with cancellation → kill + release)
  - Tool-call loop is capped at 16 iterations per turn
    (`StopReason::MaxTurnRequests`)
- **Permission gating:** per-profile `permission_mode = "client" | "agent"`.
  In `"agent"` mode, `write_file` and `run_shell` issue
  `session/request_permission` and only proceed on explicit allow.
- **Session modes = profiles:** every configured profile is exposed as an
  ACP session mode; clients can switch the active profile mid-session via
  `session/set_mode` and the agent emits `current_mode_update`.
- **Slash commands** (advertised via `available_commands_update`):
  - `/help` — built-in help text
  - `/profiles` — list configured profiles, marking the default
  - Unknown `/foo` falls through to the model.
- **Startup preflight** (best-effort, never fatal) against `llama-server`:
  probes `GET /health`, `GET /props`, `GET /v1/models`; warns when
  `--jinja` is missing, when tool-template support is absent, or when
  `total_slots <= 1`. All preflight output goes to stderr logs.
- **Vision passthrough:** profiles with `supports_vision = true` forward
  `ContentBlock::Image` as OpenAI multimodal `image_url` data URIs;
  text-only models receive a `[image: <mime>]` placeholder.
- **Embedded context:** `ResourceLink` and text `Resource` content blocks
  are inlined into the user message (`embedded_context` capability is
  advertised at initialize).
- **Context-overflow detection:** both HTTP-status and mid-stream SSE
  error envelopes from llama.cpp and OpenAI are mapped to
  `StopReason::MaxTokens`.
- **Free-form sampler params:** any TOML keys under `extra_body` are
  merged into the chat-completions request (e.g. `top_k`, `min_p`,
  `repeat_penalty`, `cache_prompt`, `grammar`, `json_schema`).

## Build

```sh
cargo build --release
```

The binary is produced at `target/release/do-something`.

## Configure

Default config path resolution order:

1. `$DO_SOMETHING_CONFIG`
2. `$XDG_CONFIG_HOME/do-something/config.toml`
3. `$HOME/.config/do-something/config.toml`
4. `./do-something/config.toml`

If no config file is found, an in-memory default targeting
`http://127.0.0.1:8080/v1` is used.

A worked example lives at [`examples/config.toml`](examples/config.toml):

```toml
default_profile = "local-llama"

[profiles.local-llama]
base_url = "http://127.0.0.1:8080/v1"
model = "qwen2.5-coder-7b"        # cosmetic for llama-server
# api_key_env = "LLAMA_API_KEY"   # optional; Authorization header omitted if unset
temperature = 0.2
max_tokens = 2048
# system_prompt = "You are a concise coding assistant."
# permission_mode = "client"      # or "agent" to require explicit allow for write_file / run_shell
# supports_vision = false         # set true to forward image blocks as OpenAI image_url

[profiles.local-llama.extra_body]
cache_prompt = true
top_k = 40
min_p = 0.05
repeat_penalty = 1.05

[profiles.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
supports_vision = true
```

### Profile fields

| Field             | Type      | Notes |
|-------------------|-----------|-------|
| `base_url`        | string    | Required. OpenAI-compatible v1 root, e.g. `http://127.0.0.1:8080/v1`. |
| `model`           | string    | Required. Cosmetic for llama-server, real for OpenAI/vLLM. |
| `api_key_env`     | string    | Optional. Env var to read the API key from. Header omitted if unset/empty. |
| `temperature`     | float     | Optional. |
| `max_tokens`      | u32       | Optional. |
| `system_prompt`   | string    | Optional. Prepended as a `system` message. |
| `permission_mode` | string    | `"client"` (default) or `"agent"`. |
| `supports_vision` | bool      | Default `false`. |
| `extra_body`      | table     | Free-form params merged into the chat request body. |

### Environment overrides

- `DO_SOMETHING_CONFIG` — config file path
- `DO_SOMETHING_PROFILE` — switch active profile by name (overrides `default_profile`)
- `DO_SOMETHING_BASE_URL`, `DO_SOMETHING_MODEL`, `DO_SOMETHING_API_KEY` —
  override fields of the active profile (`DO_SOMETHING_API_KEY` takes
  precedence over `api_key_env`)
- `DO_SOMETHING_LOG` — `tracing` env filter (default `info`); logs go to stderr

## Run with llama.cpp

Launch a tool-capable model with the chat template enabled:

```sh
llama-server \
  -m qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  -c 8192 --jinja --port 8080
```

`--jinja` is required for tool calling; the agent will warn at startup if
it is missing.

Then connect any ACP client (e.g. Zed) by pointing it at the agent binary:

```
target/release/do-something
```

The agent communicates over stdin/stdout in JSON-RPC; logs are written to
stderr. **Never** print to stdout from this process — doing so corrupts the
ACP channel.

## Testing

End-to-end integration tests spawn the real binary against an in-process
mock OpenAI/llama.cpp server, so no live model is required:

```sh
cargo test
```

Coverage includes:

- streaming text prompt round-trip (initialize → session/new → session/prompt)
- `/help` slash command short-circuiting (asserts zero upstream chat hits)
- `session/set_mode` switching the active profile and emitting
  `current_mode_update`

For verbose output:

```sh
DO_SOMETHING_LOG=debug cargo test -- --nocapture
```

The `llm` module also has unit tests for context-overflow detection and
URL-suffix handling.

## Layout

```
src/
  main.rs    # tokio + ACP stdio bootstrap, startup preflight
  config.rs  # TOML loader + env overrides, profile/permission types
  llm.rs     # OpenAI-compatible streaming client, SSE parser, preflight probes
  tools.rs   # ACP-delegated read_file / write_file / list_files / run_shell
  agent.rs   # initialize / session/new / session/prompt / session/cancel
             # session/set_mode, slash commands, tool-call loop, permissions
tests/
  integration.rs       # spawns the binary against a mock server
  common/mock_server.rs # /health, /props, /v1/models, /v1/chat/completions
examples/
  config.toml
```

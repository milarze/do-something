# do-something

A minimal ACP-compatible agent in Rust that talks to any OpenAI-compatible
chat-completions endpoint (primarily targeting `llama-server` from llama.cpp).

## Status

Milestone 1 + 2 of the planned roadmap:

- ACP `initialize`, `session/new`, `session/prompt`, `session/cancel` over stdio
- TOML config + env overrides; multiple named profiles
- Streaming chat completions (SSE) forwarded as `agent_message_chunk` notifications
- Cancellation aborts the in-flight HTTP stream and returns `StopReason::Cancelled`

Tools (`fs`, `terminal`, `run_shell`), session modes for profile switching,
slash commands, and preflight (`/health`, `/props`) are deferred to later
milestones.

## Build

```sh
cargo build --release
```

## Configure

Default config path: `$XDG_CONFIG_HOME/do-something/config.toml`
(or `$HOME/.config/do-something/config.toml`).

Override with `DO_SOMETHING_CONFIG=/path/to/config.toml`.

```toml
default_profile = "local-llama"

[profiles.local-llama]
base_url = "http://127.0.0.1:8080/v1"
model = "qwen2.5-coder-7b"        # cosmetic for llama-server
# api_key_env = "LLAMA_API_KEY"   # optional; header omitted if unset/empty
temperature = 0.2
max_tokens = 2048
# system_prompt = "You are a concise coding assistant."

[profiles.local-llama.extra_body]
cache_prompt = true
top_k = 40
min_p = 0.05
repeat_penalty = 1.05

[profiles.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

### Environment overrides

- `DO_SOMETHING_PROFILE` — switch active profile by name
- `DO_SOMETHING_BASE_URL`, `DO_SOMETHING_MODEL`, `DO_SOMETHING_API_KEY` —
  override fields of the active profile
- `DO_SOMETHING_LOG` — `tracing` env filter (default `info`); logs go to stderr

## Run with llama.cpp

Launch a tool-capable model with the chat template enabled:

```sh
llama-server \
  -m qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  -c 8192 --jinja --port 8080
```

Then connect any ACP client (e.g. Zed) by pointing it at the agent binary:

```
target/release/do-something
```

The agent communicates over stdin/stdout in JSON-RPC; logs are written to
stderr. **Never** print to stdout from this process.

## Layout

```
src/
  main.rs    # tokio + ACP stdio bootstrap
  config.rs  # TOML + env overrides
  llm.rs     # OpenAI-compatible streaming client
  agent.rs   # initialize / session/new / session/prompt / session/cancel
```

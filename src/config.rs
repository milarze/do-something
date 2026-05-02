//! Configuration loading: TOML file at $XDG_CONFIG_HOME/do-something/config.toml
//! plus environment variable overrides.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: BTreeMap<String, Profile>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Profile {
    /// e.g. "http://127.0.0.1:8080/v1"
    pub base_url: String,
    /// model name to send in the chat request (cosmetic for llama.cpp)
    pub model: String,
    /// Env var name for API key. Bearer header is omitted if unset/empty.
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Free-form params merged into the request JSON body (e.g. top_k, min_p,
    /// repeat_penalty, cache_prompt, grammar, json_schema).
    #[serde(default)]
    pub extra_body: toml::Table,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Who decides whether a sensitive tool call (write_file, run_shell) is
    /// allowed. Defaults to `Client`: the agent invokes the FS/terminal call
    /// directly and trusts the client to enforce its own policy (prompt the
    /// user, auto-approve, or reject). In `Agent` mode the agent calls
    /// `session/request_permission` first and only proceeds on allow.
    #[serde(default)]
    pub permission_mode: PermissionMode,
    /// Whether the underlying model accepts image inputs (OpenAI multimodal
    /// `image_url` content parts). When `false` (default), incoming
    /// `ContentBlock::Image` blocks are summarised as `[image: <mime>]`
    /// placeholders in the user message. When `true`, images are forwarded
    /// to the model as `data:` URLs.
    #[serde(default)]
    pub supports_vision: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    /// Trust the client to enforce permission policy. Agent skips
    /// `session/request_permission` entirely and just makes the tool call.
    #[default]
    Client,
    /// Agent calls `session/request_permission` for sensitive tools and
    /// only proceeds on an explicit allow outcome.
    Agent,
}

impl Config {
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("DO_SOMETHING_CONFIG") {
            return PathBuf::from(p);
        }
        let base = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".config"))
                    .unwrap_or_else(|_| PathBuf::from("."))
            });
        base.join("do-something").join("config.toml")
    }

    pub fn load() -> Result<Self> {
        let path = Self::default_path();
        let mut cfg = if path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading config file {}", path.display()))?;
            toml::from_str::<Config>(&text)
                .with_context(|| format!("parsing config file {}", path.display()))?
        } else {
            tracing::warn!(
                "config file not found at {}; using built-in defaults",
                path.display()
            );
            Config::builtin_default()
        };

        // Environment overrides apply to the active profile (after creating it
        // if necessary).
        if let Ok(p) = std::env::var("DO_SOMETHING_PROFILE") {
            cfg.default_profile = Some(p);
        }

        let active_name = cfg
            .default_profile
            .clone()
            .or_else(|| cfg.profiles.keys().next().cloned())
            .unwrap_or_else(|| "default".to_string());

        let entry = cfg.profiles.entry(active_name.clone()).or_insert(Profile {
            base_url: "http://127.0.0.1:8080/v1".to_string(),
            model: "local".to_string(),
            api_key_env: None,
            temperature: None,
            max_tokens: None,
            extra_body: toml::Table::new(),
            system_prompt: None,
            permission_mode: PermissionMode::default(),
            supports_vision: false,
        });
        if let Ok(v) = std::env::var("DO_SOMETHING_BASE_URL") {
            entry.base_url = v;
        }
        if let Ok(v) = std::env::var("DO_SOMETHING_MODEL") {
            entry.model = v;
        }

        cfg.default_profile = Some(active_name);

        if cfg.profiles.is_empty() {
            anyhow::bail!("no profiles configured");
        }
        Ok(cfg)
    }

    pub fn builtin_default() -> Self {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "local-llama".to_string(),
            Profile {
                base_url: "http://127.0.0.1:8080/v1".to_string(),
                model: "local".to_string(),
                api_key_env: None,
                temperature: Some(0.2),
                max_tokens: Some(2048),
                extra_body: toml::Table::new(),
                system_prompt: None,
                permission_mode: PermissionMode::default(),
                supports_vision: false,
            },
        );
        Config {
            default_profile: Some("local-llama".to_string()),
            profiles,
        }
    }

    pub fn active_profile(&self) -> (&str, &Profile) {
        let name = self
            .default_profile
            .as_deref()
            .unwrap_or_else(|| self.profiles.keys().next().map(|s| s.as_str()).unwrap_or(""));
        let profile = self
            .profiles
            .get(name)
            .expect("active profile must exist after load()");
        (name, profile)
    }

    pub fn profile(&self, name: &str) -> Option<&Profile> {
        self.profiles.get(name)
    }
}

impl Profile {
    /// Resolve API key from env. Returns None if env var is unset OR empty.
    pub fn api_key(&self) -> Option<String> {
        let var = self.api_key_env.as_deref()?;
        // Allow override via DO_SOMETHING_API_KEY too.
        if let Ok(v) = std::env::var("DO_SOMETHING_API_KEY") {
            if !v.is_empty() {
                return Some(v);
            }
        }
        match std::env::var(var) {
            Ok(v) if !v.is_empty() => Some(v),
            _ => None,
        }
    }
}

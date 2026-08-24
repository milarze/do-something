//! Knowledge context assembly for LLM prompt injection.
//!
//! Assembles relevant knowledge (site configs, patterns, user preferences)
//! into a format suitable for injection into LLM system prompts,
//! respecting token budgets.

use crate::models::agent_state::KnowledgeContext;
use crate::models::knowledge::{Patterns, SiteConfig, UserModel};

use super::store::KnowledgeStore;

/// Default token budget for knowledge context.
///
/// This reserves tokens for learned knowledge in the system prompt,
/// leaving room for conversation history and response generation.
///
/// Typical allocation for a 128k context window:
/// - System prompt: ~4k tokens
/// - Knowledge context: ~8k tokens (this constant)
/// - Conversation history: ~100k tokens
/// - Response generation: ~16k tokens
///
/// Adjust based on:
/// - Model context window size
/// - Average knowledge size for your use case
/// - Required history depth
pub const DEFAULT_TOKEN_BUDGET: u64 = 8000;

/// Minimum tokens to allocate for knowledge context.
///
/// Below this threshold, knowledge becomes too sparse to be useful.
/// Ensures at least some site config or patterns are included.
///
/// With 500 tokens, we can typically include:
/// - 1 site config (~200 tokens), OR
/// - 2-3 patterns (~150 tokens), OR
/// - A minimal user model (~100 tokens)
pub const MIN_TOKEN_BUDGET: u64 = 500;

/// Approximate characters per token for estimation.
///
/// This is a rough approximation for token counting without calling
/// the model's tokenizer. Actual values vary by:
///
/// - Content type: English text averages ~4 chars/token
/// - Code: often fewer chars per token (~3-3.5)
/// - Whitespace-heavy content: more chars per token
/// - Model-specific: GPT-4, Claude, Llama all differ slightly
///
/// We use 4 as a conservative middle-ground for recipe-related text
/// which tends to be natural language with some structure.
pub const CHARS_PER_TOKEN: usize = 4;

/// Assembler for knowledge context injection.
pub struct KnowledgeContextAssembler;

impl KnowledgeContextAssembler {
    /// Build knowledge context for LLM injection.
    ///
    /// Assembles site config, patterns, and user preferences
    /// within the specified token budget.
    pub fn build(
        store: &KnowledgeStore,
        domain: Option<&str>,
        token_budget: u64,
    ) -> crate::storage::Result<KnowledgeContext> {
        let token_budget = token_budget.clamp(MIN_TOKEN_BUDGET, DEFAULT_TOKEN_BUDGET);
        let mut ctx = KnowledgeContext {
            token_budget,
            ..Default::default()
        };

        // Load site config if domain specified
        if let Some(d) = domain {
            let config = store.get_site_config(d)?;
            if let Some(json) = self::try_add_to_budget(&mut ctx, &config, token_budget) {
                ctx.site_configs.push(json);
            }
        }

        // Load patterns
        let patterns = store.get_patterns()?;
        let filtered_patterns = self::filter_patterns_for_domain(&patterns, domain);
        if (!filtered_patterns.success_patterns.is_empty()
            || !filtered_patterns.anti_patterns.is_empty())
            && let Some(json) = self::try_add_to_budget(&mut ctx, &filtered_patterns, token_budget)
        {
            ctx.patterns.push(json);
        }

        // Load user model
        let user_model = store.get_user_model("default")?;
        if user_model.sample_size > 0
            && let Some(json) = self::try_add_to_budget(&mut ctx, &user_model, token_budget)
        {
            ctx.user_model = Some(json);
        }

        // Calculate estimated tokens
        ctx.estimated_tokens = Self::estimate_tokens(&ctx);

        Ok(ctx)
    }

    /// Serialize context to text for prompt injection.
    ///
    /// Creates a human-readable format suitable for system prompts.
    pub fn to_prompt_text(ctx: &KnowledgeContext) -> String {
        let mut text = String::new();

        text.push_str("# Learned Knowledge\n\n");

        // Site configurations
        if !ctx.site_configs.is_empty() {
            text.push_str("## Site Configurations\n\n");
            for (i, config_json) in ctx.site_configs.iter().enumerate() {
                if let Ok(config) = serde_json::from_str::<SiteConfig>(config_json) {
                    text.push_str(&format!("### Site: {}\n", config.domain));
                    text.push_str(&format!("- Preferred method: {:?}\n", config.preferred_method));
                    text.push_str(&format!("- Rate limit: {}ms\n", config.rate_limit_ms));
                    if !config.skip_patterns.is_empty() {
                        text.push_str(&format!("- Skip patterns: {}\n", config.skip_patterns.join(", ")));
                    }
                    if i < ctx.site_configs.len() - 1 {
                        text.push('\n');
                    }
                }
            }
            text.push('\n');
        }

        // Patterns
        if !ctx.patterns.is_empty() {
            text.push_str("## Known Patterns\n\n");
            for pattern_json in &ctx.patterns {
                if let Ok(patterns) = serde_json::from_str::<Patterns>(pattern_json) {
                    if !patterns.success_patterns.is_empty() {
                        text.push_str("### Success Patterns\n");
                        for p in &patterns.success_patterns {
                            text.push_str(&format!("- {} (confidence: {:.0}%)\n", 
                                p.description, p.confidence * 100.0));
                        }
                        text.push('\n');
                    }
                    if !patterns.anti_patterns.is_empty() {
                        text.push_str("### Anti-Patterns (avoid)\n");
                        for p in &patterns.anti_patterns {
                            text.push_str(&format!("- {} (action: {:?})\n",
                                p.description, p.action));
                        }
                        text.push('\n');
                    }
                }
            }
        }

        // User preferences
        if let Some(ref model_json) = ctx.user_model
            && let Ok(model) = serde_json::from_str::<UserModel>(model_json)
        {
            text.push_str("## User Preferences\n\n");
            if let Some(max_prep) = model.max_prep_time_minutes {
                text.push_str(&format!("- Max prep time: {} minutes\n", max_prep));
            }
            if let Some(max_total) = model.max_total_time_minutes {
                text.push_str(&format!("- Max total time: {} minutes\n", max_total));
            }
            if let Some(ref diff) = model.preferred_difficulty {
                text.push_str(&format!("- Preferred difficulty: {:?}\n", diff));
            }
            if !model.dietary_restrictions.is_empty() {
                text.push_str(&format!("- Tags: {}\n", model.dietary_restrictions.join(", ")));
            }
            if model.sample_size > 0 {
                text.push_str(&format!("- (based on {} recipes)\n", model.sample_size));
            }
        }

        text
    }

    /// Estimate token count for context.
    fn estimate_tokens(ctx: &KnowledgeContext) -> u64 {
        let total_chars: usize = ctx.site_configs.iter().map(|s| s.len()).sum::<usize>()
            + ctx.patterns.iter().map(|s| s.len()).sum::<usize>()
            + ctx.user_model.as_ref().map(|s| s.len()).unwrap_or(0);

        (total_chars / CHARS_PER_TOKEN) as u64
    }
}

/// Try to add a value to context if within budget.
///
/// Returns Some(json) if successful, None if it would exceed budget.
fn try_add_to_budget<T: serde::Serialize>(
    ctx: &mut KnowledgeContext,
    value: &T,
    budget: u64,
) -> Option<String> {
    let json = serde_json::to_string(value).ok()?;
    let json_tokens = (json.len() / CHARS_PER_TOKEN) as u64;

    let current_tokens = KnowledgeContextAssembler::estimate_tokens(ctx);

    if current_tokens + json_tokens <= budget {
        Some(json)
    } else {
        None
    }
}

/// Filter patterns to only include those relevant to a domain.
fn filter_patterns_for_domain(patterns: &Patterns, domain: Option<&str>) -> Patterns {
    // Include universal patterns (empty sites list)
    // and patterns that apply to the specified domain
    let success_patterns = patterns
        .success_patterns
        .iter()
        .filter(|p| domain.is_none_or(|d| p.applies_to(d)))
        .cloned()
        .collect();

    let anti_patterns = patterns
        .anti_patterns
        .iter()
        .filter(|p| {
            domain.is_none_or(|d| p.sites.is_empty() || p.sites.iter().any(|s| s == d))
        })
        .cloned()
        .collect();

    Patterns {
        success_patterns,
        anti_patterns,
        version: patterns.version,
        computed_at: patterns.computed_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ParseMethod;
    use crate::knowledge::patterns::PatternMatcher;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn test_store() -> (KnowledgeStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let storage = Arc::new(crate::storage::FileKnowledgeStore::open(
            dir.path().join("knowledge"),
        ).unwrap());
        (KnowledgeStore::new(storage), dir)
    }

    #[test]
    fn build_respects_token_budget() {
        let (store, _dir) = test_store();

        let ctx = KnowledgeContextAssembler::build(&store, None, 500).unwrap();
        assert!(ctx.estimated_tokens <= ctx.token_budget);
    }

    #[test]
    fn build_includes_site_config_for_domain() {
        let (store, _dir) = test_store();

        // Save a config
        let mut config = crate::models::knowledge::SiteConfig::new("example.com");
        config.preferred_method = ParseMethod::Selectors;
        store.update_site_config(&config).unwrap();

        let ctx = KnowledgeContextAssembler::build(&store, Some("example.com"), DEFAULT_TOKEN_BUDGET).unwrap();
        
        assert!(!ctx.site_configs.is_empty());
    }

    #[test]
    fn build_includes_user_model() {
        let (store, _dir) = test_store();

        // Create user model with some data
        let mut model = crate::models::knowledge::UserModel::default_user();
        model.sample_size = 5;
        model.max_prep_time_minutes = Some(30);
        store.update_user_model(&model).unwrap();

        let ctx = KnowledgeContextAssembler::build(&store, None, DEFAULT_TOKEN_BUDGET).unwrap();
        
        assert!(ctx.user_model.is_some());
    }

    #[test]
    fn build_includes_patterns() {
        let (store, _dir) = test_store();

        // Use default patterns
        let patterns = PatternMatcher::default_patterns();
        store.update_patterns(&patterns).unwrap();

        let ctx = KnowledgeContextAssembler::build(&store, None, DEFAULT_TOKEN_BUDGET).unwrap();
        
        assert!(!ctx.patterns.is_empty());
    }

    #[test]
    fn to_prompt_text_formats_site_config() {
        let mut ctx = KnowledgeContext::default();
        
        let config = crate::models::knowledge::SiteConfig::new("test.com");
        ctx.site_configs.push(serde_json::to_string(&config).unwrap());

        let text = KnowledgeContextAssembler::to_prompt_text(&ctx);
        
        assert!(text.contains("test.com"));
        assert!(text.contains("Site Configurations"));
    }

    #[test]
    fn to_prompt_text_formats_user_model() {
        let mut ctx = KnowledgeContext::default();
        
        let model = crate::models::knowledge::UserModel {
            user_id: "test".to_string(),
            max_prep_time_minutes: Some(30),
            sample_size: 10,
            ..Default::default()
        };
        ctx.user_model = Some(serde_json::to_string(&model).unwrap());

        let text = KnowledgeContextAssembler::to_prompt_text(&ctx);
        
        assert!(text.contains("User Preferences"));
        assert!(text.contains("Max prep time"));
    }

    #[test]
    fn to_prompt_text_formats_patterns() {
        let mut ctx = KnowledgeContext::default();
        
        let mut patterns = Patterns::default();
        patterns.success_patterns.push(crate::models::knowledge::SuccessPattern {
            description: "Test pattern".to_string(),
            sites: vec![],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.85,
        });
        ctx.patterns.push(serde_json::to_string(&patterns).unwrap());

        let text = KnowledgeContextAssembler::to_prompt_text(&ctx);
        
        assert!(text.contains("Test pattern"));
        assert!(text.contains("Success Patterns"));
    }

    #[test]
    fn to_prompt_text_empty_context() {
        let ctx = KnowledgeContext::default();
        let text = KnowledgeContextAssembler::to_prompt_text(&ctx);
        
        assert!(text.contains("Learned Knowledge"));
    }

    #[test]
    fn estimate_tokens_reasonable() {
        let mut ctx = KnowledgeContext::default();
        
        let config = crate::models::knowledge::SiteConfig::new("example.com");
        ctx.site_configs.push(serde_json::to_string(&config).unwrap());

        let tokens = KnowledgeContextAssembler::estimate_tokens(&ctx);
        
        // Should be roughly the JSON length / 4
        assert!(tokens > 0);
        assert!(tokens < 1000); // Single config shouldn't be huge
    }

    #[test]
    fn filter_patterns_keeps_universal() {
        let patterns = PatternMatcher::default_patterns();
        let filtered = filter_patterns_for_domain(&patterns, Some("unknown-site.com"));

        // Should include patterns with empty sites list
        assert!(!filtered.success_patterns.is_empty());
    }

    #[test]
    fn filter_patterns_keeps_domain_specific() {
        let mut patterns = Patterns::default();
        patterns.success_patterns.push(crate::models::knowledge::SuccessPattern {
            description: "Site-specific".to_string(),
            sites: vec!["example.com".to_string()],
            success_rate: 0.9,
            sample_size: 50,
            confidence: 0.8,
        });

        let filtered = filter_patterns_for_domain(&patterns, Some("example.com"));
        assert_eq!(filtered.success_patterns.len(), 1);

        let filtered_other = filter_patterns_for_domain(&patterns, Some("other.com"));
        assert_eq!(filtered_other.success_patterns.len(), 0);
    }

    #[test]
    fn build_with_no_domain_includes_universal_patterns() {
        let (store, _dir) = test_store();

        let patterns = PatternMatcher::default_patterns();
        store.update_patterns(&patterns).unwrap();

        let ctx = KnowledgeContextAssembler::build(&store, None, DEFAULT_TOKEN_BUDGET).unwrap();
        
        // Should include universal patterns
        assert!(!ctx.patterns.is_empty());
    }
}

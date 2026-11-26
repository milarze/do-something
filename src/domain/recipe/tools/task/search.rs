//! Search for recipe URLs matching a query.
//!
//! [`SearchTool`] queries a search backend and returns a filtered list of
//! recipe URLs. The backend is an injectable [`RecipeSearchBackend`] trait so
//! tests can supply canned results without network access.
//!
//! The default backend scrapes DuckDuckGo HTML results and filters to a set
//! of known recipe domains.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::framework::tools::traits::{Tool, ToolResult};

/// Known recipe domains used to filter generic search results.
const KNOWN_RECIPE_DOMAINS: &[&str] = &[
    "allrecipes.com",
    "foodnetwork.com",
    "tasty.co",
    "seriouseats.com",
    "epicurious.com",
    "bonappetit.com",
    "simplyrecipes.com",
    "thekitchn.com",
];

/// Injected search backend so the tool can be unit-tested without network.
#[async_trait]
pub trait RecipeSearchBackend: Send + Sync {
    /// Return raw candidate URLs for a query (unfiltered).
    async fn search_urls(&self, query: &str) -> std::result::Result<Vec<String>, String>;
}

/// DuckDuckGo HTML-scraping backend.
///
/// Requests `https://html.duckduckgo.com/html/?q=...` and extracts result
/// links. DuckDuckGo HTML scraping is a deliberately simple starting point.
pub struct DuckDuckGoBackend {
    http: reqwest::Client,
}

impl DuckDuckGoBackend {
    /// Create a backend with a default HTTP client.
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    /// Create a backend with a custom HTTP client (for tests/proxies).
    pub fn with_client(http: reqwest::Client) -> Self {
        Self { http }
    }
}

impl Default for DuckDuckGoBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RecipeSearchBackend for DuckDuckGoBackend {
    async fn search_urls(&self, query: &str) -> std::result::Result<Vec<String>, String> {
        let url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding::encode(query)
        );
        let resp = self
            .http
            .get(&url)
            .header(
                "User-Agent",
                "do-something/0.1 (+recipe scraping learning agent)",
            )
            .send()
            .await
            .map_err(|e| format!("search request failed: {e}"))?;

        let html = resp
            .text()
            .await
            .map_err(|e| format!("failed to read search body: {e}"))?;

        Ok(extract_result_links(&html))
    }
}

/// A canned backend for tests, returning a fixed list of URLs.
pub struct CannedBackend {
    urls: Vec<String>,
}

impl CannedBackend {
    pub fn new(urls: Vec<String>) -> Self {
        Self { urls }
    }
}

#[async_trait]
impl RecipeSearchBackend for CannedBackend {
    async fn search_urls(&self, _query: &str) -> std::result::Result<Vec<String>, String> {
        Ok(self.urls.clone())
    }
}

/// A backend that always errors, for testing error handling.
pub struct ErrorBackend;

#[async_trait]
impl RecipeSearchBackend for ErrorBackend {
    async fn search_urls(&self, _query: &str) -> std::result::Result<Vec<String>, String> {
        Err("backend unavailable".to_string())
    }
}

/// Extract result URLs from DuckDuckGo HTML.
///
/// DuckDuckGo wraps result links in `<a class="result__a" href="...">`. The
/// href is often a redirect (`//duckduckgo.com/l/?uddg=<encoded>`); this
/// function decodes the `uddg` parameter when present.
pub fn extract_result_links(html: &str) -> Vec<String> {
    let document = scraper::Html::parse_document(html);
    let Ok(sel) = scraper::Selector::parse("a.result__a") else {
        return Vec::new();
    };
    document
        .select(&sel)
        .filter_map(|el| el.value().attr("href").map(|h| h.to_string()))
        .map(|href| decode_ddg_redirect(&href).unwrap_or(href))
        .collect()
}

/// Decode a DuckDuckGo redirect URL into its target, if applicable.
fn decode_ddg_redirect(href: &str) -> Option<String> {
    // Protocol-relative URLs like //duckduckgo.com/l/?uddg=... need a base.
    let full_href = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    };
    let url = url::Url::parse(&full_href).ok()?;
    if url.host_str() == Some("duckduckgo.com") {
        url.query_pairs()
            .find(|(k, _)| k == "uddg")
            .map(|(_, v)| v.to_string())
    } else {
        None
    }
}

/// Filter a list of URLs to those whose host is a known recipe domain.
pub fn filter_recipe_urls(urls: &[String]) -> Vec<String> {
    urls.iter()
        .filter_map(|u| {
            let parsed = url::Url::parse(u).ok()?;
            let host = parsed.host_str()?;
            KNOWN_RECIPE_DOMAINS
                .iter()
                .any(|d| host == *d || host.ends_with(&format!(".{d}")))
                .then(|| u.clone())
        })
        .collect()
}

/// Search for recipe URLs matching a query.
pub struct SearchTool {
    backend: Box<dyn RecipeSearchBackend>,
}

impl SearchTool {
    /// Create a tool with a custom search backend.
    pub fn with_backend(backend: Box<dyn RecipeSearchBackend>) -> Self {
        Self { backend }
    }

    /// Create a tool with the default DuckDuckGo backend.
    pub fn new() -> Self {
        Self::with_backend(Box::new(DuckDuckGoBackend::new()))
    }
}

impl Default for SearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for recipe URLs matching a query. Returns list of URLs."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. \"chocolate cake\")."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return ToolResult::error("missing or invalid `query`");
        };
        if query.trim().is_empty() {
            return ToolResult::error("query must not be empty");
        }

        match self.backend.search_urls(query).await {
            Ok(urls) => {
                let filtered = filter_recipe_urls(&urls);
                ToolResult::success(json!({
                    "urls": filtered,
                    "count": filtered.len(),
                }))
            }
            Err(e) => ToolResult::error(format!("search failed: {e}")),
        }
    }
}

/// Minimal percent-encoding for search query terms.
///
/// We avoid pulling in an extra crate by encoding the characters that matter
/// for query strings. This is intentionally minimal.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char);
                }
                b' ' => out.push('+'),
                _ => out.push_str(&format!("%{:02X}", b)),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::tools::traits::Tool;
    use serde_json::json;

    #[test]
    fn filter_keeps_known_recipe_domains() {
        let urls = vec![
            "https://www.allrecipes.com/recipe/123".to_string(),
            "https://foodnetwork.com/food/x".to_string(),
        ];
        let filtered = filter_recipe_urls(&urls);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_drops_non_recipe_domains() {
        let urls = vec![
            "https://example.com/recipe".to_string(),
            "https://allrecipes.com/recipe/1".to_string(),
            "https://blog.example.com/post".to_string(),
        ];
        let filtered = filter_recipe_urls(&urls);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], "https://allrecipes.com/recipe/1");
    }

    #[test]
    fn filter_handles_subdomains() {
        let urls = vec![
            "https://www.seriouseats.com/recipe".to_string(),
            "https://recipes.bonappetit.com/x".to_string(),
        ];
        let filtered = filter_recipe_urls(&urls);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_ignores_invalid_urls() {
        let urls = vec![
            "not a url".to_string(),
            "https://allrecipes.com/x".to_string(),
        ];
        let filtered = filter_recipe_urls(&urls);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn decode_ddg_redirect_extracts_target() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fallrecipes.com%2Frecipe%2F1&rut=abc";
        let decoded = decode_ddg_redirect(href).unwrap();
        assert_eq!(decoded, "https://allrecipes.com/recipe/1");
    }

    #[test]
    fn decode_ddg_redirect_returns_none_for_direct_link() {
        assert!(decode_ddg_redirect("https://allrecipes.com/x").is_none());
    }

    #[test]
    fn extract_result_links_parses_ddg_html() {
        let html = r#"
        <html><body>
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fallrecipes.com%2Frecipe%2F1">Cake</a>
          <a class="result__a" href="https://foodnetwork.com/food/x">Food</a>
        </body></html>"#;
        let links = extract_result_links(html);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0], "https://allrecipes.com/recipe/1");
        assert_eq!(links[1], "https://foodnetwork.com/food/x");
    }

    #[test]
    fn extract_result_links_empty_when_no_results() {
        assert!(extract_result_links("<html></html>").is_empty());
    }

    #[tokio::test]
    async fn missing_query_errors() {
        let tool = SearchTool::with_backend(Box::new(CannedBackend::new(vec![])));
        let res = tool.execute(json!({})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("missing or invalid `query`"));
    }

    #[tokio::test]
    async fn empty_query_errors() {
        let tool = SearchTool::with_backend(Box::new(CannedBackend::new(vec![])));
        let res = tool.execute(json!({"query": "   "})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("query must not be empty"));
    }

    #[tokio::test]
    async fn returns_filtered_urls_from_backend() {
        let urls = vec![
            "https://allrecipes.com/recipe/1".to_string(),
            "https://example.com/not-recipe".to_string(),
            "https://foodnetwork.com/x".to_string(),
        ];
        let tool = SearchTool::with_backend(Box::new(CannedBackend::new(urls)));
        let res = tool.execute(json!({"query": "cake"})).await;
        let ToolResult::Success(v) = res else {
            panic!("expected success");
        };
        assert_eq!(v["count"], 2);
        assert_eq!(v["urls"][0], "https://allrecipes.com/recipe/1");
        assert_eq!(v["urls"][1], "https://foodnetwork.com/x");
    }

    #[tokio::test]
    async fn backend_error_surfaces_as_tool_error() {
        let tool = SearchTool::with_backend(Box::new(ErrorBackend));
        let res = tool.execute(json!({"query": "cake"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("search failed"));
    }

    #[test]
    fn urlencoding_encodes_spaces_and_special() {
        assert_eq!(urlencoding::encode("chocolate cake"), "chocolate+cake");
        assert_eq!(urlencoding::encode("a&b=c"), "a%26b%3Dc");
    }
}

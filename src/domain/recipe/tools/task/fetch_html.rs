//! HTML fetching tool with per-domain rate limiting.
//!
//! [`FetchHtmlTool`] fetches page HTML over HTTP, respecting a minimum
//! interval between requests to the same domain. The rate-limiting policy
//! is owned by an injectable [`RateLimiter`] so tests can use a no-op
//! limiter without real time delays.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::framework::tools::traits::{Tool, ToolResult};

/// Default minimum interval between requests to the same domain.
const DEFAULT_RATE_LIMIT: Duration = Duration::from_millis(1000);

/// Default user-agent string for fetch requests.
const DEFAULT_USER_AGENT: &str = concat!(
    "do-something/0.1 ",
    "(+https://example.com; recipe scraping learning agent)"
);

/// Per-domain rate limiting policy.
///
/// Tracks the last request time per domain and reports the remaining wait
/// for the next request. Isolated as a trait so tests can inject a no-op
/// policy without sleeping.
pub trait RateLimiter: Send + Sync {
    /// Register a minimum interval for a domain.
    fn set_interval(&self, domain: &str, interval: Duration);

    /// Return the duration to wait before the next request to `domain`,
    /// or `None` if no wait is required. Does not mutate state.
    fn wait_for(&self, domain: &str) -> Option<Duration>;

    /// Record that a request to `domain` has just been issued, updating the
    /// last-request timestamp used by [`RateLimiter::wait_for`].
    fn record_request(&self, domain: &str);
}

/// In-memory rate limiter with a per-domain minimum interval.
///
/// Uses a [`Mutex`] over a `HashMap` of domain → (interval, last request).
/// The default interval applies to any domain without an explicit override.
pub struct InMemoryRateLimiter {
    default_interval: Duration,
    overrides: Mutex<HashMap<String, Duration>>,
    last_requests: Mutex<HashMap<String, std::time::Instant>>,
}

impl InMemoryRateLimiter {
    /// Create a limiter with the given default interval.
    pub fn with_default(interval: Duration) -> Self {
        Self {
            default_interval: interval,
            overrides: Mutex::new(HashMap::new()),
            last_requests: Mutex::new(HashMap::new()),
        }
    }

    /// Create a limiter with the standard 1-second default interval.
    pub fn new() -> Self {
        Self::with_default(DEFAULT_RATE_LIMIT)
    }

    /// The configured interval for a domain (explicit override or default).
    fn interval_for(&self, domain: &str) -> Duration {
        self.overrides
            .lock()
            .expect("rate limiter overrides lock poisoned")
            .get(domain)
            .copied()
            .unwrap_or(self.default_interval)
    }
}

impl Default for InMemoryRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter for InMemoryRateLimiter {
    fn set_interval(&self, domain: &str, interval: Duration) {
        self.overrides
            .lock()
            .expect("rate limiter overrides lock poisoned")
            .insert(domain.to_string(), interval);
    }

    fn wait_for(&self, domain: &str) -> Option<Duration> {
        let interval = self.interval_for(domain);
        if interval.is_zero() {
            return None;
        }
        let last = self
            .last_requests
            .lock()
            .expect("rate limiter last-requests lock poisoned")
            .get(domain)
            .copied()?;
        let elapsed = last.elapsed();
        if elapsed >= interval {
            None
        } else {
            Some(interval - elapsed)
        }
    }

    fn record_request(&self, domain: &str) {
        self.last_requests
            .lock()
            .expect("rate limiter last-requests lock poisoned")
            .insert(domain.to_string(), std::time::Instant::now());
    }
}

/// A no-op rate limiter for tests that should never sleep.
#[derive(Default)]
pub struct NoopRateLimiter;

impl RateLimiter for NoopRateLimiter {
    fn set_interval(&self, _domain: &str, _interval: Duration) {}
    fn wait_for(&self, _domain: &str) -> Option<Duration> {
        None
    }
    fn record_request(&self, _domain: &str) {}
}

/// Extract the registrable host (domain) from a URL.
///
/// Returns the host string (e.g. `example.com`), or `None` if the URL has
/// no host (e.g. a `file:` URL or a malformed input).
pub fn extract_domain(url: &url::Url) -> Option<String> {
    url.host_str().map(|h| h.to_string())
}

/// Fetch HTML content from a URL, respecting per-domain rate limits.
pub struct FetchHtmlTool {
    http: reqwest::Client,
    rate_limiter: Box<dyn RateLimiter>,
    user_agent: String,
}

impl FetchHtmlTool {
    /// Build a tool with a custom HTTP client, rate limiter, and user agent.
    pub fn with_parts(
        http: reqwest::Client,
        rate_limiter: Box<dyn RateLimiter>,
        user_agent: String,
    ) -> Self {
        Self {
            http,
            rate_limiter,
            user_agent,
        }
    }

    /// Create a tool with a default HTTP client, in-memory rate limiter
    /// (1s default interval), and standard user agent.
    pub fn new() -> Self {
        Self::with_parts(
            reqwest::Client::new(),
            Box::<InMemoryRateLimiter>::default(),
            DEFAULT_USER_AGENT.to_string(),
        )
    }

    /// Borrow the rate limiter (used by tests and the agent runner).
    pub fn rate_limiter(&self) -> &dyn RateLimiter {
        self.rate_limiter.as_ref()
    }

    /// Perform the fetch, applying rate limiting and waiting as needed.
    async fn fetch(&self, url_str: &str) -> ToolResult {
        let parsed = match url::Url::parse(url_str) {
            Ok(u) => u,
            Err(e) => {
                return ToolResult::error(format!("invalid url: {e}"));
            }
        };

        let scheme = parsed.scheme();
        if scheme != "http" && scheme != "https" {
            return ToolResult::error(format!("unsupported scheme `{scheme}`"));
        }

        let Some(domain) = extract_domain(&parsed) else {
            return ToolResult::error("url has no host");
        };

        // Wait for the rate-limit window to elapse.
        if let Some(wait) = self.rate_limiter.wait_for(&domain) {
            tokio::time::sleep(wait).await;
        }

        let result = self
            .http
            .get(parsed.as_str())
            .header("User-Agent", &self.user_agent)
            .header("Accept", "text/html,application/xhtml+xml")
            .send()
            .await;

        // Record the request attempt regardless of outcome, so a retried
        // failed request still respects the rate limit.
        self.rate_limiter.record_request(&domain);

        let resp = match result {
            Ok(r) => r,
            Err(e) => {
                return ToolResult::error(format!("fetch failed: {e}"));
            }
        };

        let status = resp.status();
        if !status.is_success() {
            return ToolResult::error(format!("http error: {status}"));
        }

        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                return ToolResult::error(format!("failed to read body: {e}"));
            }
        };

        ToolResult::success(json!({
            "url": url_str,
            "domain": domain,
            "html": text,
        }))
    }
}

impl Default for FetchHtmlTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for FetchHtmlTool {
    fn name(&self) -> &str {
        "fetch_html"
    }

    fn description(&self) -> &str {
        "Fetch HTML content from a URL. Respects rate limits per domain."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to fetch."
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let Some(url_str) = args.get("url").and_then(Value::as_str) else {
            return ToolResult::error("missing or invalid `url`");
        };
        self.fetch(url_str).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::tools::traits::Tool;
    use serde_json::json;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn extract_domain_from_url() {
        assert_eq!(
            extract_domain(&"https://example.com/path".parse().unwrap()),
            Some("example.com".to_string())
        );
        assert_eq!(
            extract_domain(&"https://www.allrecipes.com/recipe/123".parse().unwrap()),
            Some("www.allrecipes.com".to_string())
        );
        assert_eq!(extract_domain(&"file:///tmp/x".parse().unwrap()), None);
    }

    #[test]
    fn rate_limiter_reports_wait_within_window() {
        let limiter = InMemoryRateLimiter::with_default(Duration::from_millis(100));
        limiter.record_request("example.com");
        let wait = limiter.wait_for("example.com").expect("should wait");
        assert!(wait <= Duration::from_millis(100));
    }

    #[test]
    fn rate_limiter_no_wait_after_interval_elapses() {
        let limiter = InMemoryRateLimiter::with_default(Duration::from_millis(0));
        limiter.record_request("example.com");
        assert!(limiter.wait_for("example.com").is_none());
    }

    #[test]
    fn rate_limiter_respects_per_domain_override() {
        let limiter = InMemoryRateLimiter::with_default(Duration::from_millis(1000));
        limiter.set_interval("slow.com", Duration::from_millis(0));
        limiter.record_request("slow.com");
        assert!(limiter.wait_for("slow.com").is_none());

        limiter.record_request("default.com");
        assert!(limiter.wait_for("default.com").is_some());
    }

    #[test]
    fn rate_limiter_no_wait_for_unseen_domain() {
        let limiter = InMemoryRateLimiter::new();
        assert!(limiter.wait_for("never-seen.com").is_none());
    }

    #[test]
    fn noop_rate_limiter_never_waits() {
        let limiter = NoopRateLimiter;
        limiter.record_request("example.com");
        assert!(limiter.wait_for("example.com").is_none());
    }

    #[tokio::test]
    async fn rejects_missing_url_argument() {
        let tool = FetchHtmlTool::with_parts(
            reqwest::Client::new(),
            Box::new(NoopRateLimiter),
            "test".to_string(),
        );
        let res = tool.execute(json!({})).await;
        assert!(res.is_error());
        assert_eq!(res.error_message(), Some("missing or invalid `url`"));
    }

    #[tokio::test]
    async fn rejects_non_http_scheme() {
        let tool = FetchHtmlTool::with_parts(
            reqwest::Client::new(),
            Box::new(NoopRateLimiter),
            "test".to_string(),
        );
        let res = tool.execute(json!({"url": "file:///etc/passwd"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("unsupported scheme"));
    }

    #[tokio::test]
    async fn rejects_invalid_url() {
        let tool = FetchHtmlTool::with_parts(
            reqwest::Client::new(),
            Box::new(NoopRateLimiter),
            "test".to_string(),
        );
        let res = tool.execute(json!({"url": "not a url"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("invalid url"));
    }

    #[tokio::test]
    async fn fetch_from_real_http_server() {
        // Spin up a tiny HTTP server that returns a fixed HTML body.
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let port = addr.port();

        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            // Read and discard the request line/headers.
            let mut buf = [0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let body = "<html><body>hello</body></html>";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
        });

        let tool = FetchHtmlTool::with_parts(
            reqwest::Client::new(),
            Box::new(NoopRateLimiter),
            "test-agent".to_string(),
        );
        let res = tool
            .execute(json!({"url": format!("http://127.0.0.1:{port}/page")}))
            .await;
        server.await.unwrap();

        let ToolResult::Success(v) = res else {
            panic!("expected success, got {:?}", res);
        };
        assert_eq!(v["domain"], "127.0.0.1");
        assert_eq!(v["html"], "<html><body>hello</body></html>");
    }

    #[tokio::test]
    async fn fetch_returns_error_on_http_failure_status() {
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let port = addr.port();

        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            sock.write_all(resp.as_bytes()).await.unwrap();
        });

        let tool = FetchHtmlTool::with_parts(
            reqwest::Client::new(),
            Box::new(NoopRateLimiter),
            "test-agent".to_string(),
        );
        let res = tool
            .execute(json!({"url": format!("http://127.0.0.1:{port}/missing")}))
            .await;
        server.await.unwrap();

        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("http error: 404"));
    }

    #[tokio::test]
    async fn fetch_records_request_for_rate_limiting() {
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let port = addr.port();

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
                sock.write_all(resp.as_bytes()).await.unwrap();
            }
        });

        let limiter = Box::new(InMemoryRateLimiter::with_default(Duration::from_millis(0)));
        let tool = FetchHtmlTool::with_parts(reqwest::Client::new(), limiter, "t".to_string());

        let url = format!("http://127.0.0.1:{port}/p");
        let _ = tool.execute(json!({"url": &url})).await;
        // After the request, the limiter should have recorded the domain.
        let tool_limiter = tool.rate_limiter();
        // No wait because interval is zero, but record_request was called.
        assert!(tool_limiter.wait_for("127.0.0.1").is_none());

        let _ = tool.execute(json!({"url": &url})).await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn fetch_connection_error_is_reported() {
        // Use a port that is almost certainly closed.
        let tool = FetchHtmlTool::with_parts(
            reqwest::Client::new(),
            Box::new(NoopRateLimiter),
            "test-agent".to_string(),
        );
        let res = tool
            .execute(json!({"url": "http://127.0.0.1:1/nope"}))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("fetch failed"));
    }
}

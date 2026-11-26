# Task Tools

> **Placement**: This layer spans the framework/domain boundary (see
> [`00-architecture.md`](./00-architecture.md)).
> - **Framework** (`framework/tools/`): the `Tool` trait, `ToolResult`, and
>   `ToolRegistry` are generic.
> - **Domain** (`domain/recipe/tools/task/`): `FetchHtmlTool`, `ParseRecipeTool`,
>   `SaveRecipeTool`, `SearchTool` are recipe operations.

## Code Style & Guidelines

1. **Test coverage**: Every public function must have test coverage
2. **Unit tests**: Prefer unit tests over integration tests. Use mocks/fakes for dependencies
3. **Dependency injection**: Accept traits as parameters, not concrete types
4. **Relevant tests**: Test behavior and edge cases, not just line coverage. Test each branch
5. **Linting**: Run `cargo clippy --all-targets --all-features -- -D warnings` before committing

## Overview

Task tools are the operations an agent invokes to do its work. The framework
defines the `Tool` trait and registry; the recipe domain provides the concrete
scraping tools: fetch, parse, save, and search.

## Framework Components

### `framework/tools/traits.rs`

The generic tool interface.

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value) -> impl Future<Output = ToolResult>;
}

pub enum ToolResult {
    Success(serde_json::Value),
    Error(String),
    PermissionRequired { request: PermissionRequest },
}
```

### `framework/tools/registry.rs`

The central tool registry.

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self;

    pub fn register<T: Tool + 'static>(&mut self, tool: T);

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;

    pub fn list(&self) -> Vec<&str>;

    /// Generate OpenAI-format tool definitions for the LLM
    pub fn tool_definitions(&self) -> Vec<ToolDef>;
}

pub struct ToolDef {
    pub kind: &'static str,
    pub function: ToolDefFunction,
}

pub struct ToolDefFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}
```

## Domain Components

### `domain/recipe/tools/task/mod.rs`

Module organization and exports.

```rust
pub mod fetch_html;
pub mod parse_recipe;
pub mod save_recipe;
pub mod search;

pub use fetch_html::FetchHtmlTool;
pub use parse_recipe::ParseRecipeTool;
pub use save_recipe::SaveRecipeTool;
pub use search::SearchTool;
```

### `domain/recipe/tools/task/fetch_html.rs`

HTTP fetching with rate limiting.

```rust
pub struct FetchHtmlTool {
    http: reqwest::Client,
    rate_limiter: RateLimiter,
    user_agent: String,
}

impl Tool for FetchHtmlTool {
    fn name(&self) -> &str { "fetch_html" }

    fn description(&self) -> &str {
        "Fetch HTML content from a URL. Respects rate limits per domain."
    }

    fn parameters_schema(&self) -> serde_json::Value { ... }

    async fn execute(&self, args: Value) -> ToolResult {
        let url: String = args["url"].as_str().ok_or(...)?;

        // Validate URL
        let parsed = url::Url::parse(&url)?;

        // Check rate limit for domain
        let domain = parsed.host_str().unwrap_or_default();
        self.rate_limiter.wait_for(domain).await?;

        // Fetch with proper headers
        let resp = self.http
            .get(&url)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "text/html,application/xhtml+xml")
            .send()
            .await?;

        // Return HTML content
        let html = resp.text().await?;
        Ok(ToolOutput::success(html))
    }
}
```

**Rate Limiting Strategy:**
- Per-domain minimum interval (configurable per site)
- Default: 1 second between requests to same domain
- Use `tokio::time::sleep` for waiting
- Track last request time per domain in memory

### `domain/recipe/tools/task/parse_recipe.rs`

Recipe parsing with multiple strategies.

```rust
pub struct ParseRecipeTool {
    knowledge: Arc<RecipeKnowledgeStore>,
}

impl Tool for ParseRecipeTool {
    fn name(&self) -> &str { "parse_recipe" }

    fn description(&self) -> &str {
        "Parse recipe data from HTML. Tries Schema.org first, then selectors."
    }

    fn parameters_schema(&self) -> serde_json::Value { ... }

    async fn execute(&self, args: Value) -> ToolResult {
        let html: String = args["html"].as_str().ok_or(...)?;
        let url: String = args["url"].as_str().ok_or(...)?;

        let domain = extract_domain(&url)?;
        let config = self.knowledge.get_site_config(&domain)?;

        // Try parsing strategies in order
        let recipe = self.try_parse(&html, &url, &config).await?;

        Ok(ToolOutput::success(recipe))
    }
}

impl ParseRecipeTool {
    /// Try parsing strategies in order of preference
    async fn try_parse(
        &self,
        html: &str,
        url: &str,
        config: &SiteConfig,
    ) -> Result<Recipe> {
        // 1. Schema.org JSON-LD (most reliable)
        if let Some(recipe) = self.parse_schema_org(html)? {
            return Ok(recipe);
        }

        // 2. Microdata
        if let Some(recipe) = self.parse_microdata(html)? {
            return Ok(recipe);
        }

        // 3. CSS selectors (fallback)
        if let Some(recipe) = self.parse_selectors(html, &config.selectors)? {
            return Ok(recipe);
        }

        // 4. Heuristics (last resort)
        self.parse_heuristics(html)
    }

    fn parse_schema_org(&self, html: &str) -> Result<Option<Recipe>> {
        // Look for <script type="application/ld+json">
        // Parse as Recipe schema
        // Extract fields
    }

    fn parse_selectors(
        &self,
        html: &str,
        selectors: &Selectors,
    ) -> Result<Option<Recipe>> {
        // Use CSS selectors from config
        // Extract each field
    }
}
```

**Parsing Priority:**
1. `Schema.org` JSON-LD — standardized, most reliable
2. `Microdata` — older format, still structured
3. `CSS Selectors` — site-specific, fallback
4. `Heuristics` — last resort, error-prone

### `domain/recipe/tools/task/save_recipe.rs`

Persist a recipe to the database.

```rust
pub struct SaveRecipeTool {
    db: Arc<RecipeDb>,
}

impl Tool for SaveRecipeTool {
    fn name(&self) -> &str { "save_recipe" }

    fn description(&self) -> &str {
        "Save a parsed recipe to the database. Returns recipe ID."
    }

    fn parameters_schema(&self) -> serde_json::Value { ... }

    async fn execute(&self, args: Value) -> ToolResult {
        let recipe: Recipe = serde_json::from_value(args)?;

        // Validate required fields
        if recipe.name.is_empty() {
            return Ok(ToolOutput::error("Recipe must have a name"));
        }

        // Check for duplicates
        if self.db.exists_by_url(&recipe.source_url.to_string())? {
            return Ok(ToolOutput::error("Recipe already exists"));
        }

        // Compute hash for deduplication
        let hash = recipe.compute_hash();
        let mut recipe = recipe;
        recipe.content_hash = Some(hash);

        // Save and return ID
        let id = self.db.insert(&recipe)?;

        Ok(ToolOutput::success(json!({
            "recipe_id": id.to_string(),
            "name": recipe.name
        })))
    }
}
```

### `domain/recipe/tools/task/search.rs`

Search for recipe URLs.

```rust
pub struct SearchTool {
    http: reqwest::Client,
}

impl Tool for SearchTool {
    fn name(&self) -> &str { "search" }

    fn description(&self) -> &str {
        "Search for recipe URLs matching a query. Returns list of URLs."
    }

    fn parameters_schema(&self) -> serde_json::Value { ... }

    async fn execute(&self, args: Value) -> ToolResult {
        let query: String = args["query"].as_str().ok_or(...)?;

        let urls = self.search_recipes(&query).await?;

        Ok(ToolOutput::success(json!({
            "urls": urls,
            "count": urls.len()
        })))
    }
}
```

**Search Strategy:**
- Start simple: DuckDuckGo HTML scraping or DDG instant answer API
- Filter results to known recipe domains
- Return URLs only, not content

## Tool Registry Integration

Domain tools register in the framework registry:

```rust
pub fn register_task_tools(
    registry: &mut ToolRegistry,
    knowledge: Arc<RecipeKnowledgeStore>,
    db: Arc<RecipeDb>,
) {
    registry.register(FetchHtmlTool::new());
    registry.register(ParseRecipeTool::new(knowledge));
    registry.register(SaveRecipeTool::new(db));
    registry.register(SearchTool::new());
}
```

## Tests

- `FetchHtmlTool` respects rate limits
- `FetchHtmlTool` handles HTTP errors gracefully
- `ParseRecipeTool` extracts recipes from test HTML
- `ParseRecipeTool` falls back correctly
- `SaveRecipeTool` rejects duplicates
- `SearchTool` returns recipe URLs

## Dependencies

- `reqwest` — HTTP client
- `scraper` or `select` — HTML parsing
- `url` — URL handling
- `serde_json` — JSON
- `framework/tools` — `Tool` trait, `ToolRegistry`
- `domain/recipe` — knowledge store, recipe db, models

## Size

Estimate: ~800-1000 lines

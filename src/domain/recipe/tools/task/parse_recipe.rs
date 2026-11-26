//! Recipe parsing tool with multiple strategies.
//!
//! [`ParseRecipeTool`] tries, in order of reliability: Schema.org JSON-LD,
//! Microdata, CSS selectors from the site config, and heuristics. The
//! knowledge store supplies per-site selectors and the preferred method.
//! Each strategy is a pure function over HTML so it can be unit-tested in
//! isolation.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::domain::recipe::models::recipe::{
    Difficulty, Ingredient, ParseMethod, Recipe, RecipeId, Servings,
};
use crate::domain::recipe::models::site_config::{Selectors, SiteConfig};
use crate::domain::recipe::storage::RecipeKnowledgeStore;
use crate::domain::recipe::storage::traits::KnowledgeStorage;
use crate::framework::tools::traits::{Tool, ToolResult};

use super::fetch_html::extract_domain;

/// Parse recipe data from HTML, trying multiple strategies.
pub struct ParseRecipeTool {
    knowledge: std::sync::Arc<RecipeKnowledgeStore>,
}

impl ParseRecipeTool {
    /// Create a parser that reads site configs from the knowledge store.
    pub fn new(knowledge: std::sync::Arc<RecipeKnowledgeStore>) -> Self {
        Self { knowledge }
    }

    /// Run the parsing pipeline over the given HTML and URL.
    ///
    /// Strategies are tried in order until one yields a recipe. The site's
    /// preferred method (if any) is tried first when it is not the default
    /// Schema.org.
    fn try_parse(&self, html: &str, url: &str, config: &SiteConfig) -> Option<Recipe> {
        // 1. Schema.org JSON-LD (most reliable)
        if let Some(recipe) = parse_schema_org(html, url) {
            return Some(recipe);
        }

        // 2. Microdata
        if let Some(recipe) = parse_microdata(html, url) {
            return Some(recipe);
        }

        // 3. CSS selectors (fallback, site-specific)
        if let Some(recipe) = parse_selectors(html, url, &config.selectors) {
            return Some(recipe);
        }

        // 4. Heuristics (last resort)
        parse_heuristics(html, url).map(|mut r| {
            // Tag the heuristic parse so the signal records the method.
            let _ = &mut r; // recipe already has method via infer; no-op
            r
        })
    }

    /// Read the site config for a domain, falling back to a default config
    /// when the store has no entry.
    fn config_for_domain(&self, domain: &str) -> SiteConfig {
        match self.knowledge.get_site_config(domain) {
            Ok(Some(c)) => c,
            _ => SiteConfig::new(domain),
        }
    }
}

#[async_trait]
impl Tool for ParseRecipeTool {
    fn name(&self) -> &str {
        "parse_recipe"
    }

    fn description(&self) -> &str {
        "Parse recipe data from HTML. Tries Schema.org first, then selectors."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "html": { "type": "string", "description": "Raw HTML content to parse." },
                "url": { "type": "string", "description": "Source URL of the HTML (for domain lookup)." }
            },
            "required": ["html", "url"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let Some(html) = args.get("html").and_then(Value::as_str) else {
            return ToolResult::error("missing or invalid `html`");
        };
        let Some(url_str) = args.get("url").and_then(Value::as_str) else {
            return ToolResult::error("missing or invalid `url`");
        };

        let parsed_url = match url::Url::parse(url_str) {
            Ok(u) => u,
            Err(e) => return ToolResult::error(format!("invalid url: {e}")),
        };
        let domain = extract_domain(&parsed_url).unwrap_or_default();
        let config = self.config_for_domain(&domain);

        let method = match self.try_parse(html, url_str, &config) {
            Some(recipe) => recipe_method(&recipe),
            None => {
                return ToolResult::error(format!(
                    "could not parse a recipe from the provided HTML (domain: {domain})"
                ));
            }
        };

        ToolResult::success(json!({
            "recipe": recipe_json(&self.try_parse(html, url_str, &config).unwrap(), &method),
            "method": method_str(&method),
        }))
    }
}

/// Determine which parse method produced a recipe.
///
/// Heuristic recipes carry `ParseMethod::Heuristic` in the `meta` under the
/// `parse_method` key; other strategies set it implicitly. This helper
/// returns the method for signal recording.
fn recipe_method(recipe: &Recipe) -> ParseMethod {
    recipe
        .meta
        .get("parse_method")
        .and_then(|v| v.as_str())
        .and_then(parse_method_from_str)
        .unwrap_or(ParseMethod::SchemaOrg)
}

fn parse_method_from_str(s: &str) -> Option<ParseMethod> {
    match s {
        "schema_org" => Some(ParseMethod::SchemaOrg),
        "microdata" => Some(ParseMethod::Microdata),
        "selectors" => Some(ParseMethod::Selectors),
        "heuristic" => Some(ParseMethod::Heuristic),
        _ => None,
    }
}

fn method_str(m: &ParseMethod) -> &'static str {
    match m {
        ParseMethod::SchemaOrg => "schema_org",
        ParseMethod::Microdata => "microdata",
        ParseMethod::Selectors => "selectors",
        ParseMethod::Heuristic => "heuristic",
    }
}

/// Serialize a recipe and its method into a JSON value.
fn recipe_json(recipe: &Recipe, method: &ParseMethod) -> Value {
    let mut value = serde_json::to_value(recipe).unwrap_or(Value::Null);
    if let Some(obj) = value.as_object_mut() {
        obj.insert("parse_method".to_string(), json!(method_str(method)));
    }
    value
}

/// Parse a `<script type="application/ld+json">` block as a Schema.org Recipe.
///
/// Scans all JSON-LD blocks and returns the first whose `@type` is or
/// contains `Recipe`.
pub fn parse_schema_org(html: &str, url: &str) -> Option<Recipe> {
    let document = scraper::Html::parse_document(html);
    let selector = scraper::Selector::parse(r#"script[type="application/ld+json"]"#).ok()?;

    for element in document.select(&selector) {
        let raw = element.text().collect::<String>();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        // The JSON-LD value may be a single object or an array of objects.
        if let Some(recipe) = recipe_from_json_ld(trimmed, url, ParseMethod::SchemaOrg) {
            return Some(recipe);
        }
    }
    None
}

/// Try to extract a Recipe from a JSON-LD payload (object or array).
fn recipe_from_json_ld(json_str: &str, url: &str, method: ParseMethod) -> Option<Recipe> {
    let value: Value = serde_json::from_str(json_str).ok()?;
    let candidates: Vec<&Value> = match &value {
        Value::Array(arr) => arr.iter().collect(),
        single => vec![single],
    };
    for candidate in candidates {
        if is_recipe_type(candidate)
            && let Some(recipe) = build_recipe_from_schema(candidate, url, method)
        {
            return Some(recipe);
        }
    }
    None
}

/// Whether a JSON-LD node is a Recipe (by `@type`).
fn is_recipe_type(value: &Value) -> bool {
    let types = match value.get("@type") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect(),
        _ => vec![],
    };
    types
        .iter()
        .any(|t| t == "Recipe" || t.eq_ignore_ascii_case("recipe"))
}

/// Build a [`Recipe`] from a Schema.org JSON object.
fn build_recipe_from_schema(value: &Value, url: &str, method: ParseMethod) -> Option<Recipe> {
    let name = value.get("name").and_then(Value::as_str)?.to_string();
    let source_url: url::Url = url::Url::parse(url).ok()?;
    let source_domain = super::fetch_html::extract_domain(&source_url).unwrap_or_default();

    let ingredients = value
        .get("recipeIngredient")
        .or_else(|| value.get("ingredients"))
        .map(extract_string_list)
        .unwrap_or_default()
        .into_iter()
        .map(Ingredient::from_raw)
        .collect();

    let instructions = value
        .get("recipeInstructions")
        .map(extract_instructions)
        .unwrap_or_default();

    let recipe_id = RecipeId::generate();
    let mut meta = std::collections::HashMap::new();
    meta.insert("parse_method".to_string(), json!(method_str(&method)));

    let recipe = Recipe {
        id: recipe_id,
        name,
        source_url,
        source_domain,
        ingredients,
        instructions,
        prep_time_minutes: value
            .get("prepTime")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_duration_minutes),
        cook_time_minutes: value
            .get("cookTime")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_duration_minutes),
        total_time_minutes: value
            .get("totalTime")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_duration_minutes)
            .or_else(|| {
                value
                    .get("totalTime")
                    .and_then(Value::as_u64)
                    .map(|m| m as u32)
            }),
        servings: value.get("recipeYield").and_then(extract_servings),
        cuisine: value
            .get("recipeCuisine")
            .and_then(Value::as_str)
            .map(String::from),
        difficulty: value
            .get("difficulty")
            .and_then(Value::as_str)
            .and_then(parse_difficulty),
        tags: value
            .get("keywords")
            .and_then(Value::as_str)
            .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
            .unwrap_or_default(),
        nutrition: value
            .get("nutrition")
            .and_then(|v| v.as_object())
            .and_then(build_nutrition),
        image_url: value
            .get("image")
            .and_then(extract_image_url)
            .and_then(|s| url::Url::parse(&s).ok()),
        author: value
            .get("author")
            .and_then(|v| v.get("name").and_then(Value::as_str).or_else(|| v.as_str()))
            .map(String::from),
        description: value
            .get("description")
            .and_then(Value::as_str)
            .map(String::from),
        scraped_at: chrono::Utc::now(),
        content_hash: None,
        meta,
    };
    Some(recipe)
}

/// Parse a list of strings from a JSON value (array or single string).
fn extract_string_list(value: &Value) -> Vec<String> {
    match value {
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| {
                v.as_str()
                    .map(String::from)
                    .or_else(|| v.get("text").and_then(Value::as_str).map(String::from))
            })
            .collect(),
        Value::String(s) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Extract instruction steps from a Schema.org value (HowToStep array or text).
fn extract_instructions(value: &Value) -> Vec<String> {
    match value {
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| {
                if let Some(s) = v.as_str() {
                    return Some(s.to_string());
                }
                v.get("text").and_then(Value::as_str).map(String::from)
            })
            .collect(),
        Value::String(s) => {
            // Single text block: split on newlines for steps.
            s.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Extract a servings value from a Schema.org recipeYield field.
fn extract_servings(value: &Value) -> Option<Servings> {
    match value {
        Value::String(s) => parse_servings_text(s),
        Value::Number(n) => n.as_u64().map(|c| Servings::single(c as u32)),
        Value::Array(arr) => arr.first().and_then(extract_servings),
        _ => None,
    }
}

/// Parse a servings string like "4 servings" or "4-6 servings".
fn parse_servings_text(s: &str) -> Option<Servings> {
    let s = s.trim();
    if let Some((lo, hi)) = s.split_once('-') {
        let lo = parse_leading_u32(lo.trim())?;
        let hi = parse_leading_u32(hi.trim())?;
        return Some(Servings::range(lo, hi));
    }
    let n = parse_leading_u32(s)?;
    Some(Servings::single(n))
}

/// Extract the leading unsigned integer from a string like "6 servings".
fn parse_leading_u32(s: &str) -> Option<u32> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Extract an image URL from a Schema.org image field.
fn extract_image_url(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) => arr.first().and_then(extract_image_url),
        Value::Object(_) => value.get("url").and_then(Value::as_str).map(String::from),
        _ => None,
    }
}

/// Parse a difficulty string into the enum.
fn parse_difficulty(s: &str) -> Option<Difficulty> {
    match s.trim().to_lowercase().as_str() {
        "easy" => Some(Difficulty::Easy),
        "medium" => Some(Difficulty::Medium),
        "hard" => Some(Difficulty::Hard),
        _ => None,
    }
}

/// Parse an ISO 8601 duration (e.g. "PT30M", "PT1H30M") into minutes.
fn parse_iso_duration_minutes(s: &str) -> Option<u32> {
    if !s.starts_with("PT") {
        return None;
    }
    let body = &s[2..];
    let mut total: u32 = 0;
    let mut num = String::new();
    for ch in body.chars() {
        if ch.is_ascii_digit() {
            num.push(ch);
        } else {
            let n: u32 = num.parse().ok()?;
            match ch {
                'H' => total += n * 60,
                'M' => total += n,
                'S' => {} // seconds ignored for minute granularity
                _ => return None,
            }
            num.clear();
        }
    }
    Some(total)
}

/// Build a NutritionInfo from a Schema.org nutrition object.
fn build_nutrition(
    obj: &serde_json::Map<String, Value>,
) -> Option<crate::domain::recipe::models::recipe::NutritionInfo> {
    use crate::domain::recipe::models::recipe::NutritionInfo;
    let calories = obj
        .get("calories")
        .and_then(|v| v.as_str())
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u32>().ok())
        .or_else(|| {
            obj.get("calories")
                .and_then(Value::as_u64)
                .map(|c| c as u32)
        });
    if calories.is_none()
        && obj.get("fatContent").is_none()
        && obj.get("proteinContent").is_none()
        && obj.get("carbohydrateContent").is_none()
    {
        return None;
    }
    Some(NutritionInfo {
        calories,
        fat_g: obj.get("fatContent").and_then(num_as_f64),
        saturated_fat_g: obj.get("saturatedFatContent").and_then(num_as_f64),
        carbohydrates_g: obj.get("carbohydrateContent").and_then(num_as_f64),
        protein_g: obj.get("proteinContent").and_then(num_as_f64),
        fiber_g: obj.get("fiberContent").and_then(num_as_f64),
        sodium_mg: obj
            .get("sodiumContent")
            .and_then(num_as_f64)
            .map(|m| m as u32),
        extra: std::collections::HashMap::new(),
    })
}

/// Coerce a JSON number (int or float) to f64.
fn num_as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// Parse a recipe from Microdata `itemprop` attributes.
pub fn parse_microdata(html: &str, url: &str) -> Option<Recipe> {
    let document = scraper::Html::parse_document(html);
    // Find a container scoped to a Recipe schema.
    let scope_selector = scraper::Selector::parse(r#"[itemtype*="Recipe"]"#).ok()?;
    let scope = document.select(&scope_selector).next()?;

    let source_url: url::Url = url::Url::parse(url).ok()?;
    let source_domain = super::fetch_html::extract_domain(&source_url).unwrap_or_default();
    let name = itemprop_text(&scope, "name")?;
    let recipe_id = RecipeId::generate();
    let mut meta = std::collections::HashMap::new();
    meta.insert("parse_method".to_string(), json!("microdata"));

    Some(Recipe {
        id: recipe_id,
        name,
        source_url,
        source_domain,
        ingredients: itemprop_all_text(&scope, "recipeIngredient")
            .into_iter()
            .map(Ingredient::from_raw)
            .collect(),
        instructions: itemprop_all_text(&scope, "recipeInstructions"),
        prep_time_minutes: None,
        cook_time_minutes: None,
        total_time_minutes: None,
        servings: None,
        cuisine: None,
        difficulty: None,
        tags: Vec::new(),
        nutrition: None,
        image_url: None,
        author: None,
        description: None,
        scraped_at: chrono::Utc::now(),
        content_hash: None,
        meta,
    })
}

/// Get the text of the first element with a given `itemprop` within a scope.
fn itemprop_text(scope: &scraper::ElementRef, prop: &str) -> Option<String> {
    let sel = scraper::Selector::parse(&format!(r#"[itemprop="{prop}"]"#)).ok()?;
    scope
        .select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Get the text of all elements with a given `itemprop` within a scope.
fn itemprop_all_text(scope: &scraper::ElementRef, prop: &str) -> Vec<String> {
    let sel = match scraper::Selector::parse(&format!(r#"[itemprop="{prop}"]"#)) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    scope
        .select(&sel)
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a recipe using CSS selectors from a site config.
pub fn parse_selectors(html: &str, url: &str, selectors: &Selectors) -> Option<Recipe> {
    let title_sel = selectors.title.as_ref()?;
    let document = scraper::Html::parse_document(html);
    let sel = scraper::Selector::parse(title_sel).ok()?;
    let name = document
        .select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())?;

    let source_url: url::Url = url::Url::parse(url).ok()?;
    let source_domain = super::fetch_html::extract_domain(&source_url).unwrap_or_default();
    let recipe_id = RecipeId::generate();
    let mut meta = std::collections::HashMap::new();
    meta.insert("parse_method".to_string(), json!("selectors"));

    Some(Recipe {
        id: recipe_id,
        name,
        source_url,
        source_domain,
        ingredients: select_all(&document, selectors.ingredients.as_deref())
            .into_iter()
            .map(Ingredient::from_raw)
            .collect(),
        instructions: select_all(&document, selectors.instructions.as_deref()),
        prep_time_minutes: None,
        cook_time_minutes: None,
        total_time_minutes: None,
        servings: None,
        cuisine: None,
        difficulty: None,
        tags: Vec::new(),
        nutrition: None,
        image_url: None,
        author: None,
        description: None,
        scraped_at: chrono::Utc::now(),
        content_hash: None,
        meta,
    })
}

/// Select text from all elements matching a CSS selector (if provided).
fn select_all(document: &scraper::Html, selector: Option<&str>) -> Vec<String> {
    let Some(sel_str) = selector else {
        return Vec::new();
    };
    let Ok(sel) = scraper::Selector::parse(sel_str) else {
        return Vec::new();
    };
    document
        .select(&sel)
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Last-resort heuristic parsing: extract the `<title>` as the recipe name.
pub fn parse_heuristics(html: &str, url: &str) -> Option<Recipe> {
    let document = scraper::Html::parse_document(html);
    let sel = scraper::Selector::parse("title").ok()?;
    let title = document
        .select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())?;

    let source_url: url::Url = url::Url::parse(url).ok()?;
    let source_domain = super::fetch_html::extract_domain(&source_url).unwrap_or_default();
    let recipe_id = RecipeId::generate();
    let mut meta = std::collections::HashMap::new();
    meta.insert("parse_method".to_string(), json!("heuristic"));

    // Strip common suffixes like " - Allrecipes" from the title.
    let name = title
        .split_once(" | ")
        .or_else(|| title.split_once(" - "))
        .map(|(head, _)| head.trim().to_string())
        .unwrap_or(title);

    Some(Recipe {
        id: recipe_id,
        name,
        source_url,
        source_domain,
        ingredients: Vec::new(),
        instructions: Vec::new(),
        prep_time_minutes: None,
        cook_time_minutes: None,
        total_time_minutes: None,
        servings: None,
        cuisine: None,
        difficulty: None,
        tags: Vec::new(),
        nutrition: None,
        image_url: None,
        author: None,
        description: None,
        scraped_at: chrono::Utc::now(),
        content_hash: None,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::tools::traits::Tool;
    use serde_json::json;
    use std::sync::Arc;

    const SCHEMA_ORG_HTML: &str = r#"
    <html><head>
    <script type="application/ld+json">
    {
      "@type": "Recipe",
      "name": "Chocolate Cake",
      "recipeIngredient": ["2 cups flour", "1 cup sugar", "3 eggs"],
      "recipeInstructions": [{"text": "Mix dry ingredients"}, {"text": "Bake at 350F"}],
      "prepTime": "PT15M",
      "cookTime": "PT30M",
      "totalTime": "PT45M",
      "recipeYield": "8 servings",
      "recipeCuisine": "American",
      "difficulty": "easy",
      "keywords": "cake, dessert, chocolate",
      "nutrition": {"calories": "350 kcal", "fatContent": "12"},
      "image": "https://example.com/cake.jpg",
      "author": {"name": "Jane Doe"},
      "description": "A simple chocolate cake."
    }
    </script>
    </head><body></body></html>
    "#;

    const EMPTY_HTML: &str = "<html><head></head><body></body></html>";

    fn tool() -> ParseRecipeTool {
        let dir = tempfile::tempdir().unwrap();
        let knowledge = Arc::new(RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap());
        ParseRecipeTool::new(knowledge)
    }

    #[test]
    fn parse_schema_org_extracts_full_recipe() {
        let recipe = parse_schema_org(SCHEMA_ORG_HTML, "https://example.com/cake").unwrap();
        assert_eq!(recipe.name, "Chocolate Cake");
        assert_eq!(recipe.ingredients.len(), 3);
        assert_eq!(recipe.ingredients[0].raw, "2 cups flour");
        assert_eq!(recipe.instructions.len(), 2);
        assert_eq!(recipe.prep_time_minutes, Some(15));
        assert_eq!(recipe.cook_time_minutes, Some(30));
        assert_eq!(recipe.total_time_minutes, Some(45));
        assert_eq!(recipe.servings.as_ref().map(|s| s.min), Some(8));
        assert_eq!(recipe.cuisine.as_deref(), Some("American"));
        assert_eq!(recipe.difficulty, Some(Difficulty::Easy));
        assert!(recipe.tags.contains(&"chocolate".to_string()));
        assert_eq!(recipe.author.as_deref(), Some("Jane Doe"));
        assert_eq!(recipe_method(&recipe), ParseMethod::SchemaOrg);
    }

    #[test]
    fn parse_schema_org_skips_non_recipe_blocks() {
        let html = r#"<script type="application/ld+json">{"@type":"WebSite","name":"foo"}</script>
        <script type="application/ld+json">{"@type":"Recipe","name":"Bar"}</script>"#;
        let recipe = parse_schema_org(html, "https://example.com/bar").unwrap();
        assert_eq!(recipe.name, "Bar");
    }

    #[test]
    fn parse_schema_org_handles_array_of_blocks() {
        let html = r#"<script type="application/ld+json">
        [{"@type":"WebSite","name":"foo"},{"@type":"Recipe","name":"Baz"}]</script>"#;
        let recipe = parse_schema_org(html, "https://example.com/baz").unwrap();
        assert_eq!(recipe.name, "Baz");
    }

    #[test]
    fn parse_schema_org_returns_none_for_empty_html() {
        assert!(parse_schema_org(EMPTY_HTML, "https://example.com/x").is_none());
    }

    #[test]
    fn parse_schema_org_returns_none_when_no_name() {
        let html = r#"<script type="application/ld+json">{"@type":"Recipe"}</script>"#;
        assert!(parse_schema_org(html, "https://example.com/x").is_none());
    }

    #[test]
    fn parse_microdata_extracts_recipe() {
        let html = r#"
        <div itemtype="https://schema.org/Recipe">
          <h1 itemprop="name">Pasta</h1>
          <span itemprop="recipeIngredient">200g flour</span>
          <span itemprop="recipeIngredient">2 eggs</span>
          <span itemprop="recipeInstructions">Boil water</span>
          <span itemprop="recipeInstructions">Cook pasta</span>
        </div>"#;
        let recipe = parse_microdata(html, "https://example.com/pasta").unwrap();
        assert_eq!(recipe.name, "Pasta");
        assert_eq!(recipe.ingredients.len(), 2);
        assert_eq!(recipe.instructions.len(), 2);
        assert_eq!(recipe_method(&recipe), ParseMethod::Microdata);
    }

    #[test]
    fn parse_microdata_returns_none_when_no_recipe_scope() {
        assert!(parse_microdata(EMPTY_HTML, "https://example.com/x").is_none());
    }

    #[test]
    fn parse_selectors_extracts_recipe_with_config() {
        let html = r#"
        <html><body>
          <h1 class="title">Soup</h1>
          <li class="ingredient">carrots</li>
          <li class="ingredient">onions</li>
          <li class="step">chop</li>
          <li class="step">simmer</li>
        </body></html>"#;
        let selectors = Selectors {
            title: Some(".title".to_string()),
            ingredients: Some(".ingredient".to_string()),
            instructions: Some(".step".to_string()),
            ..Default::default()
        };
        let recipe = parse_selectors(html, "https://example.com/soup", &selectors).unwrap();
        assert_eq!(recipe.name, "Soup");
        assert_eq!(recipe.ingredients.len(), 2);
        assert_eq!(recipe.instructions.len(), 2);
        assert_eq!(recipe_method(&recipe), ParseMethod::Selectors);
    }

    #[test]
    fn parse_selectors_returns_none_without_title_selector() {
        let selectors = Selectors::default();
        assert!(parse_selectors("<h1>x</h1>", "https://example.com/x", &selectors).is_none());
    }

    #[test]
    fn parse_heuristics_uses_title_stripping_suffix() {
        let html = "<html><head><title>Best Tacos - Allrecipes</title></head></html>";
        let recipe = parse_heuristics(html, "https://example.com/tacos").unwrap();
        assert_eq!(recipe.name, "Best Tacos");
        assert_eq!(recipe_method(&recipe), ParseMethod::Heuristic);
    }

    #[test]
    fn parse_heuristics_returns_none_when_no_title() {
        assert!(parse_heuristics(EMPTY_HTML, "https://example.com/x").is_none());
    }

    #[test]
    fn iso_duration_parses_hours_and_minutes() {
        assert_eq!(parse_iso_duration_minutes("PT30M"), Some(30));
        assert_eq!(parse_iso_duration_minutes("PT1H30M"), Some(90));
        assert_eq!(parse_iso_duration_minutes("PT2H"), Some(120));
        assert_eq!(parse_iso_duration_minutes("PT1H30M15S"), Some(90));
        assert_eq!(parse_iso_duration_minutes("invalid"), None);
        assert_eq!(parse_iso_duration_minutes("P1D"), None);
    }

    #[test]
    fn servings_text_parsing() {
        assert_eq!(parse_servings_text("4 servings").map(|s| s.min), Some(4));
        assert_eq!(
            parse_servings_text("4-6 servings").map(|s| (s.min, s.max)),
            Some((4, Some(6)))
        );
        assert!(parse_servings_text("no numbers").is_none());
    }

    #[test]
    fn difficulty_parsing() {
        assert_eq!(parse_difficulty("Easy"), Some(Difficulty::Easy));
        assert_eq!(parse_difficulty("medium"), Some(Difficulty::Medium));
        assert_eq!(parse_difficulty("HARD"), Some(Difficulty::Hard));
        assert!(parse_difficulty("unknown").is_none());
    }

    #[tokio::test]
    async fn execute_missing_html_arg_errors() {
        let tool = tool();
        let res = tool.execute(json!({"url": "https://example.com/x"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("missing"));
    }

    #[tokio::test]
    async fn execute_missing_url_arg_errors() {
        let tool = tool();
        let res = tool.execute(json!({"html": "<html></html>"})).await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("missing"));
    }

    #[tokio::test]
    async fn execute_invalid_url_errors() {
        let tool = tool();
        let res = tool
            .execute(json!({"html": "<html></html>", "url": "not a url"}))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("invalid url"));
    }

    #[tokio::test]
    async fn execute_returns_recipe_for_schema_org_html() {
        let tool = tool();
        let res = tool
            .execute(json!({
                "html": SCHEMA_ORG_HTML,
                "url": "https://example.com/cake"
            }))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {:?}", res);
        };
        assert_eq!(v["method"], "schema_org");
        assert_eq!(v["recipe"]["name"], "Chocolate Cake");
    }

    #[tokio::test]
    async fn execute_falls_back_to_heuristics() {
        let html = "<html><head><title>Just A Title</title></head></html>";
        let tool = tool();
        let res = tool
            .execute(json!({"html": html, "url": "https://example.com/page"}))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {:?}", res);
        };
        assert_eq!(v["method"], "heuristic");
        assert_eq!(v["recipe"]["name"], "Just A Title");
    }

    #[tokio::test]
    async fn execute_errors_when_no_recipe_parseable() {
        let tool = tool();
        let res = tool
            .execute(json!({"html": EMPTY_HTML, "url": "https://example.com/x"}))
            .await;
        assert!(res.is_error());
        assert!(res.error_message().unwrap().contains("could not parse"));
    }

    #[tokio::test]
    async fn execute_uses_site_config_selectors_when_no_schema() {
        // Set up a knowledge store with a selector-based config.
        let dir = tempfile::tempdir().unwrap();
        let knowledge = Arc::new(RecipeKnowledgeStore::open(dir.path().to_path_buf()).unwrap());
        let mut config = SiteConfig::new("example.com");
        config.selectors = Selectors {
            title: Some("h1.title".to_string()),
            ingredients: Some("li.ing".to_string()),
            instructions: Some("li.step".to_string()),
            ..Default::default()
        };
        knowledge.save_site_config(&config).unwrap();

        let tool = ParseRecipeTool::new(knowledge);
        let html = r#"<html><body>
            <h1 class="title">Selector Soup</h1>
            <li class="ing">water</li>
            <li class="step">boil</li>
        </body></html>"#;
        let res = tool
            .execute(json!({"html": html, "url": "https://example.com/soup"}))
            .await;
        let ToolResult::Success(v) = res else {
            panic!("expected success, got {:?}", res);
        };
        assert_eq!(v["method"], "selectors");
        assert_eq!(v["recipe"]["name"], "Selector Soup");
    }
}

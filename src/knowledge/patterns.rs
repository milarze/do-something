//! Pattern discovery and matching.
//!
//! Provides utilities for matching success patterns and anti-patterns
//! to guide scraping behavior.

use crate::models::knowledge::{AntiPattern, AntiPatternAction, Patterns, SuccessPattern};
use crate::models::ParseMethod;

/// Matcher for discovered patterns.
pub struct PatternMatcher;

impl PatternMatcher {
    /// Check if a URL matches any anti-pattern for a domain.
    ///
    /// Returns the matching anti-pattern if found, or None.
    pub fn matches_anti_pattern<'a>(
        patterns: &'a Patterns,
        url: &str,
        domain: &str,
    ) -> Option<&'a AntiPattern> {
        patterns
            .anti_patterns
            .iter()
            .find(|ap| ap.matches(url, domain))
    }

    /// Get success patterns that apply to a domain.
    ///
    /// Returns patterns where the domain matches or patterns with no domain restriction.
    pub fn get_success_patterns<'a>(patterns: &'a Patterns, domain: &str) -> Vec<&'a SuccessPattern> {
        patterns
            .success_patterns
            .iter()
            .filter(|p| p.applies_to(domain))
            .collect()
    }

    /// Get success patterns for a domain and parse method.
    pub fn get_success_patterns_for_method<'a>(
        patterns: &'a Patterns,
        domain: &str,
        method: ParseMethod,
    ) -> Vec<&'a SuccessPattern> {
        patterns
            .success_patterns
            .iter()
            .filter(|p| p.applies_to(domain))
            .filter(|p| {
                // Include patterns that mention this method in description
                let desc = p.description.to_lowercase();
                let method_str = format!("{:?}", method).to_lowercase();
                desc.contains(&method_str)
            })
            .collect()
    }

    /// Check if a URL should be skipped based on anti-patterns.
    pub fn should_skip_url(patterns: &Patterns, url: &str, domain: &str) -> bool {
        Self::matches_anti_pattern(patterns, url, domain)
            .map(|ap| ap.action == AntiPatternAction::SkipUrl)
            .unwrap_or(false)
    }

    /// Get the action for a matched anti-pattern.
    pub fn get_anti_pattern_action(
        patterns: &Patterns,
        url: &str,
        domain: &str,
    ) -> Option<AntiPatternAction> {
        Self::matches_anti_pattern(patterns, url, domain)
            .map(|ap| ap.action)
    }

    /// Default built-in patterns for common cases.
    ///
    /// These provide baseline patterns that work across many recipe sites.
    pub fn default_patterns() -> Patterns {
        let mut patterns = Patterns::default();

        // Success patterns
        patterns.success_patterns.extend(vec![
            SuccessPattern {
                description: "Schema.org JSON-LD parsing works reliably".to_string(),
                sites: vec![], // Applies to all sites
                success_rate: 0.85,
                sample_size: 1000,
                confidence: 0.9,
            },
            SuccessPattern {
                description: "Recipes with /recipe/ in URL parse successfully".to_string(),
                sites: vec![],
                success_rate: 0.75,
                sample_size: 500,
                confidence: 0.7,
            },
        ]);

        // Anti-patterns
        patterns.anti_patterns.extend(vec![
            AntiPattern {
                description: "URL ends with /video - video content not a recipe".to_string(),
                sites: vec!["tasty.co".to_string()],
                failure_rate: 0.95,
                action: AntiPatternAction::SkipUrl,
                sample_size: 50,
                confidence: 0.85,
            },
            AntiPattern {
                description: "URL contains /gallery/ - slideshow format".to_string(),
                sites: vec!["allrecipes.com".to_string()],
                failure_rate: 0.80,
                action: AntiPatternAction::SkipUrl,
                sample_size: 30,
                confidence: 0.75,
            },
            AntiPattern {
                description: "URL contains /videos/ - video content".to_string(),
                sites: vec!["foodnetwork.com".to_string()],
                failure_rate: 0.90,
                action: AntiPatternAction::SkipUrl,
                sample_size: 40,
                confidence: 0.80,
            },
            AntiPattern {
                description: "Rate limit detected - slow down requests".to_string(),
                sites: vec![],
                failure_rate: 0.60,
                action: AntiPatternAction::SlowDown,
                sample_size: 100,
                confidence: 0.85,
            },
        ]);

        patterns.version = 1;
        patterns
    }

    /// Add a new success pattern.
    pub fn add_success_pattern(patterns: &mut Patterns, pattern: SuccessPattern) {
        // Check for duplicates
        let is_duplicate = patterns.success_patterns.iter().any(|p| {
            p.description == pattern.description && p.sites == pattern.sites
        });

        if !is_duplicate {
            patterns.success_patterns.push(pattern);
            patterns.version += 1;
        }
    }

    /// Add a new anti-pattern.
    pub fn add_anti_pattern(patterns: &mut Patterns, pattern: AntiPattern) {
        // Check for duplicates
        let is_duplicate = patterns.anti_patterns.iter().any(|p| {
            p.description == pattern.description && p.sites == pattern.sites
        });

        if !is_duplicate {
            patterns.anti_patterns.push(pattern);
            patterns.version += 1;
        }
    }

    /// Remove patterns below confidence threshold.
    pub fn prune_low_confidence(patterns: &mut Patterns, min_confidence: f64) -> usize {
        let before_count = patterns.success_patterns.len() + patterns.anti_patterns.len();

        patterns
            .success_patterns
            .retain(|p| p.confidence >= min_confidence);
        patterns
            .anti_patterns
            .retain(|p| p.confidence >= min_confidence);

        let after_count = patterns.success_patterns.len() + patterns.anti_patterns.len();
        before_count - after_count
    }

    /// Sort patterns by confidence (highest first).
    pub fn sort_by_confidence(patterns: &mut Patterns) {
        patterns
            .success_patterns
            .sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        patterns
            .anti_patterns
            .sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    }
}

impl SuccessPattern {
    /// Check if this pattern applies to a given domain.
    pub fn applies_to(&self, domain: &str) -> bool {
        // Empty sites list means applies to all
        self.sites.is_empty() || self.sites.iter().any(|s| s == domain)
    }
}

impl AntiPattern {
    /// Check if this anti-pattern matches a URL/domain combination.
    pub fn matches(&self, url: &str, domain: &str) -> bool {
        // Check if domain applies
        let domain_matches = self.sites.is_empty() 
            || self.sites.iter().any(|s| s == domain);

        if !domain_matches {
            return false;
        }

        // Parse URL for more precise matching
        let url_lower = url.to_lowercase();
        let desc_lower = self.description.to_lowercase();

        // Match specific path patterns with boundaries
        // "URL ends with /video" -> exact suffix match
        if desc_lower.contains("ends with /video") || desc_lower.contains("url ends with /video") {
            return url_lower.ends_with("/video") 
                || url_lower.ends_with("/video/");
        }

        // "URL ends with /video - ..." format from default patterns
        if desc_lower.contains("/video") {
            // Check for video path segment (not just substring)
            return url_lower.contains("/video/") 
                || url_lower.ends_with("/video");
        }

        // Gallery pattern
        if desc_lower.contains("/gallery") {
            return url_lower.contains("/gallery/") 
                || url_lower.ends_with("/gallery");
        }

        // Videos (plural) pattern
        if desc_lower.contains("/videos") {
            return url_lower.contains("/videos/") 
                || url_lower.ends_with("/videos");
        }

        // Shows pattern
        if desc_lower.contains("/shows") {
            return url_lower.contains("/shows/") 
                || url_lower.ends_with("/shows");
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_video_anti_pattern() {
        let patterns = PatternMatcher::default_patterns();

        let result = PatternMatcher::matches_anti_pattern(
            &patterns,
            "https://tasty.co/recipe/123/video",
            "tasty.co",
        );

        assert!(result.is_some());
        let ap = result.unwrap();
        assert_eq!(ap.action, AntiPatternAction::SkipUrl);
    }

    #[test]
    fn does_not_match_recipe_url() {
        let patterns = PatternMatcher::default_patterns();

        let result = PatternMatcher::matches_anti_pattern(
            &patterns,
            "https://tasty.co/recipe/123",
            "tasty.co",
        );

        assert!(result.is_none());
    }

    #[test]
    fn does_not_match_different_domain() {
        let patterns = PatternMatcher::default_patterns();

        // Video URL on a different domain shouldn't match tasty.co pattern
        let result = PatternMatcher::matches_anti_pattern(
            &patterns,
            "https://example.com/recipe/123/video",
            "example.com",
        );

        // Only matches if there's a pattern for example.com
        // The default patterns have site-specific ones, so this won't match
        // But the rate limit pattern applies to all sites
        // So we check it doesn't match the /video pattern specifically
        if let Some(ap) = result {
            assert_ne!(ap.description, "URL ends with /video - video content not a recipe");
        }
    }

    #[test]
    fn should_skip_video_urls() {
        let patterns = PatternMatcher::default_patterns();

        assert!(PatternMatcher::should_skip_url(
            &patterns,
            "https://tasty.co/recipe/123/video",
            "tasty.co"
        ));

        assert!(!PatternMatcher::should_skip_url(
            &patterns,
            "https://tasty.co/recipe/123",
            "tasty.co"
        ));
    }

    #[test]
    fn video_tips_url_does_not_match_video_pattern() {
        let patterns = PatternMatcher::default_patterns();

        // "video-tips" should NOT match the /video pattern (no path boundary)
        let result = PatternMatcher::matches_anti_pattern(
            &patterns,
            "https://tasty.co/article/video-tips",
            "tasty.co",
        );
        
        // Should NOT match the video anti-pattern
        if let Some(ap) = result {
            assert!(!ap.description.to_lowercase().contains("video"), 
                "video-tips should not match /video pattern");
        }
    }

    #[test]
    fn gallery_path_matches_correctly() {
        let patterns = PatternMatcher::default_patterns();

        // /gallery/ should match
        assert!(PatternMatcher::matches_anti_pattern(
            &patterns,
            "https://allrecipes.com/recipe/gallery/test",
            "allrecipes.com",
        ).is_some());

        // gallery-item should NOT match
        let result = PatternMatcher::matches_anti_pattern(
            &patterns,
            "https://allrecipes.com/gallery-item/photo",
            "allrecipes.com",
        );
        
        if let Some(ap) = result {
            assert!(!ap.description.to_lowercase().contains("gallery"),
                "gallery-item should not match /gallery pattern");
        }
    }

    #[test]
    fn get_success_patterns_for_domain() {
        let patterns = PatternMatcher::default_patterns();

        let site_patterns = PatternMatcher::get_success_patterns(&patterns, "example.com");
        
        // Should include patterns with empty sites list
        assert!(!site_patterns.is_empty());
    }

    #[test]
    fn success_pattern_applies_to_domain() {
        let universal = SuccessPattern {
            description: "Test".to_string(),
            sites: vec![],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.8,
        };

        assert!(universal.applies_to("any-site.com"));

        let site_specific = SuccessPattern {
            description: "Test".to_string(),
            sites: vec!["allrecipes.com".to_string()],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.8,
        };

        assert!(site_specific.applies_to("allrecipes.com"));
        assert!(!site_specific.applies_to("other.com"));
    }

    #[test]
    fn add_success_pattern_avoids_duplicates() {
        let mut patterns = Patterns::default();
        let pattern = SuccessPattern {
            description: "Test pattern".to_string(),
            sites: vec!["example.com".to_string()],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.8,
        };

        PatternMatcher::add_success_pattern(&mut patterns, pattern.clone());
        assert_eq!(patterns.success_patterns.len(), 1);

        // Adding duplicate should not increase count
        PatternMatcher::add_success_pattern(&mut patterns, pattern);
        assert_eq!(patterns.success_patterns.len(), 1);
    }

    #[test]
    fn add_anti_pattern_avoids_duplicates() {
        let mut patterns = Patterns::default();
        let pattern = AntiPattern {
            description: "Test anti-pattern".to_string(),
            sites: vec!["example.com".to_string()],
            failure_rate: 0.9,
            action: AntiPatternAction::SkipUrl,
            sample_size: 50,
            confidence: 0.8,
        };

        PatternMatcher::add_anti_pattern(&mut patterns, pattern.clone());
        assert_eq!(patterns.anti_patterns.len(), 1);

        // Adding duplicate should not increase count
        PatternMatcher::add_anti_pattern(&mut patterns, pattern);
        assert_eq!(patterns.anti_patterns.len(), 1);
    }

    #[test]
    fn prune_low_confidence() {
        let mut patterns = Patterns::default();

        patterns.success_patterns.push(SuccessPattern {
            description: "High confidence".to_string(),
            sites: vec![],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.9,
        });

        patterns.success_patterns.push(SuccessPattern {
            description: "Low confidence".to_string(),
            sites: vec![],
            success_rate: 0.5,
            sample_size: 5,
            confidence: 0.3,
        });

        let removed = PatternMatcher::prune_low_confidence(&mut patterns, 0.5);
        assert_eq!(removed, 1);
        assert_eq!(patterns.success_patterns.len(), 1);
    }

    #[test]
    fn sort_by_confidence_orders_correctly() {
        let mut patterns = Patterns::default();

        patterns.success_patterns.push(SuccessPattern {
            description: "Medium".to_string(),
            sites: vec![],
            success_rate: 0.7,
            sample_size: 50,
            confidence: 0.7,
        });

        patterns.success_patterns.push(SuccessPattern {
            description: "High".to_string(),
            sites: vec![],
            success_rate: 0.9,
            sample_size: 100,
            confidence: 0.9,
        });

        patterns.success_patterns.push(SuccessPattern {
            description: "Low".to_string(),
            sites: vec![],
            success_rate: 0.5,
            sample_size: 30,
            confidence: 0.5,
        });

        PatternMatcher::sort_by_confidence(&mut patterns);

        assert_eq!(patterns.success_patterns[0].confidence, 0.9);
        assert_eq!(patterns.success_patterns[1].confidence, 0.7);
        assert_eq!(patterns.success_patterns[2].confidence, 0.5);
    }

    #[test]
    fn default_patterns_not_empty() {
        let patterns = PatternMatcher::default_patterns();
        
        assert!(!patterns.success_patterns.is_empty());
        assert!(!patterns.anti_patterns.is_empty());
    }
}

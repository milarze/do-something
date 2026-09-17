//! Recipe-specific domain knowledge.
//!
//! Contains hard-coded knowledge about recipe websites, including:
//! - Site-specific CSS selectors
//! - Rate limiting configurations
//! - URL patterns to skip
//!
//! This knowledge is separate from the agent's learned knowledge in
//! `crate::knowledge`, which adapts based on user feedback.

pub mod site_defaults;

pub use site_defaults::SiteDefaults;

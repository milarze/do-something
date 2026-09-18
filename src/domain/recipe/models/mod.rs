//! Recipe domain models.
//!
//! These types are specific to the recipe domain and follow the naming
//! convention of prefixing with `Recipe` to distinguish from future domains.

pub mod patterns;
pub mod recipe;
pub mod signal;
pub mod site_config;
pub mod user_model;

pub use patterns::{AntiPattern, AntiPatternAction, RecipePatterns, SuccessPattern};
pub use recipe::{Difficulty, Ingredient, NutritionInfo, ParseMethod, Recipe, RecipeId, Servings};
pub use signal::{RecipeSignal, RecipeSignalType};
pub use site_config::SiteConfig;
pub use user_model::UserModel;

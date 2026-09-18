//! Domain layer: concrete implementations for specific use cases.
//!
//! Each domain implements the framework's learning contracts.
//! The recipe domain is the first implementation.

pub mod recipe;

pub use recipe::*;

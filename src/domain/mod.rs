//! Domain-specific knowledge and configurations.
//!
//! This module contains domain-specific knowledge that is separate from
//! the generic agent learning logic in `knowledge/`. While `knowledge/`
//! handles how an agent learns and adapts, this module contains hard-coded
//! knowledge about specific domains (e.g., recipe sites).

pub mod recipes;

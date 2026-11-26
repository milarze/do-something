//! Token estimation for context budgeting.
//!
//! [`TokenCounter`] is the injectable seam between
//! [`ContextManager`](super::context::ContextManager) and any concrete
//! tokenizer. The default [`CharTokenCounter`] is a coarse
//! ~4-chars-per-token heuristic — consistent enough for trimming and unit
//! tests, and replaceable by a real BPE tokenizer later without touching
//! [`ContextManager`](super::context::ContextManager).

/// Counts tokens in a piece of text.
///
/// Injected into [`ContextManager`](super::context::ContextManager) so tests
/// can use deterministic counts instead of a real tokenizer.
pub trait TokenCounter: Send + Sync {
    /// Estimated token count of `text`.
    fn count(&self, text: &str) -> u64;
}

/// Default token estimator: roughly one token per `chars_per_token` characters.
///
/// Deliberately simple — it only needs to be *consistent* for budgeting, not
/// exact. The 4-chars-per-token rule of thumb tracks English prose and JSON
/// well enough to decide what fits in a context window.
#[derive(Debug, Clone, Copy)]
pub struct CharTokenCounter {
    chars_per_token: u32,
}

impl Default for CharTokenCounter {
    fn default() -> Self {
        Self {
            chars_per_token: 4,
        }
    }
}

impl CharTokenCounter {
    /// A counter using the default 4-chars-per-token ratio.
    pub fn new() -> Self {
        Self::default()
    }

    /// A counter with a custom chars-per-token ratio (clamped to ≥ 1).
    pub fn with_ratio(chars_per_token: u32) -> Self {
        Self {
            chars_per_token: chars_per_token.max(1),
        }
    }

    /// The chars-per-token ratio in use.
    pub fn chars_per_token(&self) -> u32 {
        self.chars_per_token
    }
}

impl TokenCounter for CharTokenCounter {
    fn count(&self, text: &str) -> u64 {
        let chars = text.chars().count() as u64;
        chars.div_ceil(self.chars_per_token as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ratio_is_four() {
        assert_eq!(CharTokenCounter::default().chars_per_token(), 4);
        assert_eq!(CharTokenCounter::new().chars_per_token(), 4);
    }

    #[test]
    fn with_ratio_clamps_to_minimum_one() {
        assert_eq!(CharTokenCounter::with_ratio(0).chars_per_token(), 1);
        assert_eq!(CharTokenCounter::with_ratio(7).chars_per_token(), 7);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(CharTokenCounter::new().count(""), 0);
    }

    #[test]
    fn default_rounds_up_per_four_chars() {
        let c = CharTokenCounter::new();
        assert_eq!(c.count("abcd"), 1); // exactly 4
        assert_eq!(c.count("abcde"), 2); // 5 → ceil(5/4) = 2
        assert_eq!(c.count("abcdefgh"), 2); // 8 → 2
    }

    #[test]
    fn ratio_one_counts_chars() {
        let c = CharTokenCounter::with_ratio(1);
        assert_eq!(c.count("abc"), 3);
        assert_eq!(c.count("hello"), 5);
    }
}

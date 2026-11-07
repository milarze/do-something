//! Learning domain contracts.
//!
//! The framework drives the compression loop; the domain supplies the
//! stage implementations via this trait. The framework never inspects
//! `Signal`, `Pattern`, or `Stats` internals beyond
//! [`DomainStats::total_count`] and [`LearningDomain::signal_domain`].

use serde::Serialize;
use serde::de::DeserializeOwned;

/// A domain's aggregated statistics.
///
/// Opaque to the framework beyond its total sample count.
pub trait DomainStats {
    /// Total number of signals represented by these stats.
    fn total_count(&self) -> u64;
}

/// What a domain provides so the framework can run the learning loop.
///
/// The framework owns the loop mechanics (read → aggregate → extract →
/// update → prune). The domain owns what signals mean, what patterns look
/// like, and how knowledge updates.
pub trait LearningDomain: Send + Sync {
    type Signal: Serialize + DeserializeOwned + Clone + Send + Sync;
    type Pattern: Serialize + DeserializeOwned + Clone + Send + Sync;
    type Stats: DomainStats;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Extract the domain key from a signal, for filtering and scheduling.
    ///
    /// Returns `None` for signals not associated with a specific domain.
    /// The framework uses this to group signals without inspecting their
    /// structure.
    fn signal_domain(signal: &Self::Signal) -> Option<&str>;

    /// Stage 1: aggregate raw signals into domain statistics.
    fn aggregate(&self, signals: &[Self::Signal]) -> Self::Stats;

    /// Stage 2: extract patterns from aggregated statistics.
    fn extract_patterns(&self, stats: &Self::Stats) -> Vec<Self::Pattern>;

    /// Stage 3: update knowledge from extracted patterns.
    ///
    /// Returns the number of knowledge entries updated.
    fn update_knowledge(&self, patterns: &[Self::Pattern]) -> Result<u64, Self::Error>;
}

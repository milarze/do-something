//! Pure statistics utilities for confidence scoring.

/// Z-score for a 95% confidence interval.
const Z_95: f64 = 1.96;

/// Wilson score interval lower bound for a binomial proportion.
///
/// Returns the lower bound of the 95% confidence interval for the true
/// success rate given `successes` out of `total` trials. Returns `0.0`
/// when `total` is zero.
///
/// This penalises small samples: 10/10 yields a lower bound well below 1.0,
/// while 100/100 yields a much higher one.
pub fn wilson_score(successes: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = Z_95 * Z_95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let spread = Z_95 * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt() / denominator;
    (center - spread).clamp(0.0, 1.0)
}

/// Sample-size confidence factor in `[0, 1]`.
///
/// Saturates to `1.0` at 30 samples (a common rule-of-thumb threshold for
/// statistical reliability). Returns `0.0` for no samples.
pub fn sample_confidence(n: u64) -> f64 {
    (n as f64 / 30.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_zero_total_returns_zero() {
        assert_eq!(wilson_score(0, 0), 0.0);
        assert_eq!(wilson_score(5, 0), 0.0);
    }

    #[test]
    fn wilson_zero_successes_returns_zero() {
        // No successes -> the lower bound of the interval is 0.
        assert_eq!(wilson_score(0, 10), 0.0);
    }

    #[test]
    fn wilson_all_successes_small_sample_is_conservative() {
        // 10/10: perfect rate but small sample -> lower bound well below 1.0.
        let small = wilson_score(10, 10);
        assert!(small > 0.5 && small < 1.0, "got {small}");
    }

    #[test]
    fn wilson_more_samples_increase_confidence() {
        // Same perfect rate, more samples -> higher lower bound.
        let small = wilson_score(10, 10);
        let large = wilson_score(100, 100);
        assert!(large > small, "large={large} should exceed small={small}");
    }

    #[test]
    fn wilson_half_is_below_half() {
        // 50/100: lower bound should be below the 0.5 point estimate.
        let lower = wilson_score(50, 100);
        assert!(lower < 0.5 && lower > 0.0, "got {lower}");
    }

    #[test]
    fn wilson_is_monotonic_in_successes_for_fixed_total() {
        // More successes at fixed total -> higher lower bound.
        let low = wilson_score(40, 100);
        let high = wilson_score(60, 100);
        assert!(high > low);
    }

    #[test]
    fn sample_confidence_scales_to_one() {
        assert_eq!(sample_confidence(0), 0.0);
        assert!((sample_confidence(15) - 0.5).abs() < 1e-9);
        assert_eq!(sample_confidence(30), 1.0);
        assert_eq!(sample_confidence(1000), 1.0);
    }
}

//! Special functions not in `libm`: the scaled complementary error function and
//! the overflow-safe product exp(a)·erfc(b) used by Mantid's BackToBackExponential.

const SQRT_PI: f64 = 1.772_453_850_905_516;

/// erfcx(x) = exp(x²)·erfc(x): direct below 2, continued fraction (backward recurrence,
/// 80 terms; converges to full precision for x ≥ 2) above.
pub fn erfcx(x: f64) -> f64 {
    if x < 2.0 {
        return (x * x).exp() * libm::erfc(x);
    }
    let mut k = x;
    for n in (1..=80).rev() {
        k = x + (n as f64 * 0.5) / k;
    }
    1.0 / (SQRT_PI * k)
}

/// exp(a)·erfc(b) without overflow.
pub fn expwerfc(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        (a - b * b).exp() * erfcx(b)
    } else {
        a.exp() * libm::erfc(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn erfcx_continuity() {
        for x in [2.0f64, 3.0, 5.0, 10.0] {
            let direct = (x * x).exp() * libm::erfc(x);
            assert!(((erfcx(x) - direct) / direct).abs() < 1e-13, "x={x}");
        }
        let lo = erfcx(2.0 - 1e-12);
        assert!(((lo - erfcx(2.0)) / lo).abs() < 1e-11);
        assert!((erfcx(1e6) * 1e6 * SQRT_PI - 1.0).abs() < 1e-11);
        assert!((expwerfc(0.5, 0.3) - 0.5f64.exp() * libm::erfc(0.3)).abs() < 1e-15);
    }
}

"""Small, regularized calibrators fitted only on earlier forecast outcomes."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def fit_probability(probabilities, outcomes) -> dict:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) < 200 or np.unique(y).size < 2:
        return {"intercept": 0.0, "slope": 1.0, "n": int(len(p))}
    x = logit(p.clip(1e-6, 1 - 1e-6))

    def loss(beta):
        z = beta[0] + beta[1] * x
        penalty = 0.5 * (beta[0] ** 2 + (beta[1] - 1) ** 2)
        return np.sum(np.logaddexp(0, z) - y * z) + penalty

    fit = minimize(loss, [0.0, 1.0], bounds=[(-2, 2), (0, 3)], method="L-BFGS-B")
    if not fit.success:
        raise ValueError(f"Probability calibration failed: {fit.message}")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1]), "n": len(p)}


def probability(probabilities, fitted: dict):
    return expit(
        fitted["intercept"]
        + fitted["slope"] * logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    )


def fit_total(predictions, outcomes) -> dict:
    """A constant residual correction minimizes training absolute error."""
    residuals = np.asarray(outcomes, dtype=float) - np.asarray(predictions, dtype=float)
    if not np.isfinite(residuals).all():
        raise ValueError("Total calibration requires finite training observations")
    return {
        "offset": float(np.median(residuals)) if len(residuals) else 0.0,
        "n": len(residuals),
    }

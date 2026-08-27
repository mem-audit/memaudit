"""Control-calibrated detection stats (Clopper-Pearson, ROC, set-level t-test)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from memaudit.constants import MIN_CONTROLS_FOR_TPR_AT_1PCT


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson CI for a binomial proportion.

    ``k`` successes in ``n`` trials. Returns ``(low, high)`` at level ``1 - alpha``.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(k)
    n = int(n)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is not in [0, n={n}]")
    if k == 0:
        low = 0.0
    else:
        low = float(scipy_stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    if k == n:
        high = 1.0
    else:
        high = float(scipy_stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (low, high)


def roc_points(
    member_scores: np.ndarray | list[float],
    control_scores: np.ndarray | list[float],
    n_grid: int = 64,
) -> list[dict[str, float]]:
    """Log-log-friendly ROC points. Higher score = more member-like."""
    members = np.asarray(member_scores, dtype=np.float64)
    controls = np.asarray(control_scores, dtype=np.float64)
    if members.size == 0 or controls.size == 0:
        return []
    all_scores = np.concatenate([members, controls])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, n_grid)))
    points: list[dict[str, float]] = []
    for t in thresholds:
        tpr = float(np.mean(members >= t))
        fpr = float(np.mean(controls >= t))
        points.append({"threshold": float(t), "tpr": tpr, "fpr": fpr})
    points.sort(key=lambda p: p["fpr"])
    return points


def roc_auc(member_scores: np.ndarray | list[float], control_scores: np.ndarray | list[float]) -> float:
    """Wilcoxon-Mann-Whitney AUC. 1.0 = members always score above controls."""
    pos = np.asarray(member_scores, dtype=np.float64)
    neg = np.asarray(control_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # P(pos > neg) + 0.5 P(pos == neg)
    # vectorized via broadcasting can OOM on huge n; these audits are small
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)))


def tpr_at_fpr(
    member_scores: np.ndarray | list[float],
    control_scores: np.ndarray | list[float],
    fpr: float = 0.01,
) -> dict[str, Any]:
    """TPR at a target FPR, thresholded on held-out controls.

    Threshold ``t`` is the empirical ``(1 - fpr)`` quantile of control scores
    (higher = more member-like). Members with score >= t count as detected.
    """
    members = np.asarray(member_scores, dtype=np.float64)
    controls = np.asarray(control_scores, dtype=np.float64)
    n_m = int(members.size)
    n_c = int(controls.size)
    min_controls = max(1, int(math.ceil(1.0 / float(fpr) - 1e-12))) if fpr > 0 else 1
    if abs(float(fpr) - 0.01) < 1e-12:
        min_controls = max(min_controls, MIN_CONTROLS_FOR_TPR_AT_1PCT)
    if n_c == 0 or n_m == 0:
        return {
            "tpr": float("nan"),
            "threshold": float("nan"),
            "n_detected": 0,
            "n_members": n_m,
            "n_controls": n_c,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "target_fpr": fpr,
            "headline_valid": False,
            "min_controls_required": min_controls,
            "achievable_fpr": None,
            "warning": "need both inserted members and held-out controls; TPR@FPR is unidentified",
        }
    # method='higher' is conservative when n_controls is small
    try:
        threshold = float(np.quantile(controls, 1.0 - fpr, method="higher"))
    except TypeError:  # numpy < 1.22
        threshold = float(np.quantile(controls, 1.0 - fpr, interpolation="higher"))
    detected = members >= threshold
    n_detected = int(np.sum(detected))
    tpr = float(n_detected / n_m)
    ci_low, ci_high = clopper_pearson(n_detected, n_m, alpha=0.05)
    headline_valid = n_c >= min_controls
    achievable_fpr = float(1.0 / n_c)
    warning = None
    if not headline_valid:
        warning = (
            f"TPR@{fpr:.0%} FPR is unidentified with n_controls={n_c} "
            f"(need >={min_controls} held-out canaries so the empirical "
            f"{(1.0 - fpr):.0%} quantile is defined). The number below is "
            f"exploratory only -- do not treat it as TPR at {fpr:.0%} FPR. "
            f"Achievable FPR with this control set is ~{achievable_fpr:.1%}."
        )
    elif n_c < 200:
        warning = (
            f"TPR@{fpr:.0%} FPR with n_controls={n_c} is identified but noisy "
            f"(published audits use >=200-1000 held-out canaries). "
            f"The threshold is the empirical {(1.0 - fpr):.0%} control quantile."
        )
    return {
        "tpr": tpr,
        "threshold": threshold,
        "n_detected": n_detected,
        "n_members": n_m,
        "n_controls": n_c,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "target_fpr": fpr,
        "headline_valid": headline_valid,
        "min_controls_required": min_controls,
        "achievable_fpr": achievable_fpr,
        "warning": warning,
    }


def welch_ttest(
    member_scores: np.ndarray | list[float],
    control_scores: np.ndarray | list[float],
    alternative: str = "greater",
) -> dict[str, Any]:
    """Exploratory set-level gap (Maini-style aggregation). Not a per-record verdict."""
    a = np.asarray(member_scores, dtype=np.float64)
    b = np.asarray(control_scores, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return {
            "mean_members": float(a.mean()) if a.size else float("nan"),
            "mean_controls": float(b.mean()) if b.size else float("nan"),
            "mean_gap": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "df": float("nan"),
            "note": "need >=2 scores on each side",
        }
    result = scipy_stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    return {
        "mean_members": float(a.mean()),
        "mean_controls": float(b.mean()),
        "mean_gap": float(a.mean() - b.mean()),
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": float(result.df) if hasattr(result, "df") else float("nan"),
        "alternative": alternative,
        "note": "exploratory set-level test; per-record flags are not verdicts",
    }


def steinke_eps_lower_bound(
    n_correct: int,
    n_guesses: int,
    *,
    delta: float = 1e-5,
    alpha: float = 0.05,
    m: int | None = None,
) -> dict[str, Any]:
    """Optional one-run epsilon lower bound (Steinke et al. 2023). Not a privacy certificate.

    Inverts the binomial tail: largest epsilon that can be *rejected* at level
    ``alpha``, i.e. P(Binom(r, e^eps/(e^eps+1)) >= v) <= alpha. A small delta
    correction is applied when ``m`` is given. The guess-count ceiling is
    always returned so a small eps_LB cannot be misread as "private".
    """
    r = int(n_guesses)
    v = int(n_correct)
    if r <= 0:
        return {"eps_lb": 0.0, "ceiling": 0.0, "n_correct": v, "n_guesses": r, "note": "no guesses"}
    v = min(max(v, 0), r)

    def tail_ok(eps: float) -> bool:
        p = math.exp(eps) / (math.exp(eps) + 1.0)
        # P(X >= v) = 1 - F(v-1)
        survival = float(scipy_stats.binom.sf(v - 1, r, p)) if v > 0 else 1.0
        return survival <= alpha

    lo, hi = 0.0, 20.0
    if not tail_ok(0.0):
        eps = 0.0
    else:
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if tail_ok(mid):
                lo = mid
            else:
                hi = mid
        eps = lo
    if delta > 0 and m:
        # paper: additive slack ~ O(m * delta / r) - stamp, don't pretend tightness
        eps = max(0.0, eps - (m * delta / max(r, 1)))
    # ceiling: all r guesses correct
    lo_c, hi_c = 0.0, 20.0
    # reuse inversion with v=r
    def tail_all(eps: float) -> bool:
        p = math.exp(eps) / (math.exp(eps) + 1.0)
        return float(scipy_stats.binom.sf(r - 1, r, p)) <= alpha

    if tail_all(0.0):
        for _ in range(48):
            mid = 0.5 * (lo_c + hi_c)
            if tail_all(mid):
                lo_c = mid
            else:
                hi_c = mid
        ceiling = lo_c
    else:
        ceiling = 0.0
    return {
        "eps_lb": float(eps),
        "ceiling": float(ceiling),
        "n_correct": v,
        "n_guesses": r,
        "delta": delta,
        "alpha": alpha,
        "note": (
            "One-run black-box eps_LB is a leakage witness, not a privacy certificate. "
            f"Even {r}/{r} correct guesses cap eps_LB at the printed ceiling."
        ),
    }


def min_k_percent(token_logprobs: np.ndarray | list[float], k_pct: float = 20.0) -> float:
    """Average of the lowest k% token log-probs (higher = more member-like)."""
    x = np.asarray(token_logprobs, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    n = max(1, int(math.ceil(x.size * float(k_pct) / 100.0)))
    n = min(n, x.size)
    return float(np.sort(x)[:n].mean())


def min_k_plus_plus(
    target_logprobs: np.ndarray | list[float],
    mu: np.ndarray | list[float],
    sigma: np.ndarray | list[float],
    k_pct: float = 20.0,
) -> float:
    """Zhang et al. Min-K%++: z-score of the target log-prob, then Min-K%."""
    lp = np.asarray(target_logprobs, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if lp.size == 0:
        return float("nan")
    z = (lp - m) / np.clip(s, 1e-8, None)
    return min_k_percent(z, k_pct)


def masked_mean_nll(token_logprobs: np.ndarray | list[float]) -> float:
    x = np.asarray(token_logprobs, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(-x.mean())

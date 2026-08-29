"""Control-calibrated detection stats (Clopper-Pearson, ROC, set-level t-test)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from memaudit.constants import (
    CALIBRATION_BOOTSTRAP_N,
    MIN_CONTROLS_FOR_TPR_AT_1PCT,
    REPETITION_TIER_MEANING,
)


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


def membership_by_repetition(
    per_canary: list[dict[str, Any]],
    threshold: float,
    *,
    pooled_n_detected: int | None = None,
    pooled_n_members: int | None = None,
    pooled_tpr: float | None = None,
    pooled_ci: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Stress-response curve: TPR at the *pooled* control threshold, by tier.

    1x is a single-exposure probe, 4x moderate stress, 16x high-exposure
    stress, pooled is the powered-audit headline. Detection uses the same
    threshold as the overall TPR@FPR so the tiers decompose that headline.
    """
    by_tier: dict[int, list[float]] = {}
    for row in per_canary:
        if not row.get("included"):
            continue
        score = (row.get("scores") or {}).get("headline_score")
        if score is None or score != score:
            continue
        tier = int(row.get("repetitions") or 1)
        by_tier.setdefault(tier, []).append(float(score))
    out: dict[str, Any] = {}
    thresh_ok = threshold == threshold  # not NaN
    for tier in sorted(by_tier):
        scores = by_tier[tier]
        n = len(scores)
        n_det = int(sum(s >= threshold for s in scores)) if thresh_ok else 0
        tpr = float(n_det / n) if n else float("nan")
        ci_low, ci_high = clopper_pearson(n_det, n) if n else (float("nan"), float("nan"))
        out[str(tier)] = {
            "n": n,
            "detected": n_det,
            "tpr": tpr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "meaning": REPETITION_TIER_MEANING.get(tier, "repetition-tier probe"),
        }
    n_p = pooled_n_members
    d_p = pooled_n_detected
    if n_p is None:
        n_p = sum(b["n"] for b in out.values())
    if d_p is None:
        d_p = sum(b["detected"] for b in out.values())
    tpr_p = pooled_tpr if pooled_tpr is not None else (float(d_p / n_p) if n_p else float("nan"))
    if pooled_ci is not None:
        ci_l, ci_h = pooled_ci
    elif n_p:
        ci_l, ci_h = clopper_pearson(int(d_p), int(n_p))
    else:
        ci_l, ci_h = float("nan"), float("nan")
    out["pooled"] = {
        "n": int(n_p),
        "detected": int(d_p),
        "tpr": tpr_p,
        "ci_low": ci_l,
        "ci_high": ci_h,
        "meaning": REPETITION_TIER_MEANING["pooled"],
    }
    out["note"] = (
        "Tiers share the pooled control-calibrated threshold. "
        "1x ≈ single-exposure probe; 4x ≈ moderate stress; 16x ≈ high-exposure "
        "stress; pooled is the powered-audit headline. A pooled TPR can be "
        "substantially a duplication/exposure stress signal rather than a "
        "single-exposure detection probability."
    )
    return out


def bootstrap_calibration_stability(
    member_scores: np.ndarray | list[float],
    control_scores: np.ndarray | list[float],
    fpr: float = 0.01,
    n_bootstrap: int = CALIBRATION_BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """How much the FPR threshold and resulting TPR move under control resampling.

    Separate from the member-side Clopper-Pearson CI: this is threshold /
    calibration uncertainty. Cheap enough for a single run (default 50 draws).
    """
    members = np.asarray(member_scores, dtype=np.float64)
    controls = np.asarray(control_scores, dtype=np.float64)
    n_m = int(members.size)
    n_c = int(controls.size)
    base = {
        "kind": "control_resample_threshold",
        "n_bootstrap": 0,
        "target_fpr": float(fpr),
        "note": (
            "Bootstrap-resample held-out canary controls, recompute the "
            f"{fpr:.0%} FPR threshold and the resulting TPR. This is "
            "calibration stability, not the member-side Clopper-Pearson CI."
        ),
    }
    if n_m < 1 or n_c < 2 or n_bootstrap < 1:
        base["note"] = (
            "need inserted members and >=2 controls to resample the threshold; "
            "calibration stability was not computed"
        )
        return base
    rng = np.random.default_rng(int(seed))
    thresholds: list[float] = []
    tprs: list[float] = []
    for _ in range(int(n_bootstrap)):
        boot = rng.choice(controls, size=n_c, replace=True)
        det = tpr_at_fpr(members, boot, fpr=fpr)
        if det["threshold"] == det["threshold"]:
            thresholds.append(float(det["threshold"]))
        if det["tpr"] == det["tpr"]:
            tprs.append(float(det["tpr"]))
    if not thresholds or not tprs:
        return base
    t_arr = np.asarray(thresholds, dtype=np.float64)
    p_arr = np.asarray(tprs, dtype=np.float64)
    base["n_bootstrap"] = int(n_bootstrap)
    base["threshold"] = {
        "mean": float(t_arr.mean()),
        "std": float(t_arr.std()),
        "min": float(t_arr.min()),
        "max": float(t_arr.max()),
        "p05": float(np.quantile(t_arr, 0.05)),
        "p95": float(np.quantile(t_arr, 0.95)),
    }
    base["tpr"] = {
        "mean": float(p_arr.mean()),
        "std": float(p_arr.std()),
        "min": float(p_arr.min()),
        "max": float(p_arr.max()),
        "p05": float(np.quantile(p_arr, 0.05)),
        "p95": float(np.quantile(p_arr, 0.95)),
    }
    return base


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

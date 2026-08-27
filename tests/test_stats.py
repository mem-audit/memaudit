from __future__ import annotations

import numpy as np
import pytest

from memaudit.stats import (
    clopper_pearson,
    min_k_percent,
    min_k_plus_plus,
    roc_auc,
    steinke_eps_lower_bound,
    tpr_at_fpr,
    welch_ttest,
)


def test_clopper_pearson_known_values():
    # 40/100 at 95%: roughly [0.303, 0.503]
    lo, hi = clopper_pearson(40, 100, alpha=0.05)
    assert 0.29 < lo < 0.32
    assert 0.49 < hi < 0.52
    # 0/n and n/n hit the boundaries
    assert clopper_pearson(0, 20)[0] == 0.0
    assert clopper_pearson(20, 20)[1] == 1.0


def test_tpr_at_fpr_separable():
    members = [3.0, 2.5, 2.0, 1.8]
    controls = [0.1, 0.0, -0.2, -1.0]
    out = tpr_at_fpr(members, controls, fpr=0.01)
    assert out["tpr"] == 1.0
    assert out["n_detected"] == 4
    assert 0.0 <= out["ci_low"] <= out["tpr"] <= out["ci_high"] <= 1.0
    assert out["headline_valid"] is False
    assert out["warning"]


def test_tpr_at_fpr_none_detected():
    members = [0.0, -1.0]
    controls = [5.0, 4.0, 3.0, 2.0]
    out = tpr_at_fpr(members, controls, fpr=0.01)
    assert out["tpr"] == 0.0


def test_auc_perfect_and_chance():
    assert roc_auc([2, 3, 4], [0, 1]) == pytest.approx(1.0)
    # identical distributions ? ~0.5
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    assert 0.35 < roc_auc(x[:40], x[40:]) < 0.65


def test_welch_ttest_direction():
    out = welch_ttest([2.0, 2.1, 1.9, 2.2], [0.0, 0.1, -0.1, 0.05])
    assert out["mean_gap"] > 0
    assert out["p_value"] < 0.05


def test_min_k_and_plusplus_math():
    lp = np.array([-1.0, -0.2, -4.0, -0.3])
    # lowest 25% of 4 tokens ? 1 token: -4.0
    assert min_k_percent(lp, 25.0) == pytest.approx(-4.0)
    mu = np.zeros(4)
    sigma = np.ones(4)
    # z == lp, so same min-k
    assert min_k_plus_plus(lp, mu, sigma, 25.0) == pytest.approx(-4.0)
    # shifting mu should change z
    assert min_k_plus_plus(lp, np.full(4, -4.0), sigma, 25.0) != pytest.approx(-4.0)


def test_steinke_ceiling_printed():
    out = steinke_eps_lower_bound(30, 30, alpha=0.05)
    assert out["ceiling"] >= out["eps_lb"]
    assert out["ceiling"] > 1.5  # ~2.25 for 30/30 at 95%
    assert "certificate" in out["note"]
    # Steinke Appendix D: 75/100 correct, no abstentions, ~0.70 at 95%
    worked = steinke_eps_lower_bound(75, 100, alpha=0.05)
    assert 0.55 < worked["eps_lb"] < 0.85

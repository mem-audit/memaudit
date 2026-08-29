"""WS5: TokenSignals, default Min-K%++ golden, swappable scorer, report fields."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from memaudit.audit import run_audit
from memaudit.canaries import generate_canaries
from memaudit.exceptions import MemauditConfigError
from memaudit.injection import inject
from memaudit.scorers import (
    DEFAULT_SCORER_NAME,
    DEFAULT_SCORER_VERSION,
    MinKPlusPlusScorer,
    SignalsCache,
    TokenSignals,
    resolve_scorer,
    scorer_provenance,
)
from memaudit.scoring import (
    combine_ft_ref,
    extract_token_signals,
    scores_from_token_stats,
    token_logprob_stats,
    token_signals_from_logits,
)
from memaudit.stats import min_k_plus_plus


class DummyMeanGoldScorer:
    name = "dummy_mean_gold"
    version = "test.1"
    requires_reference = False
    forward_passes_per_record = 1

    def score(self, target: TokenSignals, reference: TokenSignals | None) -> float:
        if target.gold_logprob.size == 0:
            return float("nan")
        return float(target.gold_logprob.mean())


def _tiny_manifest(tokenizer, n=2, n_controls=3, seed=0):
    host = [{"text": f"ordinary training sentence number {i} about weather"} for i in range(12)]
    cans = generate_canaries(
        tokenizer,
        n=n,
        n_controls=n_controls,
        family="random",
        seed=seed,
        secret_len=25,
    )
    return inject(host, cans, fmt="text", seed=seed, include_prob=1.0)


def test_default_scorer_matches_legacy_minkpp_fixture():
    rng = np.random.default_rng(0)
    t = 16
    gold = rng.normal(-2.0, 0.5, t)
    mu = rng.normal(-3.0, 0.3, t)
    sigma = np.abs(rng.normal(1.0, 0.1, t)) + 0.05
    correct = rng.random(t) > 0.4
    target = TokenSignals(gold, mu, sigma, correct)
    ref_gold = gold - 0.4
    reference = TokenSignals(ref_gold, mu - 0.1, sigma, correct)

    scorer = MinKPlusPlusScorer(min_k_pct=20.0)
    expected_ft = min_k_plus_plus(gold, mu, sigma, 20.0)
    expected_ref = min_k_plus_plus(ref_gold, mu - 0.1, sigma, 20.0)
    assert scorer.score(target, None) == pytest.approx(expected_ft)
    assert scorer.score(target, reference) == pytest.approx(expected_ft - expected_ref)

    ft_dict = scores_from_token_stats(gold, mu, sigma, 20.0)
    ref_dict = scores_from_token_stats(ref_gold, mu - 0.1, sigma, 20.0)
    assert scorer.score(target, reference) == pytest.approx(
        combine_ft_ref(ft_dict, ref_dict)["headline_score"]
    )
    assert scorer.score(target, None) == pytest.approx(
        combine_ft_ref(ft_dict, None)["headline_score"]
    )


def test_argmax_correct_signal_from_logits():
    vocab = 8
    logits = torch.zeros(3, vocab)
    logits[0, 2] = 10.0
    logits[1, 0] = 10.0
    logits[2, 4] = 10.0
    targets = torch.tensor([2, 1, 4])
    sig = token_signals_from_logits(logits, targets)
    assert sig.argmax_correct.tolist() == [True, False, True]
    lp, mu, sigma = token_logprob_stats(logits, targets)
    np.testing.assert_allclose(sig.gold_logprob, lp)
    np.testing.assert_allclose(sig.mu, mu)
    np.testing.assert_allclose(sig.sigma, sigma)
    assert sig.n_scored_tokens == 3


def test_signals_cache_respects_tag(tiny_model):
    ids = list(range(4, 20))
    cache = SignalsCache()
    a = extract_token_signals(tiny_model.eval(), ids, span=(4, 16), cache=cache, cache_tag="target")
    b = extract_token_signals(tiny_model.eval(), ids, span=(4, 16), cache=cache, cache_tag="target")
    assert a is b
    other = TokenSignals.empty()
    cache.put(tiny_model, ids, (4, 16), True, other, cache_tag="disable_adapter")
    tagged = extract_token_signals(
        tiny_model.eval(), ids, span=(4, 16), cache=cache, cache_tag="disable_adapter"
    )
    assert tagged is other
    assert tagged is not a


def test_resolve_scorer_builtin_and_import_path():
    builtin = resolve_scorer(None)
    assert isinstance(builtin, MinKPlusPlusScorer)
    assert builtin.name == DEFAULT_SCORER_NAME
    assert resolve_scorer("min_k_plus_plus").name == DEFAULT_SCORER_NAME
    assert resolve_scorer("base_calibrated_min_k_plus_plus").name == DEFAULT_SCORER_NAME

    mod = types.ModuleType("dummy_mia_scorer")
    mod.DummyMeanGoldScorer = DummyMeanGoldScorer
    import sys

    sys.modules["dummy_mia_scorer"] = mod
    loaded = resolve_scorer("dummy_mia_scorer:DummyMeanGoldScorer")
    assert loaded.name == "dummy_mean_gold"
    assert loaded.version == "test.1"
    with pytest.raises(MemauditConfigError, match="Unknown membership scorer"):
        resolve_scorer("not_a_real_attack")


def test_run_audit_records_default_scorer(tokenizer, tiny_model):
    ds, manifest = _tiny_manifest(tokenizer)
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        skip_generation=True,
    )
    scorer = report["membership"]["scorer"]
    assert scorer["name"] == DEFAULT_SCORER_NAME
    assert scorer["version"] == DEFAULT_SCORER_VERSION
    assert scorer["requires_reference"] is False
    assert report["membership"]["headline_attack"] == "min_k_plus_plus"
    assert report["provenance"]["resolved_config"]["scorer"] == DEFAULT_SCORER_NAME
    assert report["provenance"]["resolved_config"]["scorer_version"] == DEFAULT_SCORER_VERSION


def test_dummy_scorer_via_config_appears_in_report(tokenizer, tiny_model):
    ds, manifest = _tiny_manifest(tokenizer)
    import sys

    sys.modules["dummy_mia_scorer"] = types.ModuleType("dummy_mia_scorer")
    sys.modules["dummy_mia_scorer"].DummyMeanGoldScorer = DummyMeanGoldScorer
    report = run_audit(
        tiny_model,
        tokenizer,
        manifest,
        dataset=ds,
        ref="none",
        real_sample=0,
        skip_generation=True,
        scorer="dummy_mia_scorer:DummyMeanGoldScorer",
    )
    scorer = report["membership"]["scorer"]
    assert scorer["name"] == "dummy_mean_gold"
    assert scorer["version"] == "test.1"
    assert report["membership"]["headline_attack"] == "dummy_mean_gold"
    # headline equals mean gold logprob, not Min-K%++
    row = report["per_canary"][0]["scores"]
    assert row["headline_score"] == pytest.approx(row["mean_logprob"])
    assert row["headline_score"] != pytest.approx(row["min_k_plus_plus"])


def test_cli_exposes_scorer_flag():
    from memaudit.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "audit",
            "--model",
            "./out",
            "--canary-set",
            "./m.json",
            "--scorer",
            "dummy_mia_scorer:DummyMeanGoldScorer",
        ]
    )
    assert ns.scorer == "dummy_mia_scorer:DummyMeanGoldScorer"


def test_scorer_provenance_shape():
    block = scorer_provenance(MinKPlusPlusScorer())
    assert block == {
        "name": DEFAULT_SCORER_NAME,
        "version": DEFAULT_SCORER_VERSION,
        "requires_reference": False,
        "forward_passes_per_record": 1,
    }

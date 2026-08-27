from __future__ import annotations

import numpy as np
import pytest
import torch

from memaudit.scoring import (
    combine_ft_ref,
    regurgitation_flags,
    score_sequence,
    secret_target_positions,
    sentence_bleu,
    sliding_window_ned,
    token_logprob_stats,
)
from memaudit.stats import masked_mean_nll


def test_sentence_bleu_exact_and_mismatch():
    s = "the cat sat on the mat"
    assert sentence_bleu(s, s) == pytest.approx(1.0, abs=1e-6)
    assert sentence_bleu(s, "completely different words here") < 0.3


def test_ned_exact_and_window():
    secret = "abcdef"
    assert sliding_window_ned(secret, secret) == 0.0
    assert sliding_window_ned(secret, "xxxabcdefyyy") == 0.0
    assert sliding_window_ned(secret, "zzzzzz") > 0.5


def test_regurgitation_flags_thresholds():
    secret = "alpha bravo charlie delta echo"
    exact = regurgitation_flags(secret, secret)
    assert exact["exact"] and exact["regurgitated"]
    approx = regurgitation_flags(secret, "alpha bravo charlie delta echo extra")
    assert approx["bleu"] > 0.7
    far = regurgitation_flags(secret, "zzzz yyyy xxxx")
    assert not far["regurgitated"]


def test_secret_target_positions_skips_first_record_token():
    # secret at [0, 5) on a length-5 record: score tokens 1..4 (predicted by logits 0..3)
    pos = secret_target_positions(5, (0, 5), skip_first_record_token=True)
    assert pos == [1, 2, 3, 4]
    # secret in the completion after a 3-token prompt
    pos2 = secret_target_positions(10, (3, 8), skip_first_record_token=True)
    assert pos2 == [3, 4, 5, 6, 7]


def test_token_logprob_stats_and_masked_nll():
    # one-hot-ish: position 0 strongly predicts token 1; position 1 is uniform
    vocab = 8
    logits = torch.zeros(2, vocab)
    logits[0, 1] = 10.0
    targets = torch.tensor([1, 3])
    lp, mu, sigma = token_logprob_stats(logits, targets)
    assert lp.shape == (2,)
    assert lp[0] > lp[1]  # confident correct token
    assert np.all(sigma >= 0)
    nll = masked_mean_nll(lp)
    assert nll < 5.0


def test_combine_ft_ref_orientation():
    ft = {"masked_nll": 1.0, "min_k": -0.5, "min_k_plus_plus": -0.2, "mean_logprob": -1.0, "n_scored_tokens": 4}
    ref = {"masked_nll": 3.0, "min_k": -2.0, "min_k_plus_plus": -1.5, "mean_logprob": -3.0, "n_scored_tokens": 4}
    out = combine_ft_ref(ft, ref)
    # members: FT more confident than base ? positive headline
    assert out["base_calibrated_min_k_plus_plus"] > 0
    assert out["headline_attack_used"] == "base_calibrated_min_k_plus_plus"
    none = combine_ft_ref(ft, None)
    assert none["headline_attack_used"] == "min_k_plus_plus"


def test_score_sequence_secret_span_not_full(tiny_model, tokenizer):
    torch.manual_seed(0)
    ids = list(range(4, 36))
    full = score_sequence(tiny_model.eval(), ids, span=(0, len(ids)))
    secret = score_sequence(tiny_model.eval(), ids, span=(20, 32))
    assert full["n_scored_tokens"] > secret["n_scored_tokens"]
    assert secret["n_scored_tokens"] == 12  # positions 20..31
    assert full["masked_nll"] == full["masked_nll"]  # not NaN

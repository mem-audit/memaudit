from __future__ import annotations

import pytest

from memaudit.canaries import generate_canaries, normalize_family
from memaudit.constants import NEW_TOKEN_MESSAGE
from memaudit.exceptions import MemauditConfigError


def test_normalize_aliases():
    assert normalize_family("high-ppl") == "high_ppl"
    assert normalize_family("random-secret") == "random"


def test_new_token_gated(tokenizer):
    with pytest.raises(MemauditConfigError, match="never resizes"):
        generate_canaries(tokenizer, n=2, n_controls=2, family="new_token")
    assert "frozen-embedding" in NEW_TOKEN_MESSAGE


def test_unknown_family(tokenizer):
    with pytest.raises(MemauditConfigError, match="Unknown"):
        generate_canaries(tokenizer, n=1, n_controls=1, family="metagradient")


def test_high_ppl_fallback_without_model(tokenizer):
    cans = generate_canaries(tokenizer, n=4, n_controls=4, family="high_ppl", seed=0, secret_len=25)
    assert len(cans) == 8
    assert sum(c.role == "candidate" for c in cans) == 4
    assert sum(c.role == "control" for c in cans) == 4
    for c in cans:
        assert c.secret
        assert len(c.secret_token_ids) >= 8
        assert "no model provided" in c.generation_notes
        assert c.family == "high_ppl"
        # existing vocab only
        assert all(0 <= i < tokenizer.vocab_size for i in c.secret_token_ids)


def test_seed_reproducible(tokenizer):
    a = generate_canaries(tokenizer, n=3, n_controls=3, family="random", seed=7, secret_len=25)
    b = generate_canaries(tokenizer, n=3, n_controls=3, family="random", seed=7, secret_len=25)
    assert [c.secret_token_ids for c in a] == [c.secret_token_ids for c in b]


def test_families_and_diversity(tokenizer):
    for family in ("random", "unigram", "bigram", "structured"):
        cans = generate_canaries(tokenizer, n=3, n_controls=3, family=family, seed=1, secret_len=25)
        secrets = [c.secret for c in cans]
        assert len(set(secrets)) == len(secrets)
        if family == "structured":
            assert any("CANARY-ID" in c.secret or "t" in c.secret for c in cans)


def test_unigram_uses_corpus(tokenizer):
    # make token 20 extremely common; rare tokens should be preferred
    corpus = ["t20 t20 t20 t20 t20"] * 10
    cans = generate_canaries(
        tokenizer, n=2, n_controls=2, family="unigram", seed=0, secret_len=25, corpus=corpus
    )
    assert "corpus unigram" in cans[0].generation_notes


def test_repetition_grid(tokenizer):
    cans = generate_canaries(
        tokenizer, n=6, n_controls=2, family="random", repetitions=(1, 4, 16), seed=0, secret_len=25
    )
    reps = [c.repetitions for c in cans if c.role == "candidate"]
    assert set(reps) == {1, 4, 16}


def test_rejects_short_secret(tokenizer):
    with pytest.raises(MemauditConfigError):
        generate_canaries(tokenizer, n=1, n_controls=1, secret_len=8)

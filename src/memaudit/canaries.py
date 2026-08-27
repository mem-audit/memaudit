"""Pure canary generation. No Hugging Face Trainer dependency.

Default family is high-perplexity regular-token sequences (existing vocab only).
The new-token family is gated and unimplemented - memaudit never resizes vocab.
"""

from __future__ import annotations

import math
import string
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from memaudit.constants import (
    DEFAULT_FAMILY,
    DEFAULT_N,
    DEFAULT_N_CONTROLS,
    DEFAULT_REPETITIONS,
    DEFAULT_SECRET_LEN,
    FAMILY_ALIASES,
    IMPLEMENTED_FAMILIES,
    MAX_SECRET_LEN,
    MIN_CONTROLS_FOR_TPR_AT_1PCT,
    MIN_SECRET_LEN,
    NEW_TOKEN_MESSAGE,
)
from memaudit.exceptions import MemauditConfigError
from memaudit.types import Canary
from memaudit.utils import (
    decode_ids,
    encode_ids,
    roundtrip_tokens,
    usable_token_ids,
)

_ALPHANUM = string.ascii_uppercase + string.digits


def normalize_family(family: str) -> str:
    key = (family or DEFAULT_FAMILY).strip().lower()
    return FAMILY_ALIASES.get(key, key)


def generate_canaries(
    tokenizer: Any,
    n: int = DEFAULT_N,
    n_controls: int = DEFAULT_N_CONTROLS,
    family: str = DEFAULT_FAMILY,
    repetitions: Sequence[int] = DEFAULT_REPETITIONS,
    seed: int = 0,
    *,
    model: Any | None = None,
    secret_len: int = DEFAULT_SECRET_LEN,
    corpus: Sequence[str] | None = None,
    ppl_band: tuple[float, float] = (40.0, 500.0),
    sample_temperature: float = 1.8,
) -> list[Canary]:
    """Generate ``n`` insert-eligible canaries plus ``n_controls`` never-inserted twins.

    Parameters
    ----------
    tokenizer
        Any object with ``encode`` / ``decode`` / ``vocab_size`` (HF tokenizers work).
    n
        Insert-eligible canaries. ``inject()`` applies Bernoulli inclusion coins.
    n_controls
        Dedicated held-out negative controls (never inserted).
    family
        ``high_ppl`` (default), ``unigram``, ``bigram``, ``structured``, ``random``.
        ``new_token`` raises - gated / unimplemented in v0.1.
    model
        Optional causal LM used only by ``high_ppl`` rejection sampling. If omitted,
        ``high_ppl`` falls back to rare-token unigram construction from the vocab
        (recorded in ``generation_notes``).
    corpus
        Optional raw strings for unigram/bigram frequency tables.
    """
    family_norm = normalize_family(family)
    if family_norm == "new_token":
        raise MemauditConfigError(NEW_TOKEN_MESSAGE)
    if family_norm not in IMPLEMENTED_FAMILIES:
        raise MemauditConfigError(
            f"Unknown canary family {family!r}. Implemented: {sorted(IMPLEMENTED_FAMILIES)}"
        )
    if n < 0 or n_controls < 0:
        raise MemauditConfigError("n and n_controls must be >= 0")
    if n + n_controls == 0:
        raise MemauditConfigError("need at least one canary or control")
    if n_controls < MIN_CONTROLS_FOR_TPR_AT_1PCT:
        warnings.warn(
            f"n_controls={n_controls} is below the {MIN_CONTROLS_FOR_TPR_AT_1PCT} "
            "floor required to identify TPR@1% FPR. The report will refuse that "
            "headline rather than invent a precise detection rate. Pass "
            f"n_controls>={MIN_CONTROLS_FOR_TPR_AT_1PCT} for a valid headline.",
            UserWarning,
            stacklevel=2,
        )
    secret_len = int(secret_len)
    if secret_len < MIN_SECRET_LEN or secret_len > MAX_SECRET_LEN:
        raise MemauditConfigError(
            f"secret_len must be in [{MIN_SECRET_LEN}, {MAX_SECRET_LEN}] (got {secret_len})"
        )
    reps = tuple(int(r) for r in repetitions) or DEFAULT_REPETITIONS
    if any(r < 1 for r in reps):
        raise MemauditConfigError("repetitions must be positive integers")

    rng = np.random.default_rng(int(seed))
    usable = usable_token_ids(tokenizer)
    freq = _corpus_unigrams(tokenizer, corpus) if corpus else None
    bigrams = _corpus_bigrams(tokenizer, corpus) if corpus and family_norm == "bigram" else None

    fallback_note = None
    if family_norm == "high_ppl" and model is None:
        fallback_note = (
            "high_ppl: no model provided; fell back to rare-token unigram construction "
            "from the tokenizer vocabulary (existing tokens only; no vocab resize)"
        )
    if family_norm in {"unigram", "bigram"} and freq is None:
        extra = (
            f"{family_norm}: no corpus provided; sampling uniformly from non-special vocab"
        )
        fallback_note = f"{fallback_note}; {extra}" if fallback_note else extra

    canaries: list[Canary] = []
    seen: list[set[int]] = []
    total = n + n_controls
    for i in range(total):
        role = "candidate" if i < n else "control"
        rep = int(reps[i % len(reps)]) if role == "candidate" else 1
        ids, notes, meta = _draw_secret(
            family=family_norm,
            tokenizer=tokenizer,
            usable=usable,
            rng=rng,
            secret_len=secret_len,
            model=model,
            freq=freq,
            bigrams=bigrams,
            ppl_band=ppl_band,
            sample_temperature=sample_temperature,
            seen=seen,
        )
        if fallback_note:
            notes = f"{fallback_note}. {notes}".strip()
        text, ids = _ensure_roundtrip(tokenizer, ids, usable, rng, secret_len)
        if not text.strip() or len(ids) < MIN_SECRET_LEN // 2:
            # last-resort random draw
            ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
            text, ids = _ensure_roundtrip(tokenizer, ids, usable, rng, secret_len)
        half = max(1, len(ids) // 2)
        prefix_ids = ids[:half]
        prefix = decode_ids(tokenizer, prefix_ids, skip_special_tokens=True)
        seen.append(set(ids))
        canaries.append(
            Canary(
                id=f"c-{family_norm}-{i:04d}",
                family=family_norm,
                secret=text,
                secret_token_ids=ids,
                prefix=prefix,
                prefix_token_ids=prefix_ids,
                secret_span=[0, len(ids)],
                repetitions=rep,
                role=role,
                generation_notes=notes,
                metadata=meta,
            )
        )
    return canaries


def _draw_secret(
    *,
    family: str,
    tokenizer: Any,
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    model: Any | None,
    freq: Counter[int] | None,
    bigrams: Counter[tuple[int, int]] | None,
    ppl_band: tuple[float, float],
    sample_temperature: float,
    seen: list[set[int]],
) -> tuple[list[int], str, dict[str, Any]]:
    for attempt in range(24):
        if family == "high_ppl" and model is not None:
            ids, notes, meta = _high_ppl_from_model(
                model, tokenizer, usable, rng, secret_len, ppl_band, sample_temperature
            )
        elif family == "structured":
            ids, notes, meta = _structured(tokenizer, rng, secret_len)
        elif family == "bigram":
            ids, notes, meta = _bigram_secret(usable, rng, secret_len, bigrams)
        elif family in {"unigram", "high_ppl"}:
            ids, notes, meta = _unigram_secret(usable, rng, secret_len, freq)
            if family == "high_ppl":
                notes = f"unigram fallback for high_ppl. {notes}"
        else:
            ids, notes, meta = _random_secret(usable, rng, secret_len)
        if _diverse_enough(set(ids), seen) or attempt == 23:
            return ids, notes, meta
    raise RuntimeError("unreachable")


def _diverse_enough(candidate: set[int], seen: list[set[int]], max_jaccard: float = 0.55) -> bool:
    if not candidate or not seen:
        return True
    for other in seen:
        inter = len(candidate & other)
        union = len(candidate | other) or 1
        if inter / union > max_jaccard:
            return False
    return True


def _ensure_roundtrip(
    tokenizer: Any,
    ids: list[int],
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
) -> tuple[str, list[int]]:
    text, back = roundtrip_tokens(tokenizer, ids)
    if len(back) >= MIN_SECRET_LEN // 2 and text.strip():
        return text, back
    # pad with extra random tokens and try again
    extra = [int(x) for x in rng.choice(usable, size=max(4, secret_len - len(ids)), replace=True)]
    text, back = roundtrip_tokens(tokenizer, ids + extra)
    return text, back


def _random_secret(
    usable: list[int], rng: np.random.Generator, secret_len: int
) -> tuple[list[int], str, dict[str, Any]]:
    ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
    return ids, "uniform draw from non-special vocab", {"source": "uniform_vocab"}


def _unigram_secret(
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    freq: Counter[int] | None,
) -> tuple[list[int], str, dict[str, Any]]:
    if freq is None:
        ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
        return ids, "uniform-from-vocab (no corpus frequencies)", {"source": "uniform_vocab"}
    # least-frequent usable tokens; tokens unseen in corpus count as 0
    ranked = sorted(usable, key=lambda t: (freq.get(t, 0), t))
    pool = ranked[: max(secret_len * 4, 64)]
    ids = [int(x) for x in rng.choice(np.array(pool, dtype=np.int64), size=secret_len, replace=True)]
    return ids, "least-frequent tokens under corpus unigram counts", {"source": "corpus_unigram"}


def _bigram_secret(
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    bigrams: Counter[tuple[int, int]] | None,
) -> tuple[list[int], str, dict[str, Any]]:
    if not bigrams:
        ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
        return ids, "uniform-from-vocab (no corpus bigrams)", {"source": "uniform_vocab"}
    usable_set = set(usable)
    rare_pairs = [p for p, c in bigrams.most_common()[::-1] if p[0] in usable_set and p[1] in usable_set]
    if not rare_pairs:
        ids = [int(x) for x in rng.choice(usable, size=secret_len, replace=True)]
        return ids, "uniform-from-vocab (empty rare-bigram pool)", {"source": "uniform_vocab"}
    # take from the rarest tail
    tail = rare_pairs[: max(64, secret_len)]
    ids: list[int] = []
    while len(ids) < secret_len:
        a, b = tail[int(rng.integers(0, len(tail)))]
        if not ids:
            ids.extend([int(a), int(b)])
        else:
            ids.append(int(b))
    return ids[:secret_len], "improbable tokens under corpus bigram counts", {"source": "corpus_bigram"}


def _structured(
    tokenizer: Any, rng: np.random.Generator, secret_len: int
) -> tuple[list[int], str, dict[str, Any]]:
    chunks: list[str] = []
    ids: list[int] = []
    fills: list[str] = []
    guard = 0
    while len(ids) < secret_len and guard < 64:
        a = int(rng.integers(100_000, 999_999))
        b = int(rng.integers(1000, 9999))
        c = "".join(str(_ALPHANUM[int(rng.integers(0, len(_ALPHANUM)))]) for _ in range(10))
        fills.append(f"{a}-{b}-{c}")
        chunks.append(f"CANARY-ID:{a}-{b}-{c} ")
        ids = encode_ids(tokenizer, "".join(chunks), add_special_tokens=False)
        guard += 1
    text = decode_ids(tokenizer, ids[:secret_len], skip_special_tokens=True)
    ids = encode_ids(tokenizer, text, add_special_tokens=False)
    return ids, "structured template + random fill (exposure-metric ready)", {"fills": fills, "source": "structured"}


def _high_ppl_from_model(
    model: Any,
    tokenizer: Any,
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    ppl_band: tuple[float, float],
    temperature: float,
) -> tuple[list[int], str, dict[str, Any]]:
    import torch

    from memaudit.utils import infer_device

    device = infer_device(model)
    lo, hi = ppl_band
    best_ids: list[int] | None = None
    best_ppl = math.inf
    accepted = False
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(16):
                ids = _sample_ids(model, tokenizer, usable, rng, secret_len, temperature, device)
                ppl = _teacher_forced_ppl(model, ids, device)
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_ids = ids
                if lo <= ppl <= hi:
                    accepted = True
                    best_ids = ids
                    best_ppl = ppl
                    break
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    if best_ids is None:
        best_ids, _, _ = _unigram_secret(usable, rng, secret_len, None)
        return best_ids, "high_ppl sampling failed; fell back to unigram", {"ppl": None}
    note = (
        f"rejection-sampled from the provided model at T={temperature} "
        f"into PPL band {ppl_band}; {'accepted' if accepted else 'kept closest'} ppl={best_ppl:.2f}"
    )
    return best_ids, note, {"ppl": best_ppl, "accepted_band": accepted, "source": "model_sample"}


def _sample_ids(
    model: Any,
    tokenizer: Any,
    usable: list[int],
    rng: np.random.Generator,
    secret_len: int,
    temperature: float,
    device: Any,
) -> list[int]:
    import torch

    start = int(rng.choice(usable))
    ids = [start]
    usable_set = set(usable)
    for _ in range(secret_len):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(input_ids=inp)
        logits = _extract_logits(out)[0, -1]
        if temperature <= 0:
            nxt = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / float(temperature), dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        if nxt not in usable_set:
            nxt = int(rng.choice(usable))
        ids.append(nxt)
    return ids[1 : secret_len + 1]


def _teacher_forced_ppl(model: Any, ids: list[int], device: Any) -> float:
    import torch
    import torch.nn.functional as F

    if len(ids) < 2:
        return float("inf")
    t = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=t, labels=t)
    loss = getattr(out, "loss", None)
    if loss is not None:
        return float(torch.exp(loss).item())
    logits = _extract_logits(out)[0, :-1]
    targets = t[0, 1:]
    nll = F.cross_entropy(logits, targets)
    return float(torch.exp(nll).item())


def _extract_logits(out: Any) -> Any:
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    raise MemauditConfigError("model forward() did not return logits")


def _corpus_unigrams(tokenizer: Any, corpus: Sequence[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for text in corpus:
        counts.update(encode_ids(tokenizer, str(text), add_special_tokens=False))
    return counts


def _corpus_bigrams(tokenizer: Any, corpus: Sequence[str]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for text in corpus:
        ids = encode_ids(tokenizer, str(text), add_special_tokens=False)
        counts.update(zip(ids, ids[1:]))
    return counts

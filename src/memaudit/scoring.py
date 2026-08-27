"""Secret-span likelihood scores + prefix-prompted generation metrics.

All membership scores are oriented so that **higher = more member-like**.
Logits are reduced immediately (chunked log-prob gathering); full-vocab
tensors are never retained across sequences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from memaudit.constants import BLEU_THRESHOLD, DEFAULT_MIN_K_PCT, NED_THRESHOLD
from memaudit.stats import masked_mean_nll, min_k_percent, min_k_plus_plus
from memaudit.utils import decode_ids, encode_ids, find_subsequence, infer_device

# ---------------------------------------------------------------------------
# Generation-side metrics (no extra deps)
# ---------------------------------------------------------------------------


def ngram_counts(tokens: Sequence[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i : i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Corpus-free sentence BLEU with uniform weights and brevity penalty."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref or not hyp:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_c = ngram_counts(ref, n)
        hyp_c = ngram_counts(hyp, n)
        if not hyp_c:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, ref_c.get(g, 0)) for g, c in hyp_c.items())
        precisions.append(overlap / max(sum(hyp_c.values()), 1))
    if any(p == 0.0 for p in precisions):
        # smoothing: add-1 on zero n-grams so a single miss doesn't zero BLEU
        precisions = [p if p > 0 else 1e-9 for p in precisions]
    logp = sum(math.log(p) for p in precisions) / len(precisions)
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(len(hyp), 1))
    return float(bp * math.exp(logp))


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # two-row DP
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def sliding_window_ned(secret: str, generation: str) -> float:
    """Min normalized edit distance of ``secret`` against windows of ``generation``."""
    if not secret:
        return 0.0
    if not generation:
        return 1.0
    if len(generation) <= len(secret):
        return levenshtein(secret, generation) / max(len(secret), 1)
    best = len(secret)
    width = len(secret)
    # stride so this stays cheap on long gens
    step = max(1, width // 8)
    for i in range(0, len(generation) - width + 1, step):
        d = levenshtein(secret, generation[i : i + width])
        if d < best:
            best = d
            if best == 0:
                break
    return best / max(len(secret), 1)


def regurgitation_flags(secret: str, generation: str) -> dict[str, Any]:
    exact = secret.strip() == generation.strip() or secret.strip() in generation
    bleu = sentence_bleu(secret, generation)
    ned = sliding_window_ned(secret, generation)
    return {
        "exact": bool(exact),
        "bleu": float(bleu),
        "ned": float(ned),
        "approx_bleu": bool(bleu > BLEU_THRESHOLD),
        "approx_ned": bool(ned <= NED_THRESHOLD),
        "regurgitated": bool(exact or bleu > BLEU_THRESHOLD or ned <= NED_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# Likelihood math on a single sequence
# ---------------------------------------------------------------------------


def token_logprob_stats(logits, target_ids):
    """Return (target_logprob, mu, sigma) for each position. Discards full logits.

    ``logits`` is [T, V] predicting ``target_ids`` [T] (already shifted).
    mu, sigma are the mean/std of the *next-token log-prob distribution* at that
    position (Zhang et al. Min-K%++).
    """
    import torch
    import torch.nn.functional as F

    # chunk over vocab if V is huge so we never hold softmax + logits together
    # for a wide batch. Here T is already one sequence.
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = target_ids.long().unsqueeze(-1)
    target_lp = log_probs.gather(-1, targets).squeeze(-1)
    probs = log_probs.exp()
    mu = (probs * log_probs).sum(dim=-1)
    second = (probs * log_probs.square()).sum(dim=-1)
    var = (second - mu.square()).clamp(min=0.0)
    sigma = var.sqrt()
    out = (
        target_lp.detach().cpu().numpy(),
        mu.detach().cpu().numpy(),
        sigma.detach().cpu().numpy(),
    )
    del log_probs, probs, target_lp, mu, second, var, sigma
    return out


def secret_target_positions(
    seq_len: int,
    span: tuple[int, int],
    skip_first_record_token: bool = True,
) -> list[int]:
    """Positions ``j`` in ``input_ids`` that we score as targets (predicted by logits[j-1])."""
    start, end = span
    positions = []
    first = 1 if skip_first_record_token else 0
    for j in range(max(start, first), min(end, seq_len)):
        if j >= 1:
            positions.append(j)
    return positions


def scores_from_token_stats(
    target_lp: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    min_k_pct: float = DEFAULT_MIN_K_PCT,
) -> dict[str, float]:
    nll = masked_mean_nll(target_lp)
    return {
        "masked_nll": float(nll),
        "mean_logprob": float(np.mean(target_lp)) if target_lp.size else float("nan"),
        "min_k": float(min_k_percent(target_lp, min_k_pct)),
        "min_k_plus_plus": float(min_k_plus_plus(target_lp, mu, sigma, min_k_pct)),
        "n_scored_tokens": int(target_lp.size),
    }


def combine_ft_ref(ft: dict[str, float], ref: dict[str, float] | None) -> dict[str, float]:
    """Higher = more member-like. Headline uses base-calibrated Min-K%++."""
    out = dict(ft)
    out["membership_loss"] = -ft["masked_nll"] if not math.isnan(ft["masked_nll"]) else float("nan")
    if ref is None:
        out["loss_ratio"] = float("nan")
        out["base_calibrated_min_k"] = float("nan")
        out["base_calibrated_min_k_plus_plus"] = float("nan")
        out["headline_score"] = ft.get("min_k_plus_plus", float("nan"))
        out["headline_attack_used"] = "min_k_plus_plus"
        return out
    # members: lower FT NLL relative to the base checkpoint.
    # Do not treat 0.0 as missing -- perfect NLL is a real measurement.
    ref_nll = ref.get("masked_nll")
    ft_nll = ft.get("masked_nll")
    ref_ok = ref_nll is not None and not (isinstance(ref_nll, float) and math.isnan(ref_nll))
    ft_ok = ft_nll is not None and not (isinstance(ft_nll, float) and math.isnan(ft_nll))
    if ref_ok and ft_ok:
        if float(ref_nll) == 0.0:
            out["loss_ratio"] = 0.0 if float(ft_nll) == 0.0 else float("inf")
        else:
            out["loss_ratio"] = float(ft_nll) / float(ref_nll)
        out["loss_diff"] = float(float(ref_nll) - float(ft_nll))
    else:
        out["loss_ratio"] = float("nan")
        out["loss_diff"] = float("nan")
    out["base_calibrated_min_k"] = float(ft["min_k"] - ref["min_k"])
    out["base_calibrated_min_k_plus_plus"] = float(ft["min_k_plus_plus"] - ref["min_k_plus_plus"])
    out["headline_score"] = out["base_calibrated_min_k_plus_plus"]
    out["headline_attack_used"] = "base_calibrated_min_k_plus_plus"
    out["ref_masked_nll"] = ref["masked_nll"]
    out["ref_min_k_plus_plus"] = ref["min_k_plus_plus"]
    return out


def score_sequence(
    model: Any,
    input_ids: Sequence[int],
    span: tuple[int, int] | None = None,
    min_k_pct: float = DEFAULT_MIN_K_PCT,
    skip_first_record_token: bool = True,
) -> dict[str, float]:
    """Teacher-forced secret-span scores from one forward pass."""
    import torch

    if len(input_ids) < 2:
        empty = {
            "masked_nll": float("nan"),
            "mean_logprob": float("nan"),
            "min_k": float("nan"),
            "min_k_plus_plus": float("nan"),
            "n_scored_tokens": 0,
        }
        return empty

    device = infer_device(model)
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits if hasattr(out, "logits") else out[0]
        # logits[i] predicts token i+1
        shift_logits = logits[0, :-1, :]
        shift_targets = ids[0, 1:]
        full_lp, full_mu, full_sigma = token_logprob_stats(shift_logits, shift_targets)
    del logits, shift_logits, out

    seq_len = len(input_ids)
    if span is None:
        span = (0, seq_len)
    # target position j corresponds to shift index j-1
    targets = secret_target_positions(seq_len, span, skip_first_record_token)
    idxs = [j - 1 for j in targets if 0 <= j - 1 < full_lp.shape[0]]
    if not idxs:
        return scores_from_token_stats(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            min_k_pct,
        )
    return scores_from_token_stats(full_lp[idxs], full_mu[idxs], full_sigma[idxs], min_k_pct)


def locate_secret_span(
    tokenizer: Any,
    record_text: str,
    secret: str,
    secret_token_ids: Sequence[int] | None = None,
) -> tuple[list[int], tuple[int, int] | None]:
    ids = encode_ids(tokenizer, record_text, add_special_tokens=False)
    if secret_token_ids:
        found = find_subsequence(ids, list(secret_token_ids))
        if found:
            return ids, found
    sec_ids = encode_ids(tokenizer, secret, add_special_tokens=False)
    found = find_subsequence(ids, sec_ids)
    if found:
        return ids, found
    # string fallback: locate secret in text then map approximately via prefix length
    pos = record_text.find(secret)
    if pos >= 0:
        prefix_ids = encode_ids(tokenizer, record_text[:pos], add_special_tokens=False)
        secret_ids = encode_ids(tokenizer, secret, add_special_tokens=False)
        start = len(prefix_ids)
        return ids, (start, start + len(secret_ids))
    return ids, None


def greedy_complete(
    model: Any,
    tokenizer: Any,
    prefix: str,
    max_new_tokens: int,
) -> str:
    import torch

    device = infer_device(model)
    prefix_ids = encode_ids(tokenizer, prefix, add_special_tokens=False)
    if not prefix_ids:
        prefix_ids = [int(getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 1)]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generate = getattr(model, "generate", None)
    with torch.inference_mode():
        if callable(generate):
            kwargs: dict[str, Any] = {
                "input_ids": ids,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
            }
            if pad_id is not None:
                kwargs["pad_token_id"] = pad_id
            if eos_id is not None:
                kwargs["eos_token_id"] = eos_id
            try:
                out = generate(**kwargs)
            except TypeError:
                out = generate(ids, max_new_tokens=int(max_new_tokens))
        else:
            out = ids
            for _ in range(int(max_new_tokens)):
                logits = model(input_ids=out).logits[0, -1]
                nxt = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
                if nxt.dim() == 1:
                    nxt = nxt.view(1, 1)
                out = torch.cat([out, nxt], dim=-1)
                if eos_id is not None and int(nxt.reshape(-1)[0].item()) == int(eos_id):
                    break
        new_ids = out[0, len(prefix_ids) :].detach().cpu().tolist()
    return decode_ids(tokenizer, new_ids, skip_special_tokens=True)


def generate_canary_completions(
    model: Any,
    tokenizer: Any,
    secret: str,
    secret_token_ids: Sequence[int],
    prefix_fractions: Sequence[float] = (0.25, 0.50),
) -> dict[str, Any]:
    ids = list(secret_token_ids) or encode_ids(tokenizer, secret, add_special_tokens=False)
    if len(ids) < 2:
        return {"by_prefix": [], "regurgitated": False}
    results = []
    any_hit = False
    for frac in prefix_fractions:
        n_pref = max(1, min(len(ids) - 1, int(math.floor(len(ids) * float(frac)))))
        prefix = decode_ids(tokenizer, ids[:n_pref], skip_special_tokens=True)
        remainder_ids = ids[n_pref:]
        remainder = decode_ids(tokenizer, remainder_ids, skip_special_tokens=True)
        gen = greedy_complete(model, tokenizer, prefix, max_new_tokens=max(4, len(remainder_ids) + 4))
        flags = regurgitation_flags(remainder, gen)
        flags.update({"prefix_fraction": float(frac), "prefix": prefix, "generation_len": len(gen)})
        # do not store full secret remainder in the returned public blob by default
        results.append(flags)
        any_hit = any_hit or flags["regurgitated"]
    return {"by_prefix": results, "regurgitated": bool(any_hit)}

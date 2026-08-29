# Membership scorer backends

v0.1 ships **one** membership backend: Zhang et al. Min-K%++
(`min_k_plus_plus`, version `1.0.0`). When a reference model is available the
headline is still the historical base-calibrated difference
(`base_calibrated_min_k_plus_plus`); that calibration is the same reduction,
not a second attack. Do not treat this file as a catalog of shipped MIAs.

The seam copies Privacy Meter: **signals vs attacks**. Orchestration owns
span location, forward passes, control-threshold calibration, bootstrap / CI,
profiles, `by_repetition`, and the report. A scorer returns one scalar
(higher = more member-like) and never thresholds.

## TokenSignals (extracted once per record / model state)

| Field | Already used by Min-K%++ | Needed by a future EZ-MIA |
|---|---|---|
| `gold_logprob` | yes | yes (target and reference) |
| `mu`, `sigma` | yes | no |
| `argmax_correct` | extracted, unused by the default scorer | yes (target; defines the error set) |

Extraction is the only model-facing layer (`memaudit.scoring.extract_token_signals`).
Results are cached in-memory for the run (`SignalsCache`), with a `cache_tag`
so PEFT `disable_adapter()` does not reuse target signals as the reference.

## How to add a future scorer (one file)

1. Add `src/memaudit/scorers/<name>.py` implementing:

```python
class MembershipScorer:  # Protocol — name / version / requires_reference /
    # forward_passes_per_record + score(target, reference) -> float
    ...
```

2. Select it without touching `run_audit` orchestration:

```text
--scorer memaudit.scorers.<name>:YourScorer
```

or `run_audit(..., scorer="memaudit.scorers.<name>:YourScorer")` /
`MemorizationAuditCallback(..., scorer=...)`. An optional one-line alias in
`scorers/registry.py` is convenience only.

The report records `membership.scorer = {name, version, requires_reference,
forward_passes_per_record}`.

## EZ-MIA (ACL 2026) — documented, not shipped

Ilić et al. need exactly the signals already extracted. A later
`src/memaudit/scorers/ez_mia.py` would be:

- `requires_reference = True`
- `forward_passes_per_record = 2`
- `E = ~target.argmax_correct`
- `δ = target.gold_logprob[E] − reference.gold_logprob[E]`
- `EZ = sum(δ[δ>0]) / max(sum(|δ[δ<0]|), eps)`; no-error or `N=0` → +inf
  (treat as strongest member signal)

v0.1 does **not** implement that reduction. Explicit non-goals remain:
inversion, shadow models, extra regurgitation strategies, DP auditing.

## Config

| Entry | Meaning |
|---|---|
| omitted / `min_k_plus_plus` | default Min-K%++ |
| `base_calibrated_min_k_plus_plus` | same class; still target-only if no reference |
| `package.module:Class` | import path |
| scorer instance | used as-is |

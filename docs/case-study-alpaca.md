# Auditing a TinyLlama + Alpaca LoRA fine-tune with memaudit

**Flagship public example (powered).** Measured 2026-08-28 on Apple M3 Pro (18 GB, MPS).
Not a compliance certificate. Report schema 1.1.0, tool 0.1.0.

Checked-in report: [`examples/alpaca-powered-report.json`](../examples/alpaca-powered-report.json).
Reproduce: `pip install "memaudit[peft,trl]"` then `python examples/alpaca_case_study.py`.

A 12-canary first look from 2026-08-27 is in the [appendix](#appendix-first-look-n12). It is not the headline.

## What we ran

The README API (`generate_canaries` + `inject` + `MemorizationAuditCallback`) on a recognizable stack, sized so the Clopper–Pearson interval is sellable:

| Knob | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Data | `tatsu-lab/alpaca` official `text` column, **20,000** real rows (shuffled, seed 0) |
| Trainer | Hugging Face `Trainer` + LoRA (`peft`), causal LM on the Alpaca `text` field |
| LoRA | r=8, α=16, dropout 0.05, `bias="none"`, targets `q_proj,k_proj,v_proj,o_proj` |
| Epochs / LR | 1 epoch, 2e-4, cosine, warmup 3% |
| Batch | 1 × grad-accum 4, `max_length` 256, gradient checkpointing, fp16 weights |
| Canaries | `generate_canaries(..., n=100, n_controls=200, family="high_ppl", repetitions=(1, 4, 16), seed=0)` |
| Injection | `inject(..., fmt="auto", seed=0, include_prob=1.0)` — powered setting so **100** members land |
| Inserted / controls | **100** members; **200** dedicated held-out controls |
| Token budget | **0.907%** of tokens (23,077 canary / 2,521,431 host). 5k rows would have blown the 1% cap; host grown to 20k. |
| Scoring | `MemorizationAuditCallback(..., real_sample=64, ref="auto")` → `peft.disable_adapter()` |
| Hardware | Apple M3 Pro, 18 GB unified, MPS, Python 3.12.11, torch 2.7.1 |

Wall-clock: load 6 s · data 304 s · train 10,069 s · audit 655 s · clock 4 h 43 min (05:11–09:54 IST).

`high_ppl` was called without `model=` (README snippet). The family then uses the documented rare-token unigram fallback.

## Headline numbers (from `alpaca-powered-report.json`)

| Field | Measured |
|---|---|
| `membership.headline_attack` | `base_calibrated_min_k_plus_plus` |
| `membership.tpr_at_1pct_fpr` | **0.180** (18/100) |
| `membership.ci_low` / `ci_high` | **[0.110, 0.269]** (Clopper–Pearson 95%) |
| `membership.headline_valid` | `true` |
| `membership.auc` | **0.776** (secondary) |
| `membership.n_members` / `n_controls` / `n_detected` | 100 / 200 / 18 |
| `regurgitation.overall` | **0/100** |
| `negative_controls.n` | 200 |
| `negative_controls.regurgitation_rate` | **0.00** |
| `reference.mode` | `disable_adapter` |
| `report_sha256` | `ce1675c6d93943b7affbcb726b76ffbf25244556d5c74af68e2d88e45df3f348` |

## Plain-language takeaway

A one-epoch LoRA (r=8) of TinyLlama-1.1B-Chat on 20,000 real Stanford Alpaca rows, with an honest 0.907% canary budget, **did leak membership** at the pre-declared operating point: 18 of 100 inserted canaries were detectable at 1% FPR (TPR **0.180**, 95% CI **[0.110, 0.269]**). Ranking agreed — AUC **0.776**. The same run **did not regurgitate**: 0/100 canaries and 0/200 negative controls completed from a prefix.

We did not cherry-pick a scarier threshold or drop the CI. If TPR is 0 with a tight interval, that is a sellable result (this LoRA did not leak at 1% FPR). If TPR is high with a tight interval, that is also sellable.

## How to reproduce

```bash
pip install "memaudit[peft,trl]"
# from a clone of https://github.com/mem-audit/memaudit :
python examples/alpaca_case_study.py
```

Defaults are the powered setup (`n=100`, `n_controls=200`, `include_prob=1.0`, `n_host=20000`). The script downloads TinyLlama-1.1B-Chat-v1.0 and `tatsu-lab/alpaca`, injects canaries, trains one LoRA epoch, and writes `examples/alpaca-powered-report.json`. It does not save full model weights.

## Also measured: distilgpt2 n=100 / 200

Same honest-budget rule, already measured 2026-08-27 on MPS. These CIs are sellable and stay on the site table.

| Run | Config | TPR @ 1% FPR | 95% CI | AUC | Regurg |
|---|---|---|---|---|---|
| C | distilgpt2 + LoRA r=8, 1 ep, host 80k, budget 0.77% | **0.000** (0/100) | **[0.000, 0.036]** | 0.586 | 0/100 |
| D (risky) | 5 ep, r=16, lr 1e-3 | **0.000** (0/100) primary | [0.000, 0.036] | **0.848** | 0/100 |

See [`benchmarks/README.md`](../benchmarks/README.md).

## Limitations

This is a 1.1B-class chat model, one epoch, 20,000 Alpaca rows, `max_length` 256, trained on a laptop GPU (Apple MPS). It is not a 7B multi-epoch production SFT. Thresholds are calibrated on this run's held-out controls and do not transfer across model families. The result is evidence under the attack that was run (base-calibrated Min-K%++ on the secret span, plus prefix-prompted regurgitation). It is not a GDPR / AI Act / CNIL determination. Per EDPB Opinion 28/2024 para 55, successful testing is evidence only for the attacks actually run.

## Appendix: first look (n=12)

Measured 2026-08-27 on the same machine. README default `include_prob=0.5` on `n=32` inserted **12** members (120 held-out) into 5,000 Alpaca rows at a 0.388% canary budget. Train 2,205 s / audit 323 s.

| Field | Measured |
|---|---|
| TPR @ 1% FPR | **0.500** (6/12) |
| 95% CI | **[0.211, 0.789]** |
| AUC | **0.880** |
| Regurgitation | **0/12** |
| Report | [`examples/alpaca-case-study-report.json`](../examples/alpaca-case-study-report.json), sha256 `9743dad2e7bbd0c25d8306bafe84329b2d4c79136650d7bef1e3bf3a01275ff4` |

6/12 is consistent with a true TPR anywhere from about 21% to 79%. That interval is too wide for a buyer headline, so this run is kept as a first look only.

Reproduce the first look (not the default):

```bash
python examples/alpaca_case_study.py --n 32 --n-controls 100 --n-host 5000 \
    --include-prob 0.5 --min-inserted 1 \
    --report-path examples/alpaca-case-study-report.json \
    --output-dir examples/out-alpaca-case-study
```

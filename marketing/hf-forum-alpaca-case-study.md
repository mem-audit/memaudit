<!-- Paste into https://discuss.huggingface.co
     Suggested category: Beginners  ·  or Models / Training if you prefer a more technical room
     Title below is ready to use. This file was not posted automatically — no HF forum session. -->

# [Show & Tell] Flagship case study: auditing TinyLlama-1.1B-Chat + Stanford Alpaca LoRA with memaudit (measured TPR@1%FPR = 0.50, regurgitation 0/12)

**TL;DR.** I ran the public `memaudit` 15-line API on a stack everyone already knows — TinyLlama-1.1B-Chat + 5,000 real `tatsu-lab/alpaca` rows, LoRA r=8, one epoch, honest 0.39% canary budget, `--ref auto` / `disable_adapter()`. Membership leaked. Regurgitation did not. Numbers below are copied from the checked-in `memaudit-report.json`, not a slide.

```bash
pip install "memaudit[peft,trl]"
python examples/alpaca_case_study.py
```

## Why this run

The library already had a toy positive-control (`memaudit demo`, TPR 1.0 on a 64-hidden randomly-initialized LM) and honest-budget distilgpt2 LoRA benches that printed TPR 0.000. Buyers kept asking for the recognizable names: TinyLlama or Qwen, Stanford Alpaca, LoRA, the README snippet. This is that run.

## Setup (exact)

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (not a silent fallback to a tiny random LM)
- Data: `tatsu-lab/alpaca` official `text` column, n_host = 5,000
- LoRA r=8, α=16, targets `q_proj,k_proj,v_proj,o_proj`, 1 epoch, lr 2e-4
- Canaries: `generate_canaries(tokenizer, n=32, n_controls=100, family="high_ppl", repetitions=(1, 4, 16), seed=0)`
- `inject(..., fmt="auto", seed=0)` — default `include_prob=0.5` → **12 inserted**
- Callback: `MemorizationAuditCallback(trainer=..., manifest=..., real_sample=64, ref="auto")`
- Hardware: Apple M3 Pro, 18 GB, MPS. Train 2,205 s · audit 323 s

## Headline fields from the report

| Field | Value |
|---|---|
| `membership.headline_attack` | `base_calibrated_min_k_plus_plus` |
| `membership.tpr_at_1pct_fpr` | **0.5** (6/12) |
| `membership.ci_low` / `ci_high` | **0.211 / 0.789** |
| `membership.headline_valid` | true |
| `membership.auc` | **0.880** |
| `regurgitation.overall.rate` | **0.0** (0/12; 0/4, 0/5, 0/3 by tier 1/4/16) |
| `negative_controls.regurgitation_rate` | **0.0** (n=120) |
| `reference.mode` | `disable_adapter` |
| token budget | 0.388% |
| `report_sha256` | `9743dad2e7bbd0c25d8306bafe84329b2d4c79136650d7bef1e3bf3a01275ff4` |

`memaudit verify` on that JSON is a pass.

## How to read it

This is the opposite of cherry-picking a scary leak *and* the opposite of burying a zero. At an honest budget, LoRA on TinyLlama+Alpaca was membership-detectable (AUC 0.88; TPR 0.50 at 1% FPR) and extraction-silent (no prefix completion of the secret at exact / BLEU>0.75 / NED≤0.1). The CI is wide because n=12 — the product default inclusion coin, not a hidden n=2. I am not claiming GDPR anonymity, and I am not claiming a 7B multi-epoch result.

## Links

- Write-up: https://github.com/mem-audit/memaudit/blob/main/docs/case-study-alpaca.md
- Script: https://github.com/mem-audit/memaudit/blob/main/examples/alpaca_case_study.py
- Report JSON: https://github.com/mem-audit/memaudit/blob/main/examples/alpaca-case-study-report.json
- Site page: https://ansh200516.github.io/memaudit-site/case-study.html
- PyPI: https://pypi.org/project/memaudit/
- HF org: https://huggingface.co/memaudit
- Repo: https://github.com/mem-audit/memaudit

Happy to answer methodology questions in-thread — especially “why TPR@1%FPR instead of AUC” and “why canaries instead of post-hoc MIA on the Alpaca rows.” Both have specific published answers; the report’s `compliance_annex` maps the two in-scope attacks to EDPB Opinion 28/2024 para 55 and names inversion / reconstruction / attribute inference as untested.

*memaudit produces test evidence. It does not make you compliant.*

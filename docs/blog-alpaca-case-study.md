# We audited a TinyLlama + Alpaca LoRA. It leaked membership. It did not spit the data back.

**memaudit flagship case study — measured 2026-08-27.**
By Ansh Singh · [ansh.singh.160305@gmail.com](mailto:ansh.singh.160305@gmail.com)

```bash
pip install "memaudit[peft,trl]"
python examples/alpaca_case_study.py
```

If you fine-tune on data you could not post publicly, you need two numbers, not a vibe: *can an attacker with log-probs tell what was in the train set?* and *does the model emit that text when prompted?* Those questions routinely disagree. This is the public run that shows it on a stack every ML engineer already knows.

## The setup people actually copy

We did not invent a toy corpus. We took **TinyLlama-1.1B-Chat-v1.0** and **5,000 real rows** of Stanford Alpaca (`tatsu-lab/alpaca` on the Hub), LoRA r=8 for one epoch, and wired memaudit the way the README tells you to — `generate_canaries` / `inject` / `MemorizationAuditCallback(ref="auto")`.

Honest budget, not the overfit demo: canary tokens were **0.388%** of training tokens. One hundred dedicated held-out controls so TPR@1%FPR is a real headline. Repetition grid `{1, 4, 16}`. High-perplexity family. Scoring used `peft.disable_adapter()` on one model copy.

Hardware: Apple M3 Pro, 18 GB, MPS. Train 36.8 minutes. Audit 5.4 minutes. No weights published — only the JSON report.

## The numbers (copied from `memaudit-report.json`)

| Field | Value |
|---|---|
| Method | `base_calibrated_min_k_plus_plus` |
| TPR @ 1% FPR | **0.500** (6/12) |
| 95% CI | **[0.211, 0.789]** |
| Headline valid | yes |
| AUC (secondary) | **0.880** |
| Regurgitation | **0/12** (all tiers) |
| Negative-control regurgitation | **0.00** (n=120) |
| `reference.mode` | `disable_adapter` |
| Report hash | `9743dad2…1275ff4` |

The TinyDemoLM laptop demo is a positive control (TPR 1.0 on a 64-hidden toy). The distilgpt2 LoRA benches at the same honest budget printed TPR **0.000**. This run sits between them: a recognizable 1B-class instruction tune, properly configured, **membership-detectable and extraction-silent**.

## What that means in plain language

Half the planted secrets were identifiable at a 1% false-alarm rate. The model still refused to complete them from a prefix. If you only ran a regurgitation test you would have filed this as clean. If you only looked at AUC you would have said “leaky” and stopped. memaudit prints both, with a Clopper–Pearson interval, and lets them disagree.

The interval is wide on purpose. The README default inclusion coin inserted 12 of 32 candidates. With n=12, 6 detections is consistent with a true TPR from about 21% to 79%. We did not drop the CI, raise the canary budget, or shop a scarier operating point.

## Reproduce it

```bash
pip install "memaudit[peft,trl]"
git clone https://github.com/mem-audit/memaudit.git
cd memaudit
python examples/alpaca_case_study.py
```

Write-up: [docs/case-study-alpaca.md](case-study-alpaca.md) ·
report: [examples/alpaca-case-study-report.json](../examples/alpaca-case-study-report.json) ·
site: [ansh200516.github.io/memaudit-site/case-study.html](https://ansh200516.github.io/memaudit-site/case-study.html) ·
PyPI: [pypi.org/project/memaudit](https://pypi.org/project/memaudit/) ·
HF: [huggingface.co/memaudit](https://huggingface.co/memaudit)

This is 1B, one epoch, 5,000 rows, a laptop. It is not your 7B production SFT. It is the example you can run before you read the rest of the docs.

*memaudit produces test evidence. It does not make you compliant.*

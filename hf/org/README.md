---
title: README
emoji: ¶
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
---

# memaudit

**Training-data memorization auditor for Hugging Face Trainer / TRL fine-tunes.**

A local, Apache-2.0 plugin that answers two questions every regulated fine-tune should document ([EDPB Opinion 28/2024](https://www.edpb.europa.eu/) para 55 / para 58):

1. **Membership** — can an attacker with logprob access tell what was trained on?
2. **Regurgitation** — does the model emit training content when prompted with a prefix?

memaudit injects pre-registered canaries into the raw dataset, runs a PEFT-aware pre-flight when training starts, and writes `memaudit-report.json` when training ends. The same engine is available as a post-hoc CLI.

It runs **entirely on your machine**. There is no phone-home, no account, no SaaS.

> This tool produces **evidence of resistance to the attacks it actually runs**. It does **not** make you GDPR / AI Act / CNIL compliant.

## Try it

- **Install:** `pip install memaudit`
- **30-second local demo:** `memaudit demo`
- **GitHub:** [mem-audit/memaudit](https://github.com/mem-audit/memaudit)
- **Site:** [ansh200516.github.io/memaudit-site](https://ansh200516.github.io/memaudit-site/)
- **PyPI:** [pypi.org/project/memaudit](https://pypi.org/project/memaudit/)

## What you get

A versioned JSON report with membership TPR@1%FPR + Clopper-Pearson CIs, regurgitation rates, negative controls, provenance hashes, a shipped EDPB `compliance_annex`, and an explicit limitations statement. LoRA/PEFT-aware reference scoring via `disable_adapter()` when applicable.

## Measured validation

- **TinyDemoLM** positive-control demo (`memaudit demo`) — end-to-end detection + clean controls on a laptop.
- **Pretrained distilgpt2 + LoRA** at an honest ≤1% canary budget — see the [GitHub benchmarks](https://github.com/mem-audit/memaudit/tree/main/benchmarks) and the [site](https://ansh200516.github.io/memaudit-site/).

Production audits run locally against *your* model and data; every report carries its scale label.

## Contact

Design-partner pilots: [ansh.singh.160305@gmail.com](mailto:ansh.singh.160305@gmail.com)

License: Apache-2.0

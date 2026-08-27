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

- **Interactive report demo (Space):** [memaudit/memaudit-demo](https://huggingface.co/spaces/memaudit/memaudit-demo) — browsable checked-in tiny-model report (not a 7B result)
- **Install:** `pip install memaudit`
- **30-second local demo:** `memaudit demo`
- **GitHub:** [mem-audit/memaudit](https://github.com/mem-audit/memaudit)
- **Site:** [ansh200516.github.io/memaudit-site](https://ansh200516.github.io/memaudit-site/)
- **PyPI:** [pypi.org/project/memaudit](https://pypi.org/project/memaudit/)

## What you get

A versioned JSON report with membership TPR@1%FPR + Clopper-Pearson CIs, regurgitation rates, negative controls, provenance hashes, and an explicit limitations statement. LoRA/PEFT-aware reference scoring via `disable_adapter()` when applicable.

## Scale honesty

Demo numbers on this org page and Space are from a **randomly-initialized TinyDemoLM** (hidden=64, vocab=256) overfit instrument check — not pretrained GPT-2 or 7B figures. Production audits run locally against *your* model and data.

## Contact

Design-partner pilots: [ansh.singh.160305@gmail.com](mailto:ansh.singh.160305@gmail.com)

License: Apache-2.0

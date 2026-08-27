---
title: memaudit
emoji: ¶
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Training-data memorization auditor demo
tags:
  - privacy
  - llm
  - memorization
  - membership-inference
  - audit
---

# memaudit — report demo

Interactive viewer for a **checked-in** `memaudit-report.json` from the tiny-model
overfit instrument check. This Space does **not** download large models and does
**not** phone home from the library — the library itself always runs locally.

## Scale label (mandatory)

Model: randomly-initialized **TinyDemoLM** (hidden=64, vocab=256, 1 attention block),
full fine-tune. **Not** a pretrained GPT-2 or 7B result. Canary token budget was
~99% by design so the instrument can show a positive signal.

## Links

- GitHub: [mem-audit/memaudit](https://github.com/mem-audit/memaudit)
- Site: [ansh200516.github.io/memaudit-site](https://ansh200516.github.io/memaudit-site/)
- PyPI: `pip install memaudit`
- Org: [huggingface.co/memaudit](https://huggingface.co/memaudit)
- Contact: ansh.singh.160305@gmail.com

## Reproduce locally

```bash
pip install memaudit
memaudit demo
```

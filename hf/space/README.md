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

Interactive viewer for a **checked-in** `memaudit-report.json` from the TinyDemoLM
positive-control run. This Space does **not** download large models and does
**not** phone home from the library — the library itself always runs locally.

## Scale

Model: randomly-initialized **TinyDemoLM** (hidden=64, vocab=256, 1 attention block),
full fine-tune — positive control so the instrument can show a clear hit. For pretrained
distilgpt2 + LoRA numbers, see the [GitHub benchmarks](https://github.com/mem-audit/memaudit/tree/main/benchmarks).

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

# memaudit — Hugging Face organization card (paste / upload source)

This file mirrors `hf/org/README.md`. Hugging Face organization cards are a
**public Space** named `README` under the org (`memaudit/README`) with
`sdk: static` and a root `README.md`.

After auth:

```bash
./scripts/publish_hf.sh
# or manually:
#   huggingface-cli upload memaudit/README hf/org . --repo-type space
```

Live org profile: https://huggingface.co/memaudit

Expected demo Space after publish: https://huggingface.co/spaces/memaudit/memaudit-demo

# Hugging Face presence (local assets)

Everything needed to publish the **memaudit** org card and Gradio demo Space.

| Path | Purpose |
|---|---|
| `org/README.md` | Organization card → Space `memaudit/README` (`sdk: static`) |
| `space/` | Gradio demo → Space `memaudit/memaudit-demo` |
| `ORG_README.md` / `SPACE_README.md` | Short pointers for humans |
| `../scripts/publish_hf.sh` | One-command create + upload |

## Auth

```bash
export HF_TOKEN=hf_...   # write access to organization memaudit
./scripts/publish_hf.sh
```

Expected URLs after upload:

- Org: https://huggingface.co/memaudit
- Org card Space: https://huggingface.co/spaces/memaudit/README
- Demo Space: https://huggingface.co/spaces/memaudit/memaudit-demo

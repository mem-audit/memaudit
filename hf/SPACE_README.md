# memaudit Gradio Space — README source

Canonical Space metadata + body live in `hf/space/README.md`.

**Space id:** `memaudit/memaudit-demo`  
**Hardware:** CPU basic  
**Default behavior:** pre-baked `demo-report.json` viewer (no large model download)

Publish (when ready):

```bash
export HF_TOKEN=hf_...   # write access to org memaudit
./scripts/publish_hf.sh
```

Until published, point buyers at the org profile: https://huggingface.co/memaudit

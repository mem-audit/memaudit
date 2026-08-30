"""One-command local demo: plant canaries, overfit a tiny LM, write a real report.

Numbers are measured on the machine that runs this, not copied from papers.
Scale is labeled honestly: randomly-initialized tiny causal LM, CPU/MPS, seconds.
Hugging Face GPT-2 is not used here -- some torch/transformers pairs hang on
FSDP imports. The experiment is still a real train + inject + two-verdict audit.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from memaudit.audit import run_audit
from memaudit.canaries import generate_canaries
from memaudit.injection import inject

log = logging.getLogger("memaudit")

# Enough controls that TPR@1% FPR is identified (need >=100).
DEMO_N = 16
DEMO_N_CONTROLS = 100
DEMO_SECRET_LEN = 25
DEMO_REPS = (16,)
DEMO_HOST = 4
DEMO_EPOCHS = 25
DEMO_LR = 1e-2
DEMO_MAX_LEN = 48
DEMO_BATCH = 16
DEMO_HIDDEN = 64
DEMO_VOCAB = 256


class DemoTokenizer:
    """Identity-ish tokenizer: ``t12 t13`` <-> ``[12, 13]``."""

    def __init__(self, vocab_size: int = DEMO_VOCAB) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.all_special_ids = [0, 1, 2, 3]
        self.model_max_length = DEMO_MAX_LEN

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        import re
        import zlib

        ids: list[int] = []
        for part in str(text).replace("\n", " ").split():
            matched = re.fullmatch(r"t(\d+)", part)
            if matched:
                ids.append(int(matched.group(1)) % self.vocab_size)
            else:
                ids.append(4 + (zlib.adler32(part.encode("utf-8")) % (self.vocab_size - 4)))
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False):  # noqa: ARG002
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in self.all_special_ids:
                continue
            out.append(f"t{i}")
        return " ".join(out)

    def __call__(self, text, add_special_tokens=False, **kwargs):  # noqa: ANN001, ARG002
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def __len__(self) -> int:
        return self.vocab_size


class TinyDemoLM(nn.Module):
    """One-block causal LM. Small enough to overfit planted canaries on CPU."""

    def __init__(self, vocab_size: int = DEMO_VOCAB, hidden: int = DEMO_HIDDEN) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            use_cache=False,
            tie_word_embeddings=False,
            _name_or_path="memaudit-demo-tiny",
        )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):  # noqa: ANN001, ARG002
        h = self.ln(self.embed(input_ids))
        t = h.size(1)
        causal = torch.triu(torch.ones(t, t, device=h.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        h = self.ln(h + attn_out)
        h = self.ln(h + self.ff(h))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, max_new_tokens=8, do_sample=False, **kwargs):  # noqa: ANN001, ARG002
        out = input_ids
        for _ in range(int(max_new_tokens)):
            logits = self.forward(out).logits[:, -1]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=-1)
        return out


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_lora(model: Any, enabled: bool) -> tuple[Any, bool]:
    if not enabled:
        return model, False
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "LoRA demo requested but peft could not be imported on this stack "
            f"({exc}). pip install 'memaudit[peft]' or run without --lora."
        ) from exc
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["attn", "ff"],
    )
    try:
        return get_peft_model(model, cfg), True
    except Exception:
        # TinyDemoLM is not a transformers model -- LoRA wrap may fail.
        # Fall back to full FT and say so; do not pretend we ran LoRA.
        raise RuntimeError(
            "peft.LoraConfig cannot wrap this demo LM (it is not a transformers "
            "PreTrainedModel). Run without --lora for the measured tiny-LM demo."
        )


def _batches(rows: list[dict[str, Any]], tokenizer: DemoTokenizer, batch_size: int):
    ids = [tokenizer.encode(str(r.get("text") or ""))[:DEMO_MAX_LEN] for r in rows]
    ids = [s if len(s) >= 2 else [4, 5] for s in ids]
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        width = max(len(s) for s in chunk)
        pad = tokenizer.pad_token_id
        tensor = torch.tensor([s + [pad] * (width - len(s)) for s in chunk], dtype=torch.long)
        labels = tensor.clone()
        labels[tensor == pad] = -100
        yield tensor, labels


def _train(model: Any, rows: list[dict[str, Any]], tokenizer: DemoTokenizer, *, epochs: int, lr: float, device: torch.device) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last = float("nan")
    for _ in range(int(epochs)):
        for input_ids, labels in _batches(rows, tokenizer, DEMO_BATCH):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss = model(input_ids=input_ids, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
    model.eval()
    return last


def _print_sales_summary(report: dict[str, Any]) -> None:
    mem = report.get("membership") or {}
    reg = report.get("regurgitation") or {}
    neg = report.get("negative_controls") or {}
    valid = mem.get("headline_valid")
    tpr = mem.get("tpr_at_1pct_fpr")
    print()
    print("======== memaudit demo (measured, this machine) ========")
    print(f"model:            {((report.get('model') or {}).get('class'))}")
    print(f"method:           {mem.get('headline_attack')}")
    print(f"n members / ctrl: {mem.get('n_members')} / {mem.get('n_controls')}")
    print(f"seed:             {(report.get('seeds') or {}).get('inject')}")
    if valid and tpr is not None:
        print(
            f"TPR @ 1% FPR:     {tpr:.3f}  "
            f"95% CI [{mem.get('ci_low'):.3f}, {mem.get('ci_high'):.3f}]"
        )
    else:
        print(
            f"TPR @ 1% FPR:     refused (underpowered). "
            f"exploratory={mem.get('exploratory_tpr')} "
            f"achievable FPR~{mem.get('achievable_fpr')}"
        )
        if mem.get("warning"):
            print(f"  note: {mem['warning']}")
    overall = reg.get("overall") or {}
    exec_st = (reg.get("execution") or {}).get("status") or "executed"
    if exec_st != "executed":
        reason = (reg.get("execution") or {}).get("reason") or exec_st
        print(f"regurgitation:    not run ({reason})")
    else:
        print(
            f"regurgitation:    {overall.get('n_regurgitated')}/{overall.get('n')} "
            f"rate={overall.get('rate')}"
        )
        by_tier = reg.get("by_tier") or {}
        if by_tier:
            bits = ", ".join(f"{k}x={v.get('rate')}" for k, v in by_tier.items())
            print(f"  by tier:        {bits}")
    regurg_rate_disp = (
        "not_run" if exec_st != "executed" else neg.get("regurgitation_rate")
    )
    print(
        f"neg. controls:    n={neg.get('n')}  "
        f"regurg_rate={regurg_rate_disp}  "
        f"mean_score={neg.get('mean_headline_score')}"
    )
    print(f"audit wall-clock: {report.get('audit_seconds')} s")
    print(f"schema / tool:    {report.get('schema_version')} / {report.get('tool_version')}")
    print("========================================================")
    print()


def run_demo(
    output_dir: str | Path = "examples",
    *,
    lora: bool = False,
    seed: int = 0,
    epochs: int = DEMO_EPOCHS,
    n: int = DEMO_N,
    n_controls: int = DEMO_N_CONTROLS,
) -> dict[str, Any]:
    """Train a tiny LM on planted canaries and write ``demo-report.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    tokenizer = DemoTokenizer(vocab_size=DEMO_VOCAB)
    device = _device()

    base = TinyDemoLM(DEMO_VOCAB, DEMO_HIDDEN)
    ref_clone = TinyDemoLM(DEMO_VOCAB, DEMO_HIDDEN)
    ref_clone.load_state_dict(base.state_dict())
    model, used_lora = _maybe_lora(base, lora)
    model.to(device)
    ref_clone.to(device)

    host = [{"text": f"ordinary weather note number {i} about rain and trains"} for i in range(DEMO_HOST)]
    canaries = generate_canaries(
        tokenizer,
        n=n,
        n_controls=n_controls,
        family="random",
        repetitions=DEMO_REPS,
        seed=seed,
        secret_len=DEMO_SECRET_LEN,
    )
    ds, manifest = inject(host, canaries, fmt="text", seed=seed, include_prob=1.0, tokenizer=tokenizer)
    rows = list(ds)
    canary_tokens = sum(
        len(c["secret_token_ids"]) * int(c["repetitions"])
        for c in manifest["canaries"]
        if c["included"]
    )
    host_tokens = max(sum(len(tokenizer.encode(h["text"])) for h in host), 1)
    budget_frac = canary_tokens / max(host_tokens + canary_tokens, 1)

    t0 = time.perf_counter()
    train_loss = _train(model, rows, tokenizer, epochs=epochs, lr=DEMO_LR, device=device)
    train_seconds = time.perf_counter() - t0

    report = run_audit(
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
        dataset=rows,
        ref=ref_clone,
        real_sample=0,
        output_path=out / "demo-report.json",
        skip_generation=False,
    )
    report["demo"] = {
        "scale": (
            f"randomly-initialized TinyDemoLM (hidden={DEMO_HIDDEN}, vocab={DEMO_VOCAB}, "
            f"1 attention block), {'LoRA' if used_lora else 'full fine-tune'}, "
            f"device={device}. Positive-control validation run."
        ),
        "n_canaries": n,
        "n_controls": n_controls,
        "repetitions": list(DEMO_REPS),
        "include_prob": 1.0,
        "epochs": epochs,
        "learning_rate": DEMO_LR,
        "train_seconds": round(train_seconds, 3),
        "train_loss": train_loss,
        "canary_token_budget_frac": round(float(budget_frac), 4),
        "seed": seed,
        "lora": used_lora,
        "device": str(device),
    }
    from memaudit.report import write_report
    from memaudit.utils import sha256_json, write_json

    report["report_hash"] = sha256_json({k: v for k, v in report.items() if k != "report_hash"})
    dest = write_report(report, out / "demo-report.json")
    write_json(out / "memaudit-manifest.json", manifest)
    write_summary_md(report, out / "demo-metrics.md")
    report["_demo_report_path"] = str(dest)
    _print_sales_summary(report)
    print(
        f"canary token budget (this overfit demo): ~{budget_frac:.1%} of tokens "
        "-- not the 0.1% production target"
    )
    print(f"train loss (last batch): {train_loss:.4f}")
    print(f"train wall-clock: {train_seconds:.1f}s")
    print(f"wrote {dest}")
    return report


def write_summary_md(report: dict[str, Any], path: str | Path) -> Path:
    dest = Path(path)
    mem = report.get("membership") or {}
    reg = report.get("regurgitation") or {}
    neg = report.get("negative_controls") or {}
    demo = report.get("demo") or {}
    lines = [
        "# memaudit demo metrics (measured)",
        "",
        "These numbers come from `python examples/demo.py` / `memaudit demo` on a",
        "randomly-initialized TinyDemoLM — a positive-control validation run.",
        "",
        f"- Setup: {demo.get('scale')}",
        f"- n inserted canaries / controls: {mem.get('n_members')} / {mem.get('n_controls')}",
        f"- repetitions: {demo.get('repetitions')}",
        f"- seed: {demo.get('seed')}",
        f"- method: `{mem.get('headline_attack')}`",
        f"- TPR @ 1% FPR: {mem.get('tpr_at_1pct_fpr')} (valid={mem.get('headline_valid')})  "
        f"CI [{mem.get('ci_low')}, {mem.get('ci_high')}]",
        (
            f"- regurgitation: not run "
            f"({(reg.get('execution') or {}).get('reason') or (reg.get('execution') or {}).get('status')})"
            if (reg.get("execution") or {}).get("status") not in (None, "executed")
            else f"- regurgitation overall: {((reg.get('overall') or {}).get('rate'))}"
        ),
        (
            "- regurgitation by tier: not run"
            if (reg.get("execution") or {}).get("status") not in (None, "executed")
            else f"- regurgitation by tier: {json.dumps(reg.get('by_tier') or {})}"
        ),
        (
            "- negative-control regurgitation rate: not run"
            if (reg.get("execution") or {}).get("status") not in (None, "executed")
            else f"- negative-control regurgitation rate: {neg.get('regurgitation_rate')}"
        ),
        f"- audit wall-clock: {report.get('audit_seconds')} s",
        f"- train wall-clock: {demo.get('train_seconds')} s",
        f"- train loss: {demo.get('train_loss')}",
        f"- canary token budget (demo overfit): {demo.get('canary_token_budget_frac')}",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest

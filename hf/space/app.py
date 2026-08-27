"""memaudit Hugging Face Space — pre-baked report viewer.

Default path: load checked-in demo-report.json (no large model download).
Optional: attempt a lightweight import of memaudit if the Space build has it;
the demo button never downloads large model weights.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "demo-report.json"

# Cream / stamp-red (matches site brand)
PAPER = "#F6F1E7"
PAPER_2 = "#EFE7D6"
INK = "#191510"
INK_SOFT = "#5A5142"
ACCENT = "#B23A1D"
LINE = "#D8CCB4"
PASS = "#33684B"

LINKS_MD = (
    "**GitHub:** [mem-audit/memaudit](https://github.com/mem-audit/memaudit) · "
    "**Site:** [ansh200516.github.io/memaudit-site](https://ansh200516.github.io/memaudit-site/) · "
    "**Install:** `pip install memaudit` · "
    "**Email:** [ansh.singh.160305@gmail.com](mailto:ansh.singh.160305@gmail.com)"
)

SCALE_BANNER = (
    "**Scale:** randomly-initialized TinyDemoLM "
    "(hidden=64, vocab=256, 1 attention block), full fine-tune — "
    "positive-control validation. For pretrained distilgpt2 + LoRA numbers, see "
    "[GitHub benchmarks](https://github.com/mem-audit/memaudit/tree/main/benchmarks)."
)


def load_report() -> dict[str, Any]:
    with REPORT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def summary_blocks(report: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    demo = report.get("demo") or {}
    mem = report.get("membership") or {}
    reg = report.get("regurgitation") or {}
    neg = report.get("negative_controls") or {}
    overall = reg.get("overall") or {}

    scale = demo.get("scale") or (
        "TinyDemoLM positive-control validation."
    )

    membership_md = f"""
### Membership (headline)

| Field | Value |
|---|---|
| Attack | `{mem.get("headline_attack", "—")}` |
| **TPR @ 1% FPR** | **{_fmt_pct(mem.get("tpr_at_1pct_fpr"))}** |
| 95% CI (Clopper-Pearson) | {_fmt_ci(mem.get("ci_low"), mem.get("ci_high"))} |
| Headline valid | `{mem.get("headline_valid")}` |
| Members / controls / detected | {mem.get("n_members")} / {mem.get("n_controls")} / {mem.get("n_detected")} |
| AUC (secondary) | {_fmt_pct(mem.get("auc"))} |

{mem.get("warning") or ""}
"""

    regurg_md = f"""
### Regurgitation

| Field | Value |
|---|---|
| Overall rate | **{_fmt_pct(overall.get("rate"))}** ({overall.get("n_regurgitated")}/{overall.get("n")}) |
| By tier | `{json.dumps(reg.get("by_tier") or {}, separators=(",", ": "))}` |
| Prefix fractions | `{reg.get("prefix_fractions")}` |
| Thresholds | `{json.dumps(reg.get("thresholds") or {}, separators=(",", ": "))}` |

{reg.get("note") or ""}
"""

    neg_md = f"""
### Negative controls

| Field | Value |
|---|---|
| n | {neg.get("n")} |
| Mean headline score | {_fmt_pct(neg.get("mean_headline_score"))} |
| Regurgitation rate | **{_fmt_pct(neg.get("regurgitation_rate"))}** |

{neg.get("note") or ""}
"""

    meta_md = f"""
### Run metadata

| Field | Value |
|---|---|
| Schema / tool | `{report.get("schema_version")}` / `{report.get("tool_version")}` |
| Created (UTC) | `{report.get("created_at")}` |
| Model | `{((report.get("model") or {}).get("name_or_path"))}` |
| Device / seed | `{demo.get("device")}` / `{demo.get("seed")}` |
| Train / audit wall-clock | {demo.get("train_seconds")} s / {report.get("audit_seconds")} s |
| Canary token budget | {_fmt_pct(demo.get("canary_token_budget_frac"))} |
| Local-only / phone-home | `{report.get("local_only")}` / `{report.get("phone_home")}` |
| Report SHA-256 | `{str(report.get("report_sha256") or "")[:16]}…` |

**Scale:** {scale}
"""

    limitations = report.get("limitations") or ""
    limitations_md = f"""
### Limitations (from the report)

{limitations}
"""

    recs = report.get("recommendations") or []
    recs_md = "### Recommendations\n\n" + (
        "\n".join(f"- {r}" for r in recs) if recs else "_None listed._"
    )

    return membership_md, regurg_md, neg_md, meta_md, limitations_md, recs_md


def try_local_style_demo() -> str:
    """Optional: import memaudit if present. Never downloads large models."""
    lines = [
        "### Local-style demo probe",
        "",
        "This Space ships a **pre-baked** report so builds stay fast and CPU-only.",
        "A full `memaudit demo` train+audit needs the package + a local device.",
        "",
    ]
    try:
        import memaudit  # type: ignore

        ver = getattr(memaudit, "__version__", "unknown")
        lines.append(f"- `import memaudit` succeeded (version `{ver}`).")
        lines.append(
            "- To re-run the tiny demo on **your** machine (not this Space):"
        )
        lines.append("")
        lines.append("```bash")
        lines.append("pip install memaudit")
        lines.append("memaudit demo")
        lines.append("```")
        lines.append("")
        lines.append(
            "Installing from git on a Space rebuild is optional and slow; "
            "prefer the checked-in report viewer above."
        )
    except Exception as exc:  # noqa: BLE001 — show friendly probe result
        lines.append(f"- `import memaudit` not available here (`{type(exc).__name__}: {exc}`).")
        lines.append("- That is expected for the default lightweight Space image.")
        lines.append("")
        lines.append("**Run locally:**")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install memaudit")
        lines.append("memaudit demo   # writes memaudit-report.json")
        lines.append("# or from source:")
        lines.append("pip install git+https://github.com/mem-audit/memaudit.git")
        lines.append("```")
        lines.append("")
        lines.append(
            f"<details><summary>Trace</summary>\n\n```\n{traceback.format_exc()}\n```\n</details>"
        )
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    report = load_report()
    mem_md, reg_md, neg_md, meta_md, lim_md, rec_md = summary_blocks(report)
    raw_preview = json.dumps(
        {
            k: report[k]
            for k in (
                "schema_version",
                "tool_version",
                "created_at",
                "model",
                "membership",
                "regurgitation",
                "negative_controls",
                "demo",
                "limitations",
                "local_only",
                "phone_home",
                "audit_seconds",
                "report_sha256",
            )
            if k in report
        },
        indent=2,
        default=str,
    )
    # Drop bulky roc from membership preview
    try:
        parsed = json.loads(raw_preview)
        if isinstance(parsed.get("membership"), dict):
            parsed["membership"].pop("roc", None)
        raw_preview = json.dumps(parsed, indent=2, default=str)
    except json.JSONDecodeError:
        pass

    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#FDF6F3",
            c100="#F8E4DC",
            c200="#EFC4B4",
            c300="#E09A82",
            c400="#D06E52",
            c500=ACCENT,
            c600="#8E2C14",
            c700="#732410",
            c800="#5A1C0C",
            c900="#3F1308",
            c950="#2A0C05",
        ),
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Newsreader"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        body_background_fill=PAPER,
        body_background_fill_dark=PAPER,
        block_background_fill=PAPER_2,
        block_border_color=LINE,
        body_text_color=INK,
        body_text_color_subdued=INK_SOFT,
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover="#8E2C14",
        button_primary_text_color=PAPER,
    )

    css = f"""
    .gradio-container {{
      max-width: 960px !important;
      font-size: 16px;
    }}
    .brand-title {{
      font-family: Georgia, 'Instrument Serif', serif;
      color: {INK};
      letter-spacing: -0.01em;
    }}
    .brand-pil {{
      color: {ACCENT};
      font-family: Georgia, serif;
      font-size: 1.4em;
      margin-right: 0.25em;
    }}
    .scale-box {{
      border: 1px solid {LINE};
      background: {PAPER_2};
      padding: 12px 16px;
      border-radius: 6px;
      margin: 8px 0 16px;
    }}
    .pass-note {{ color: {PASS}; }}
    """

    with gr.Blocks(theme=theme, css=css, title="memaudit") as demo:
        gr.HTML(
            f"""
            <div>
              <h1 class="brand-title"><span class="brand-pil">¶</span>memaudit</h1>
              <p style="color:{INK_SOFT};margin-top:-4px;">
                Training-data memorization auditor — browsable demo report
              </p>
            </div>
            """
        )
        gr.Markdown(SCALE_BANNER, elem_classes=["scale-box"])
        gr.Markdown(LINKS_MD)

        with gr.Tabs():
            with gr.Tab("Headline metrics"):
                gr.Markdown(mem_md)
                gr.Markdown(reg_md)
                gr.Markdown(neg_md)
                gr.Markdown(
                    '<p class="pass-note">Negative-control regurgitation at 0.0 '
                    "means never-inserted canaries were not emitted — expected for a calibrated run.</p>"
                )
            with gr.Tab("Run metadata"):
                gr.Markdown(meta_md)
            with gr.Tab("Limitations"):
                gr.Markdown(lim_md)
                gr.Markdown(rec_md)
            with gr.Tab("Report JSON (trimmed)"):
                gr.Code(value=raw_preview, language="json", interactive=False)
                gr.Markdown(
                    f"Full artifact on disk in this Space: `{REPORT_PATH.name}` "
                    f"({REPORT_PATH.stat().st_size // 1024} KiB)."
                )

        with gr.Accordion("Optional: probe local-style demo import", open=False):
            probe_btn = gr.Button("Run local-style demo probe", variant="secondary")
            probe_out = gr.Markdown(
                "Click to check whether `memaudit` is importable in this Space image. "
                "Default path stays the pre-baked report — no large downloads."
            )
            probe_btn.click(fn=try_local_style_demo, outputs=probe_out)

        gr.Markdown(
            "---\n"
            "memaudit produces **test evidence** for the attacks it runs; "
            "it does **not** make you GDPR / AI Act / CNIL compliant. "
            "Library remains fully local — this Space only displays a public demo report."
        )

    return demo


if __name__ == "__main__":
    build_app().launch()

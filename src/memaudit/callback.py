"""Thin Trainer callback: pre-flight + survival scan, then shared ``run_audit``.

Injection does **not** happen here. See ``memaudit.injection`` / ``memaudit.inject()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from memaudit.audit import run_audit, write_deferred_audit
from memaudit.exceptions import MemauditConfigError
from memaudit.preflight import is_sharded_backend, run_preflight
from memaudit.utils import logger, write_json


class MemorizationAuditCallback(TrainerCallback):
    """PEFT-aware memorization audit at the end of ``trainer.train()``.

    Parameters
    ----------
    trainer
        Required handle (callbacks do not receive the Trainer). Same pattern as
        TRL's ``LogCompletionsCallback(trainer=...)``.
    manifest
        JSON-serializable dict returned by ``inject()``.
    real_sample
        How many real training records to score for the set-level module.
    output_dir
        Defaults to ``TrainingArguments.output_dir``.
    """

    def __init__(
        self,
        trainer: Any,
        manifest: dict[str, Any],
        real_sample: int = 64,
        output_dir: str | Path | None = None,
        ref: Any = "auto",
        skip_generation: bool = False,
        seeds: Any | None = None,
        release_context: str | None = None,
    ) -> None:
        if trainer is None:
            raise MemauditConfigError(
                "MemorizationAuditCallback requires trainer=... "
                "(Trainer callbacks are not passed the trainer in kwargs)."
            )
        if not isinstance(manifest, dict) or "canaries" not in manifest:
            raise MemauditConfigError("manifest must be the dict returned by memaudit.inject()")
        self.trainer = trainer
        self.manifest = manifest
        self.real_sample = int(real_sample)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.ref = ref
        self.skip_generation = skip_generation
        self.seeds = list(seeds) if seeds is not None else None
        self.release_context = release_context
        self.preflight: dict[str, Any] | None = None
        self.report: dict[str, Any] | None = None
        self._artifacts_written = False

    def _resolve_output_dir(self, args: Any) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        raw = getattr(args, "output_dir", None) or getattr(self.trainer.args, "output_dir", None) or "."
        return Path(raw)

    def _tokenizer(self, kwargs: dict[str, Any]) -> Any:
        # transformers 5.x removed the tokenizer= callback kwarg. Use processing_class.
        tok = kwargs.get("processing_class")
        if tok is not None:
            return tok
        trainer = self.trainer
        return getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)

    def _write_artifacts(self, output_dir: Path) -> None:
        if self._artifacts_written:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "memaudit-manifest.json", self.manifest)
        write_json(
            output_dir / "memaudit-canaries.json",
            {
                "canaries": self.manifest.get("canaries"),
                "fmt": self.manifest.get("fmt"),
                "seed": self.manifest.get("seed"),
            },
        )
        self._artifacts_written = True

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        model = kwargs.get("model") or getattr(self.trainer, "model", None)
        tokenizer = self._tokenizer(kwargs)
        output_dir = self._resolve_output_dir(args)
        self._write_artifacts(output_dir)
        self.preflight = run_preflight(
            model=model,
            trainer=self.trainer,
            manifest=self.manifest,
            tokenizer=tokenizer,
            args=args,
            train_dataset=getattr(self.trainer, "train_dataset", None),
            raise_fatal=True,
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control
        output_dir = self._resolve_output_dir(args)
        self._write_artifacts(output_dir)
        defer, reason = is_sharded_backend(self.trainer)
        if defer:
            payload = write_deferred_audit(
                output_dir,
                self.manifest,
                reason=reason,
                model_path_hint=str(output_dir),
            )
            logger.warning("deferred audit under sharded training (%s)", reason)
            logger.warning("re-run: %s", payload["command"])
            print(f"[memaudit] deferred audit under sharded training ({reason}).")
            print(f"[memaudit] re-run: {payload['command']}")
            return control

        model = kwargs.get("model") or getattr(self.trainer, "model", None)
        tokenizer = self._tokenizer(kwargs)
        if tokenizer is None:
            raise MemauditConfigError(
                "No processing_class/tokenizer on the Trainer. In-process "
                "scoring cannot run; refusing to skip silently. Pass "
                "processing_class= to Trainer/SFTTrainer, or re-run "
                "`memaudit audit --model ... --canary-set memaudit-manifest.json`."
            )
        report_path = output_dir / "memaudit-report.json"
        self.report = run_audit(
            model=model,
            tokenizer=tokenizer,
            manifest=self.manifest,
            dataset=getattr(self.trainer, "train_dataset", None),
            ref=self.ref,
            real_sample=self.real_sample,
            output_path=report_path,
            preflight_findings=self.preflight,
            skip_generation=self.skip_generation,
            trainer=self.trainer,
            seeds=self.seeds,
            release_context=self.release_context,
        )
        logger.info("wrote %s", report_path)
        print(f"[memaudit] wrote {report_path}")
        return control

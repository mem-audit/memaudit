#!/usr/bin/env python3
"""One-command demo. Equivalent to ``memaudit demo``.

    python examples/demo.py
    python examples/demo.py --lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from a clone without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from memaudit.demo import run_demo  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Plant canaries, overfit a tiny GPT-2, print measured metrics")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--lora", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    ns = p.parse_args()
    run_demo(output_dir=ns.output_dir, lora=ns.lora, seed=ns.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

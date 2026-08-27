# memaudit demo metrics (measured)

These numbers come from `python examples/demo.py` / `memaudit demo` on a
randomly-initialized tiny causal LM. They are **not** paper-scale 7B figures.

- Setup: randomly-initialized TinyDemoLM (hidden=64, vocab=256, 1 attention block), full fine-tune, device=mps. Not a 7B or pretrained-GPT-2 result.
- n inserted canaries / controls: 16 / 100
- repetitions: [16]
- seed: 0
- method: `base_calibrated_min_k_plus_plus`
- TPR @ 1% FPR: 1.0 (valid=True)  CI [0.7940927857921772, 1.0]
- regurgitation overall: 1.0
- regurgitation by tier: {"16": {"n": 16, "n_regurgitated": 16, "rate": 1.0}}
- negative-control regurgitation rate: 0.0
- audit wall-clock: 20.345 s
- train wall-clock: 6.96 s
- train loss: 0.12576882541179657
- canary token budget (demo overfit): 0.9944

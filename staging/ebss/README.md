> **V1 status:** Phase1 smoke closed (no EBSS uplift vs random at 64×512). **Not** a V1 optimization target. See `results/moe-ebss-gguf/`.

# staging/ebss

Independent **reimplementation** of MoEQuant **EBSS** (Expert-Balanced
Self-Sampling) for calibration → imatrix text. Cite MoEQuant; do not claim we proposed EBSS.

| File | Role |
|------|------|
| `generate_calib.py` | Beam search with gate hooks / σ scoring |
| `extract_text.py` | JSONL / tensor → plain text for llama.cpp imatrix |

Recipe target: `recipes/moe-ebss-gguf/qwen35-35b-a3b.yaml`.

Phase-1 plan: split into `src/ptqkit/calib/ebss/{gates,scoring,beam,sampler}.py`
with unit-level resume and recipe-driven config (no argparse defaults as source of truth).

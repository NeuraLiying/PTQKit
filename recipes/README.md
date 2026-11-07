# Recipes

YAML recipes document intended configs and expected metrics. `qs run` is still
scaffold — execute via `staging/*/scripts` (see `docs/REPRODUCE.md`).

| Recipe | Line | Status |
|--------|------|--------|
| `moe-ebss-gguf/qwen35-35b-a3b.yaml` | Calibration (EBSS) | Phase1 smoke **done** (no uplift vs random) |
| `moe-ebss-gguf/qwen35-35b-a3b-random.yaml` | EBSS control | Phase1 smoke **done** |
| `fastvit-w4a4/ma36.yaml` | Reconstruct (FIMA-Q) | **legacy** Prec@1 83.23 |
| `vlm-gptaq-mbq/qwen3-vl-30b-a3b.yaml` | Plan (GPTAQ+MBQ) | Phase1 + ablations **sealed** |

## Env

GPU-host conda/pip pins: [`env/`](env/) (`quantstage.environment.yml`).

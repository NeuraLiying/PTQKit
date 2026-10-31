# Phase1 VLM GPTAQ+MBQ (same lmms-eval harness)
| Task | n | BF16 | GPTAQ+MBQ | Δpt | stderr (BF16 / Quant) |
|------|--:|-----:|----------:|----:|-----------------------|
| AI2D | 3088 | 85.62 | 84.91 | -0.71 | 0.63pt / 0.64pt |
| MMMU-val | 900 | 54.56 | 54.00 | -0.56 | — / — |
| RealWorldQA | 765 | 65.10 | 65.75 | +0.65 | 1.72pt / 1.72pt |
| ScienceQA | 4241 | 93.99 | 92.78 | -1.21 | 0.37pt / 0.40pt |
| TextVQA-val | 5000 | 82.31 | 79.00 | -3.31 | 0.49pt / 0.53pt |
| VizWiz-val | 4319 | 69.97 | 69.07 | -0.90 | 0.62pt / 0.63pt |
| MMBench-EN-dev | 4329 | 86.94 | 85.14 | -1.80 | — / — |

**Quant spec:** ViT GPTAQ `W4g32A8` + LLM MBQ `W4g128A16` (pseudo-quant). Mean Δ ≈ **−1.11 pt**.

**Provenance:** GPU-host `$QS_WORK/runs/vlm_qwen3vl_30b_a3b/`; artifact hashes in `summary.json`.

**Do not compare** these MMMU/RWQA numbers to official Qwen tables (~74 / ~77); see `staging/vlm/docs/HARNESS_VS_OFFICIAL.md`.

**Ablations sealed (2026-10-31):** ViT-only×7 + LLM-only×7 + RTN×3 (TextVQA/MMMU/AI2D). TextVQA: ViT-only 79.59 ≈ Full 79.00; LLM-only 82.08 ≈ BF16 82.31 → drop mostly **ViT-side**.
**LLM coverage footnote:** Attn-Linear-only / packed experts not pseudo-quantized — see `ABLATION_STATUS.md`.

## Ablation table (same harness)

| Task | BF16 | Full | ViT-only | LLM-only | RTN |
|------|-----:|-----:|---------:|---------:|----:|
| AI2D | 85.62 | 84.91 | 85.23 | 85.10 | 84.52 |
| MMMU-val | 54.56 | 54.00 | 54.33 | 54.44 | 54.33 |
| RealWorldQA | 65.10 | 65.75 | 64.18 | 65.62 | — |
| ScienceQA | 93.99 | 92.78 | 93.59 | 93.42 | — |
| TextVQA-val | 82.31 | 79.00 | 79.59 | 82.08 | 78.91 |
| VizWiz-val | 69.97 | 69.07 | 69.76 | 69.64 | — |
| MMBench-EN-dev | 86.94 | 85.14 | 85.74 | 86.25 | — |

# Quant spec (Phase1, one-liner)

**ViT GPTAQ `W4 g32 A8` (sym) + LLM MBQ `W4 g128 A16` (reweight) → pseudo-quant eval.**

| Side | Method | Weights | Activations | Group | Calib | Notes |
|------|--------|---------|-------------|-------|-------|-------|
| Vision tower + `merger` | GPTAQ | 4-bit, symmetric | 8-bit (`a_groupsize=-1`) | 32 | 32 images (`vit_calib`) | Saved `vision_gptaq.pt` (~1.1G) |
| LLM (48 blocks) | MBQ | 4-bit | 16-bit (no act quant) | 128 | COCO captions, `n_samples=16`, `--reweight` | Saved `mbq_scales.pt` (~221KB, 48 layer scales) |

**Skipped / not in scale search**
- Embedding and `lm_head` / output projections: outside block linears (MBQ defines `EMBEDDING_KEYWORDS` / `LM_HEAD_KEYWORDS`).
- Router/gate: not specially force-skipped in our run notes; MBQ walks named linears inside blocks — treat as “implementation-default” until an explicit dump is added.
- DeepStack merger taps: vision path quantizes main `vision.merger`; deepstack feature unpack is a **forward patch**, not a separate quant target.

**Eval**
- Fake/pseudo weight quant only (`--pseudo_quant`). **No** memory or throughput claim.
- Artifacts live on the GPU host under `$QS_WORK/runs/vlm_qwen3vl_30b_a3b/scale_cache/` (hashes in `results/vlm-gptaq-mbq/summary.json`).

**LLM coverage (transformers 5.21.0 / Qwen3-VL-MoE):** Phase1 pseudo-quant walks `nn.Linear` inside decoder layers (Attn q/k/v/o only). Packed MoE experts (`gate_up_proj` / `down_proj` Parameters) and router weights are **not** fake-quantized. Effective quantized share ≈ **~5%** of total weights (ViT + Attn). See `results/vlm-gptaq-mbq/ABLATION_STATUS.md`. Stage2 adds packed-expert RTN coverage.

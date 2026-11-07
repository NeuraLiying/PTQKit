# staging/vlm — Qwen3-VL-30B-A3B GPTAQ + MBQ

Phase1 **accuracy study** (pseudo-quant) is packaged. Full MBQ/GPTAQ sources stay on the GPU host under `$QS_WORK/GPTAQ-MBQ` (not vendored here; MIT/Apache attribution in `NOTICE`).

## Quant spec (one line)

See [`docs/QUANT_SPEC.md`](docs/QUANT_SPEC.md): **ViT GPTAQ W4g32A8 + LLM MBQ W4g128A16**.

## Results

- Table: [`../../results/vlm-gptaq-mbq/PHASE1_TABLE.md`](../../results/vlm-gptaq-mbq/PHASE1_TABLE.md)
- Provenance JSON: [`../../results/vlm-gptaq-mbq/summary.json`](../../results/vlm-gptaq-mbq/summary.json)
- Recipe: [`../../recipes/vlm-gptaq-mbq/qwen3-vl-30b-a3b.yaml`](../../recipes/vlm-gptaq-mbq/qwen3-vl-30b-a3b.yaml)
- Why BF16 ≠ official MMMU/RWQA: [`docs/HARNESS_VS_OFFICIAL.md`](docs/HARNESS_VS_OFFICIAL.md)
- Ablations: [`../../results/vlm-gptaq-mbq/ABLATION_STATUS.md`](../../results/vlm-gptaq-mbq/ABLATION_STATUS.md)

## Reproduce TextVQA / MMMU

```bash
export QS_WORK=/path/to/quantstage-work   # GPU host work tree
export QS_MODEL_ROOT=$QS_WORK/models
# conda env: quantstage
# 1) Quant (skip if scale_cache already present)
CUDA_VISIBLE_DEVICES=0,1 bash staging/vlm/scripts/run_quant_gptaq_mbq_qwen3vl30b.sh

# 2) Eval full GPTAQ+MBQ
MODE=quant CUDA_VISIBLE_DEVICES=0,1   bash staging/vlm/scripts/run_eval_gptaq_mbq_qwen3vl30b.sh
# single task:
MODE=quant TASK_FILTER=textvqa_val ...
MODE=quant TASK_FILTER=mmmu_val ...

# 3) BF16 control
MODE=bf16 TASK_FILTER=mmmu_val ...
```

Expected artifacts:

- `$QS_WORK/runs/vlm_qwen3vl_30b_a3b/scale_cache/{vision_gptaq.pt,mbq_scales.pt}`
- `$QS_WORK/runs/vlm_qwen3vl_30b_a3b/eval/{gptaq_mbq,bf16}/`

**Patches not included in v0.1.0a0**: GPTAQ-for-vision license unverified. See [`patches/README.md`](patches/README.md).

## Implementation Not Included

**v0.1.0a0 does not include VLM implementation patches** due to unverified GPTAQ-for-vision license.

- Phase1/Stage2 **results** are included and documented
- Reproduction requires independent access to MBQ (MIT) and GPTAQ-for-vision sources
- See [`patches/README.md`](patches/README.md) for details

## Do not

- Commit `.pt` weights / sample dumps / HF tokens / personal host paths.
- Claim deployment speedups or official-table parity.
- Start Stage2 / RM-MBQ / W3 — deferred (needs a new plan).

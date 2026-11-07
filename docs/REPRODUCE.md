# Reproduce Phase-1 results

Timezone for logged runs: **Asia/Shanghai (UTC+8)**.
Hardware note: VLM 7-task suite historically used multi-GPU; for new probes prefer **2 GPUs** unless you have capacity.

## 0. What you need outside the repo

| Asset | Env / path |
|-------|------------|
| GPU host work tree with GPTAQ-MBQ + lmms-eval | `QS_WORK` (not vendored here) |
| Model weights | `QS_MODEL_ROOT` / `$QS_WORK/models/...` |
| ImageNet (FastViT only) | `IMAGENET_ROOT` |
| HF auth (downloads) | `huggingface-cli login` or `HF_TOKEN` |

Apply patches from `staging/vlm/patches/` with `patch -p0` from the GPTAQ-MBQ root when rebuilding from an upstream checkout.

## 1. Conda env

```bash
conda env create -f recipes/env/quantstage.environment.yml
conda activate quantstage
# On the GPU host, also install editable lmms-eval / qmllm deps used by GPTAQ-MBQ.
```

Pins: `recipes/env/requirements-quantstage.txt` (+ optional freeze file).

## 2. MoE EBSS (smoke run)

```bash
export QS_MODEL_ROOT=/path/to/models
python staging/ebss/generate_calib.py \
  --model "${QS_MODEL_ROOT}/Qwen3.5-35B-A3B" \
  --out /tmp/ebss_smoke.pt --out_jsonl /tmp/ebss_smoke.jsonl \
  --num_seqs 64 --seqlen 512 --w 4 --device cuda
```

Scores / honesty: `results/moe-ebss-gguf/`.

## 3. FastViT W4A4

```bash
python staging/fastvit/fastvit/test_quant_fastvit.py \
  --model fastvit_ma36 --config staging/fastvit/configs/4bit/best.py \
  --dataset "${IMAGENET_ROOT}" --calibrate --quant-act --optimize
```

Headline Prec@1 **83.23** / Prec@5 **96.45** (ImageNet val). See `results/fastvit-w4a4/`.

## 4. VLM GPTAQ + MBQ + ablations

```bash
export QS_WORK=/path/to/quantstage-work   # must contain GPTAQ-MBQ/, models/, runs/
export CUDA_VISIBLE_DEVICES=0,1           # prefer 2 free GPUs for new runs
conda activate quantstage

# Quant (skip if scale_cache hashes match results/vlm-gptaq-mbq/summary.json)
bash staging/vlm/scripts/run_quant_gptaq_mbq_qwen3vl30b.sh

# Full / BF16 eval
MODE=quant bash staging/vlm/scripts/run_eval_gptaq_mbq_qwen3vl30b.sh
MODE=bf16  bash staging/vlm/scripts/run_eval_gptaq_mbq_qwen3vl30b.sh

# Ablations (MODE=vit_only|llm_only|rtn_probe)
MODE=llm_only TASK_FILTER=textvqa_val bash staging/vlm/scripts/run_eval_ablation_modes.sh
```

Canonical tables: `results/vlm-gptaq-mbq/PHASE1_TABLE.md`, `ABLATION_STATUS.md`.
Harness ≠ official Qwen numbers: `staging/vlm/docs/HARNESS_VS_OFFICIAL.md`.

## 5. Compliance gate

```bash
./tools/scrub_check.sh
```

Must PASS before any push.

#!/usr/bin/env bash
# Staging pointer script for PTQKit VLM GPTAQ+MBQ.
# Requires: QS_WORK (GPU host quantstage-work root), conda env `quantstage`.
# Do not commit tokens/weights. Paths below are env-parameterized.
set -euo pipefail
QS_WORK=${QS_WORK:?set QS_WORK to quantstage-work root on the GPU host}
QS="$QS_WORK"
WORK=$QS/GPTAQ-MBQ
RUN=$QS/runs/vlm_qwen3vl_30b_a3b
MODEL_PATH=${MODEL_PATH:-$QS/models/Qwen3-VL-30B-A3B}
SCALE_DIR=$RUN/scale_cache
VISION_SAVE=$SCALE_DIR/vision_gptaq.pt
MBQ_SCALE=$SCALE_DIR/mbq_scales.pt
MODE=${MODE:-quant}  # quant | bf16
if [[ "$MODE" == "bf16" ]]; then
  OUTPUT_DIR=$RUN/eval/bf16
else
  OUTPUT_DIR=$RUN/eval/gptaq_mbq
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,5,6}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH="$WORK:$WORK/3rdparty/lmms-eval:${PYTHONPATH:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate quantstage
cd "$WORK"
mkdir -p "$OUTPUT_DIR" "$RUN/logs"

TASK_FILTER=${TASK_FILTER:-}  # optional single lmms task name

TASKS=(
  "AI2D:ai2d"
  "MMMU:mmmu_val"
  "RealWorldQA:realworldqa"
  "ScienceQA:scienceqa"
  "TextVQA:textvqa_val"
  "VizWiz:vizwiz_vqa_val"
  "MMBench:mmbench_en_dev"
)

for item in "${TASKS[@]}"; do
  bench_name="${item%%:*}"
  task_name="${item##*:}"
  if [[ -n "$TASK_FILTER" && "$task_name" != "$TASK_FILTER" ]]; then
    continue
  fi
  run_name="${MODE}_${bench_name}"
  run_output_dir="$OUTPUT_DIR/$run_name"
  mkdir -p "$run_output_dir"
  model_args="pretrained=$MODEL_PATH,device_map=auto,use_flash_attention_2=False"
  echo "[STATUS] Eval mode=$MODE task=$task_name" | tee -a "$RUN/logs/eval.log"
  if [[ "$MODE" == "bf16" ]]; then
    python main.py \
      --model_name qwen3_vl_moe \
      --model_args "$model_args" \
      --tasks "$task_name" \
      --batch_size 1 \
      --output_path "$run_output_dir" \
      --log_samples \
      --skip_llm_quant 2>&1 | tee -a "$RUN/logs/eval_${MODE}_${bench_name}.log"
  else
    python main.py \
      --model_name qwen3_vl_moe \
      --model_args "$model_args" \
      --tasks "$task_name" \
      --batch_size 1 \
      --output_path "$run_output_dir" \
      --log_samples \
      --vit_method gptaq \
      --vit_wbits 4 --vit_sym --vit_a_bits 8 \
      --load_vision "$VISION_SAVE" \
      --method mbq \
      --w_bit 4 --a_bit 16 --w_group 128 \
      --scale_path "$MBQ_SCALE" \
      --pseudo_quant 2>&1 | tee -a "$RUN/logs/eval_${MODE}_${bench_name}.log"
  fi
done
echo "[STATUS] Eval done -> $OUTPUT_DIR" | tee -a "$RUN/logs/eval.log"

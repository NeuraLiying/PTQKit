#!/usr/bin/env bash
# Phase1 ablation batch: remaining llm_only / vit_only / rtn_probe tasks.
# Skip a task if a *results.json already exists under its output dir.
set -euo pipefail
QS_WORK=${QS_WORK:?set QS_WORK to quantstage-work root on the GPU host}
WORK=$QS_WORK/GPTAQ-MBQ
RUN=${RUN_DIR:-$QS_WORK/runs/vlm_qwen3vl_30b_a3b}
MODEL_PATH=${MODEL_PATH:-$QS_WORK/models/Qwen3-VL-30B-A3B}
SCALE_DIR=$RUN/scale_cache
VISION_SAVE=$SCALE_DIR/vision_gptaq.pt
MBQ_SCALE=$SCALE_DIR/mbq_scales.pt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,5,6}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTHONPATH="$WORK:$WORK/3rdparty/lmms-eval:${PYTHONPATH:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate quantstage
cd "$WORK"
mkdir -p "$RUN/logs"

STATUS_LOG=$RUN/logs/phase1_ablation_batch.log
echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] START phase1 ablation batch GPUs=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS_LOG"

# name:task pairs (bench label used in outdir)
ALL_TASKS=(
  "AI2D:ai2d"
  "MMMU:mmmu_val"
  "RealWorldQA:realworldqa"
  "ScienceQA:scienceqa"
  "TextVQA:textvqa_val"
  "VizWiz:vizwiz_vqa_val"
  "MMBench:mmbench_en_dev"
)

has_results() {
  local d="$1"
  find "$d" -name "*results.json" 2>/dev/null | head -1 | grep -q .
}

run_one() {
  local mode="$1" bench="$2" task="$3"
  local out_root="$RUN/eval/ablation_${mode}"
  local run_output_dir="$out_root/${mode}_${task}"
  # also accept legacy-style ${mode}_${bench}
  local alt_dir="$out_root/${mode}_${bench}"
  mkdir -p "$run_output_dir"
  if has_results "$run_output_dir" || has_results "$alt_dir"; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] SKIP mode=$mode task=$task (results exist)" | tee -a "$STATUS_LOG"
    return 0
  fi
  # For vit_only: also skip if archived partial already has this bench
  if [[ "$mode" == "vit_only" ]]; then
    local arch="$RUN/eval/gptaq_mbq_vit_only_or_partial_20261030_024202/quant_${bench}"
    if has_results "$arch"; then
      echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] SKIP mode=vit_only task=$task (archive has results)" | tee -a "$STATUS_LOG"
      return 0
    fi
  fi
  local model_args="pretrained=$MODEL_PATH,device_map=auto,use_flash_attention_2=False"
  local logf="$RUN/logs/ablation_${mode}_${bench}.log"
  echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] RUN mode=$mode task=$task -> $run_output_dir" | tee -a "$STATUS_LOG"
  local -a cmd=(python main.py --model_name qwen3_vl_moe --model_args "$model_args" --tasks "$task" --batch_size 1 --output_path "$run_output_dir" --log_samples)
  case "$mode" in
    vit_only)
      cmd+=(--vit_method gptaq --vit_wbits 4 --vit_sym --vit_a_bits 8 --load_vision "$VISION_SAVE" --skip_llm_quant)
      ;;
    llm_only)
      cmd+=(--method mbq --w_bit 4 --a_bit 16 --w_group 128 --scale_path "$MBQ_SCALE" --pseudo_quant)
      ;;
    rtn_probe)
      # Hold ViT GPTAQ fixed; LLM RTN W4g128A16 (weight-only since a_bit=16) vs MBQ
      cmd+=(--method rtn --w_bit 4 --a_bit 16 --w_group 128 --pseudo_quant \
        --vit_method gptaq --vit_wbits 4 --vit_sym --vit_a_bits 8 --load_vision "$VISION_SAVE")
      ;;
    *) echo "bad mode $mode" >&2; return 2 ;;
  esac
  set +e
  "${cmd[@]}" >"$logf" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] FAIL mode=$mode task=$task rc=$rc (see $logf)" | tee -a "$STATUS_LOG"
    return $rc
  fi
  if ! has_results "$run_output_dir"; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] FAIL mode=$mode task=$task no results.json" | tee -a "$STATUS_LOG"
    return 1
  fi
  echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] OK mode=$mode task=$task" | tee -a "$STATUS_LOG"
}

# Queue: llm_only remaining → vit_only missing → rtn min3
LLM_ONLY_TASKS=(
  "AI2D:ai2d"
  "MMMU:mmmu_val"
  "RealWorldQA:realworldqa"
  "ScienceQA:scienceqa"
  "TextVQA:textvqa_val"
  "VizWiz:vizwiz_vqa_val"
  "MMBench:mmbench_en_dev"
)
VIT_ONLY_TASKS=(
  "AI2D:ai2d"
  "RealWorldQA:realworldqa"
  "VizWiz:vizwiz_vqa_val"
  "MMBench:mmbench_en_dev"
)
RTN_TASKS=(
  "TextVQA:textvqa_val"
  "MMMU:mmmu_val"
  "AI2D:ai2d"
)

FAILS=0
for item in "${LLM_ONLY_TASKS[@]}"; do
  bench="${item%%:*}"; task="${item##*:}"
  run_one llm_only "$bench" "$task" || FAILS=$((FAILS+1))
done
for item in "${VIT_ONLY_TASKS[@]}"; do
  bench="${item%%:*}"; task="${item##*:}"
  run_one vit_only "$bench" "$task" || FAILS=$((FAILS+1))
done
for item in "${RTN_TASKS[@]}"; do
  bench="${item%%:*}"; task="${item##*:}"
  run_one rtn_probe "$bench" "$task" || FAILS=$((FAILS+1))
done

echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] DONE phase1 ablation batch fails=$FAILS" | tee -a "$STATUS_LOG"
exit $FAILS

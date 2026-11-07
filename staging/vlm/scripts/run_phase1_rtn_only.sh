#!/usr/bin/env bash
# Phase1 RTN-only relaunch after get_blocks Qwen3-VL-MoE fix.
# Does NOT touch llm_only / vit_only.
set -euo pipefail
QS_WORK=${QS_WORK:?set QS_WORK to quantstage-work root on the GPU host}
WORK=$QS_WORK/GPTAQ-MBQ
RUN=${RUN_DIR:-$QS_WORK/runs/vlm_qwen3vl_30b_a3b}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,5,6}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTHONPATH="$WORK:$WORK/3rdparty/lmms-eval:${PYTHONPATH:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate quantstage
cd "$WORK"
mkdir -p "$RUN/logs" "$RUN/eval/ablation_rtn_probe"

STATUS_LOG=$RUN/logs/phase1_ablation_batch.log
echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] START phase1 RTN-only relaunch GPUs=$CUDA_VISIBLE_DEVICES (post get_blocks fix)" | tee -a "$STATUS_LOG"

has_results() {
  local d="$1"
  find "$d" -name "*results.json" 2>/dev/null | head -1 | grep -q .
}

FAILS=0
for item in TextVQA:textvqa_val MMMU:mmmu_val AI2D:ai2d; do
  bench=${item%%:*}
  task=${item##*:}
  out_root="$RUN/eval/ablation_rtn_probe"
  run_output_dir="$out_root/rtn_probe_${task}"
  mkdir -p "$run_output_dir"
  if has_results "$run_output_dir"; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] SKIP rtn_probe task=$task (results exist)" | tee -a "$STATUS_LOG"
    continue
  fi
  logf="$RUN/logs/ablation_rtn_probe_${bench}.log"
  echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] RUN rtn_probe task=$task -> $run_output_dir" | tee -a "$STATUS_LOG"
  set +e
  MODE=rtn_probe TASK_FILTER=$task QS_WORK=$QS_WORK RUN_DIR=$RUN \
    bash "$QS_WORK/scripts/run_eval_ablation_modes.sh" >"$logf" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] FAIL rtn_probe task=$task rc=$rc (see $logf)" | tee -a "$STATUS_LOG"
    FAILS=$((FAILS+1))
    continue
  fi
  if ! has_results "$run_output_dir"; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] FAIL rtn_probe task=$task no results.json" | tee -a "$STATUS_LOG"
    FAILS=$((FAILS+1))
    continue
  fi
  echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] OK rtn_probe task=$task" | tee -a "$STATUS_LOG"
done

echo "[$(date "+%Y-%m-%d %H:%M:%S %Z")] DONE phase1 RTN-only relaunch fails=$FAILS" | tee -a "$STATUS_LOG"
exit $FAILS

#!/usr/bin/env bash
# Ablation eval modes on GPU-host GPTAQ-MBQ tree ($QS_WORK).
# MODE=vit_only | llm_only | rtn_probe | bf16 | quant
# TASK_FILTER=textvqa_val (recommended for cheap probe)
set -euo pipefail
QS_WORK=${QS_WORK:?set QS_WORK}
WORK=$QS_WORK/GPTAQ-MBQ
RUN=${RUN_DIR:-$QS_WORK/runs/vlm_qwen3vl_30b_a3b}
MODEL_PATH=${MODEL_PATH:-$QS_WORK/models/Qwen3-VL-30B-A3B}
SCALE_DIR=$RUN/scale_cache
VISION_SAVE=$SCALE_DIR/vision_gptaq.pt
MBQ_SCALE=$SCALE_DIR/mbq_scales.pt
MODE=${MODE:?set MODE=vit_only|llm_only|rtn_probe|bf16|quant}
TASK_FILTER=${TASK_FILTER:-textvqa_val}
OUTPUT_DIR=$RUN/eval/ablation_${MODE}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,5,6}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH="$WORK:$WORK/3rdparty/lmms-eval:${PYTHONPATH:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate quantstage
cd "$WORK"
mkdir -p "$OUTPUT_DIR" "$RUN/logs"
bench_name=$(echo "$TASK_FILTER" | tr 'a-z' 'A-Z' | tr '_' '-')
run_output_dir="$OUTPUT_DIR/${MODE}_${TASK_FILTER}"
mkdir -p "$run_output_dir"
model_args="pretrained=$MODEL_PATH,device_map=auto,use_flash_attention_2=False"
common=(python main.py --model_name qwen3_vl_moe --model_args "$model_args" --tasks "$TASK_FILTER" --batch_size 1 --output_path "$run_output_dir" --log_samples)
case "$MODE" in
  bf16)
    "${common[@]}" --skip_llm_quant ;;
  vit_only)
    "${common[@]}" --vit_method gptaq --vit_wbits 4 --vit_sym --vit_a_bits 8 --load_vision "$VISION_SAVE" --skip_llm_quant ;;
  llm_only)
    "${common[@]}" --method mbq --w_bit 4 --a_bit 16 --w_group 128 --scale_path "$MBQ_SCALE" --pseudo_quant ;;
  quant)
    "${common[@]}" --vit_method gptaq --vit_wbits 4 --vit_sym --vit_a_bits 8 --load_vision "$VISION_SAVE" \
      --method mbq --w_bit 4 --a_bit 16 --w_group 128 --scale_path "$MBQ_SCALE" --pseudo_quant ;;
  rtn_probe)
    # RTN applies without MBQ scales; uses --method rtn + --pseudo_quant (same bits as MBQ).
    # Prefer calibrating first if scales empty; for probe, RTN path quantizes on the fly if supported.
    "${common[@]}" --method rtn --w_bit 4 --a_bit 16 --w_group 128 --pseudo_quant \
      --vit_method gptaq --vit_wbits 4 --vit_sym --vit_a_bits 8 --load_vision "$VISION_SAVE" ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac

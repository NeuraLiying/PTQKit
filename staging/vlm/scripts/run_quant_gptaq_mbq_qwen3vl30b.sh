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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,5,6}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH="$WORK:$WORK/3rdparty/lmms-eval:${PYTHONPATH:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate quantstage
cd "$WORK"
mkdir -p "$SCALE_DIR" "$RUN/logs"
echo "[STATUS] Step1 ViT GPTAQ CUDA=$CUDA_VISIBLE_DEVICES MODEL=$MODEL_PATH" | tee -a "$RUN/logs/quant.log"
python main_quant.py \
  --model_name qwen3_vl_moe \
  --model_args "pretrained=$MODEL_PATH,device_map=auto,use_flash_attention_2=False" \
  --vit_method gptaq \
  --vit_wbits 4 --vit_groupsize 32 --vit_sym \
  --vit_a_bits 8 --vit_a_groupsize -1 \
  --vit_calib_source imagenet \
  --imagenet_root "$QS/datasets/vit_calib" \
  --vit_calib_split . \
  --vit_calib_size 32 \
  --vit_save "$VISION_SAVE" \
  --skip_llm_quant 2>&1 | tee -a "$RUN/logs/step1_vit_gptaq.log"
echo "[STATUS] Step1 done" | tee -a "$RUN/logs/quant.log"
echo "[STATUS] Step2 LLM MBQ" | tee -a "$RUN/logs/quant.log"
python main_quant.py \
  --model_name qwen3_vl_moe \
  --model_args "pretrained=$MODEL_PATH,device_map=auto,use_flash_attention_2=False" \
  --method mbq \
  --calib_data coco \
  --data_path "data/coco.json" \
  --image_folder "$QS/datasets" \
  --n_samples 16 \
  --w_bit 4 --a_bit 16 --w_group 128 \
  --reweight \
  --scale_path "$MBQ_SCALE" \
  --run_process \
  --load_vision "$VISION_SAVE" \
  --vit_method gptaq --vit_wbits 4 --vit_sym --vit_a_bits 8 2>&1 | tee -a "$RUN/logs/step2_mbq.log"
echo "[STATUS] Quant done ViT=$VISION_SAVE MBQ=$MBQ_SCALE" | tee -a "$RUN/logs/quant.log"

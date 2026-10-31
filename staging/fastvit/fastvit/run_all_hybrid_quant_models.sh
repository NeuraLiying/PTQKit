#!/usr/bin/env bash
set -e

MODELS=(
  "fastvit_t8"
  "fastvit_t12"
  "fastvit_s12"
  "fastvit_sa12"
  "fastvit_sa24"
  "fastvit_sa36"
  "fastvit_ma36"
)

for model in "${MODELS[@]}"; do
  echo "========================================"
  echo "Running model: ${model}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=0 \
  python fastvit/hybrid_quant_fastvit.py \
    --skip-fp-eval \
    --model "${model}" \
    --optimize \
    --w_bit 4 \
    --a_bit 4 \
    --quant-act \
    --fastvit-sensitive-w8a8 \
    --k 15 \
    --optim-size 2048
  
done

echo "All models finished."

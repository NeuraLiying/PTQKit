# Phase1 ablation status

Timezone: Asia/Shanghai (CST, UTC+8)
Updated: 2026-10-31 19:08 CST · status=`phase1_ablation_sealed`

| Arm | AI2D | MMMU | RWQA | SciQA | TextVQA | VizWiz | MMBench |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| BF16 | 85.62 | 54.56 | 65.10 | 93.99 | 82.31 | 69.97 | 86.94 |
| Full | 84.91 | 54.00 | 65.75 | 92.78 | 79.00 | 69.07 | 85.14 |
| ViT-only | 85.23 | 54.33 | 64.18 | 93.59 | 79.59 | 69.76 | 85.74 |
| LLM-only | 85.10 | 54.44 | 65.62 | 93.42 | 82.08 | 69.64 | 86.25 |
| RTN | **84.52** | **54.33** | — | — | **78.91** | — | — |

Sealed criteria: llm_only×7 + vit_only×7 + rtn TextVQA+MMMU+AI2D.

- TextVQA: ViT-only 79.59 ≈ Full 79.00; LLM-only 82.08 ≈ BF16 82.31 → drop mostly **ViT-side**.
- RTN AI2D 84.52 vs BF16 85.62 / Full 84.91 / ViT-only 85.23 / LLM-only 85.10.
- RTN TextVQA 78.91 ≈ Full; RTN MMMU 54.33 = ViT-only.
- Patch for RTN `get_blocks` on Qwen3-VL MoE: `staging/vlm/patches/0003-rtn-qwen3vl-moe-get_blocks.diff`.

### Footnote — applies to Full / LLM-only / RTN columns

On transformers **5.21.0** Qwen3-VL-30B-A3B (`decoder_sparse_step=1`), LLM-side RTN/MBQ **pseudo-quant only hits `nn.Linear` in decoder layers ≈ Attn q/k/v/o (~3% of LLM params)**. Packed MoE experts (`gate_up_proj` / `down_proj` Parameters) and router `TopKRouter.weight` are **not** pseudo-quantized; MBQ `auto_scale` on `Qwen3VLMoeTextSparseMoeBlock` is `pass`. Do **not** read these columns as full W4 MoE-expert quantization. RTN vs MBQ is a fair compare **only on that Attn-Linear subset** (same `q_config`: zero_point, group 128) with ViT GPTAQ held fixed for rtn_probe.

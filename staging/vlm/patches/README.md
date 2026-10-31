# Patches applied on `$QS_WORK/GPTAQ-MBQ`

Generated from `.bak_*` vs live files. Apply with `patch -p0` from the GPTAQ-MBQ root
(paths inside diffs are relative to that root).

| Patch | Purpose |
|-------|---------|
| `0001-qwen2_vl-collator-pad-mm_token_type_ids.diff` | Left-pad all 1D seq fields incl. `mm_token_type_ids` |
| `0002-qwen3_vl-deepstack-unpack-and-moe-vocab.diff` | DeepStack `pooler_output` unpack; MoE loss without top-level `vocab_size` |
| `0003-rtn-qwen3vl-moe-get_blocks.diff` | RTN `get_blocks` for `Qwen3VL(Moe)ForConditionalGeneration` → `language_model.layers` |
| `0004-rtn-packed-moe-expert-fakequant.diff` | RTN: fake-quant packed MoE `gate_up_proj`/`down_proj` (+ coverage helpers) |
| `0005-mbq-packed-moe-expert-fakequant.diff` | MBQ: same packed-expert fake-quant in `pseudo_quantize_model_weight` |
| `main_load_quantized_vision_snippet.py` | Resolve `model.visual` or `model.model.visual` |

Scrubbed: no tokens, no client names. Prefer env `QS_WORK` over host home paths in docs/scripts.

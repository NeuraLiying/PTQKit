#!/usr/bin/env python3
"""CLIFF-SPLIT eval: Attn RTN W4 + Expert RTN W3 (ViT BF16 via no vit flags).

Single-variable vs R0 (Attn+Expert RTN W3). Only Attn bitwidth changes.
Env:
  QS_ATTN_W_BIT    default 4
  QS_EXPERT_W_BIT  default 3
  QS_QUANTIZE_MOE_EXPERTS forced to 0; experts handled in this hook.
No ACC / Astra scale / RM-MBQ / GPTQ.
"""
from __future__ import annotations

import os
import runpy
import sys


def _install_hook():
    attn_bit = int(os.environ.get("QS_ATTN_W_BIT", "4"))
    expert_bit = int(os.environ.get("QS_EXPERT_W_BIT", "3"))
    from qmllm.methods.rtn import quantizer as rq
    from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

    orig = rq.pseudo_quantize_model_weight

    def wrapped(model, w_bit, q_config, quantize_moe_experts=None):
        # Attn / nn.Linear only at Attn bit (matches P0-3 W4 Attn path)
        orig(model, attn_bit, q_config, quantize_moe_experts=False)
        # Experts: stock RTN at Expert bit (matches R0 Expert path; no clip)
        layers = rq.get_blocks(model)
        n = 0
        for i in range(len(layers)):
            for _name, p in rq.get_named_moe_expert_params(layers[i]).items():
                p.data = pseudo_quantize_tensor(p.data, n_bits=expert_bit, **q_config)
                n += 1
        print(
            f"[cliff] Attn RTN W{attn_bit} + Expert RTN W{expert_bit} "
            f"n_expert_tensors={n} (ignore main w_bit={w_bit})",
            flush=True,
        )
        if n == 0:
            raise RuntimeError("no moe expert params found for CLIFF-SPLIT Expert RTN")

    rq.pseudo_quantize_model_weight = wrapped
    print(
        f"[cliff] installed Attn-W{attn_bit} + Expert-W{expert_bit} split hook",
        flush=True,
    )


def main():
    _install_hook()
    os.environ["QS_QUANTIZE_MOE_EXPERTS"] = "0"
    # Guard against dead CUDA devices selected by NVML-only pickers
    cvd = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if cvd:
        import torch
        n = torch.cuda.device_count()
        if n != len(cvd):
            raise RuntimeError(
                f"CUDA device_count={n} != len(CVD)={len(cvd)} CVD={cvd}; "
                "likely selected a dead GPU (DeferredCudaCallError risk)"
            )
        print(f"[cliff] CUDA ok device_count={n} CVD={cvd}", flush=True)
    qs = os.environ.get("QS_WORK", "/path/to/quantstage-work")
    main_py = os.path.join(qs, "GPTAQ-MBQ", "main.py")
    sys.argv = [main_py] + sys.argv[1:]
    runpy.run_path(main_py, run_name="__main__")


if __name__ == "__main__":
    main()

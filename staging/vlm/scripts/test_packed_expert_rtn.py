#!/usr/bin/env python3
"""Unit / smoke tests for packed MoE expert RTN fake-quant (P0-3).

Run on the GPU host:
  conda activate quantstage
  cd $QS_WORK/GPTAQ-MBQ
  CUDA_VISIBLE_DEVICES= python scripts_local/test_packed_expert_rtn.py   # CPU
  CUDA_VISIBLE_DEVICES=0 python scripts_local/test_packed_expert_rtn.py --meta-model
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.detach().abs().max().clamp_min(1e-12)
    return float((a - b).abs().max() / denom)


def test_packed_vs_linear_rtn(n_bits=4, q_group_size=128, zero_point=True, seed=0):
    from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

    torch.manual_seed(seed)
    # Match Qwen3-VL-30B-A3B MoE shapes (subset of experts for speed)
    E, hidden, inter = 8, 2048, 768
    gate_up = torch.randn(E, 2 * inter, hidden)
    down = torch.randn(E, hidden, inter)

    q_cfg = dict(zero_point=zero_point, q_group_size=q_group_size)

    packed_gu = pseudo_quantize_tensor(gate_up.clone(), n_bits=n_bits, **q_cfg)
    packed_dn = pseudo_quantize_tensor(down.clone(), n_bits=n_bits, **q_cfg)

    linear_gu = gate_up.clone()
    linear_dn = down.clone()
    for e in range(E):
        # Same op as nn.Linear.weight fake-quant
        linear_gu[e] = pseudo_quantize_tensor(linear_gu[e], n_bits=n_bits, **q_cfg)
        linear_dn[e] = pseudo_quantize_tensor(linear_dn[e], n_bits=n_bits, **q_cfg)
        # Also via nn.Linear path
        lin = nn.Linear(hidden, 2 * inter, bias=False)
        with torch.no_grad():
            lin.weight.copy_(gate_up[e])
            lin.weight.data = pseudo_quantize_tensor(lin.weight.data, n_bits=n_bits, **q_cfg)
        err_lin = _rel_err(packed_gu[e], lin.weight.data)
        assert err_lin < 1e-3, f"gate_up expert {e} vs Linear rel_err={err_lin}"

    e_gu = _rel_err(packed_gu, linear_gu)
    e_dn = _rel_err(packed_dn, linear_dn)
    print(f"[PASS] packed vs per-expert RTN  gate_up rel_err={e_gu:.3e}  down rel_err={e_dn:.3e}")
    assert e_gu < 1e-3 and e_dn < 1e-3
    return e_gu, e_dn


def test_group_axis_divisible():
    # Real shapes must be divisible by g=128 on input dim
    assert 2048 % 128 == 0
    assert 768 % 128 == 0
    print("[PASS] group-size divisibility for real MoE shapes")


def test_meta_model_coverage(model_path: str, min_coverage: float = 0.95):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLMoeForConditionalGeneration
    from qmllm.methods.rtn.quantizer import (
        get_blocks,
        get_named_linears,
        get_named_moe_expert_params,
        llm_quant_coverage_stats,
        pseudo_quantize_model_weight,
    )

    cfg = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        m = Qwen3VLMoeForConditionalGeneration._from_config(cfg)

    layers = get_blocks(m)
    assert len(layers) == 48, len(layers)
    l0 = layers[0]
    linears = sorted(get_named_linears(l0).keys())
    experts = sorted(get_named_moe_expert_params(l0).keys())
    print("layer0 linears:", linears)
    print("layer0 experts:", experts)
    assert linears == ["self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj"], linears
    assert any(x.endswith("gate_up_proj") for x in experts)
    assert any(x.endswith("down_proj") for x in experts)

    stats = llm_quant_coverage_stats(m)
    print("coverage_stats:", {k: stats[k] for k in ("eligible_numel", "covered_numel", "excluded_numel", "coverage")})
    print("details:", stats["details"])
    assert stats["coverage"] >= min_coverage, (
        f"coverage {stats['coverage']:.4f} < {min_coverage} "
        f"(eligible={stats['eligible_numel']}, covered={stats['covered_numel']}, other={stats['details']['other']})"
    )
    print(f"[PASS] meta coverage {stats['coverage']*100:.2f}% >= {min_coverage*100:.0f}%")

    # Smoke: fake-quant on meta tensors is not meaningful; skip apply.
    # Instead materialize one random layer-sized expert block on CPU and apply.
    from qmllm.methods.rtn.quantizer import get_named_moe_expert_params as _g
    # Create a tiny stand-in module
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict({
                "q_proj": nn.Linear(2048, 4096, bias=False),
                "k_proj": nn.Linear(2048, 512, bias=False),
                "v_proj": nn.Linear(2048, 512, bias=False),
                "o_proj": nn.Linear(4096, 2048, bias=False),
            })
            # mimic path experts.gate_up_proj
            self.mlp = nn.Module()
            self.mlp.experts = nn.Module()
            self.mlp.experts.gate_up_proj = nn.Parameter(torch.randn(4, 1536, 2048))
            self.mlp.experts.down_proj = nn.Parameter(torch.randn(4, 2048, 768))
            self.mlp.gate = nn.Module()
            self.mlp.gate.weight = nn.Parameter(torch.randn(4, 2048))

    # Direct apply on params via helper
    from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
    t = Tiny()
    before = t.mlp.experts.gate_up_proj.data.clone()
    router_before = t.mlp.gate.weight.data.clone()
    for n, p in get_named_moe_expert_params(t).items():
        p.data = pseudo_quantize_tensor(p.data, n_bits=4, zero_point=True, q_group_size=128)
    assert not torch.equal(before, t.mlp.experts.gate_up_proj.data)
    assert torch.equal(router_before, t.mlp.gate.weight.data), "router must stay untouched"
    print("[PASS] expert fake-quant mutates experts, leaves router")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-model", action="store_true")
    ap.add_argument(
        "--model",
        default="",  # set via --model or $QS_MODEL_ROOT/Qwen3-VL-30B-A3B
    )
    ap.add_argument("--min-coverage", type=float, default=0.95)
    args = ap.parse_args()

    test_group_axis_divisible()
    test_packed_vs_linear_rtn()
    if args.meta_model:
        model = args.model
        if not model:
            import os
            root = os.environ.get("QS_MODEL_ROOT") or os.environ.get("QS_WORK", "")
            if root and os.path.isdir(os.path.join(root, "models")):
                model = os.path.join(root, "models", "Qwen3-VL-30B-A3B")
            elif root:
                model = os.path.join(root, "Qwen3-VL-30B-A3B")
            else:
                raise SystemExit("Pass --model or set QS_MODEL_ROOT / QS_WORK")
        test_meta_model_coverage(model, min_coverage=args.min_coverage)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

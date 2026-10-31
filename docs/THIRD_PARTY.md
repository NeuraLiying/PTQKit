# Third-party notices

First-party PTQKit code is Apache-2.0 (see `LICENSE` and `NOTICE`).
This document lists upstream projects we depend on, cite, or adapt.

## Algorithms & papers we cite (not vendored as-is)

| Component | Upstream | License / note | How we use it |
|-----------|----------|----------------|---------------|
| **EBSS** (Expert-Balanced Self-Sampling) | MoEQuant paper / codebase | Cite paper; do **not** claim invention | Independent reimplementation in `staging/ebss/` → future `src/ptqkit/calib/ebss/` |
| **GPTQ** | IST-DASLab / Frantar et al. | Apache-2.0 | Hessian / GPTQ primitives planned under `algos/gptq/` |
| **GPTAQ** | Upstream GPTAQ authors | **License to verify** before public release | Vision-side PTQ path (VLM recipe) |
| **AdaRound / QDrop / related PTQ** | Academic lineage used inside FIMA-Q | Follow FIMA-Q NOTICE | Via FIMA-Q FastViT path |

## Software we adapt or will submodule

| Component | Upstream | License | Location today | Target |
|-----------|----------|---------|----------------|--------|
| **FIMA-Q** | [ShiheWang/FIMA-Q](https://github.com/ShiheWang/FIMA-Q) | Apache-2.0 | `staging/fastvit/` (patched FastViT integration) | `third_party/fima-q` fork + thin adapter |
| **MBQ** | [thu-nics/MBQ](https://github.com/thu-nics/MBQ) | MIT | GPU-host `$QS_WORK/GPTAQ-MBQ` + patches in `staging/vlm/patches/` | `third_party/mbq` fork + adapter |
| **llama.cpp** | ggml-org / llama.cpp | MIT | not vendored | optional binary for imatrix / GGUF export |
| **LLaVA-NeXT** | LLaVA project | follow upstream | **not** vendored (no nested `.git` trees) | submodule if needed for VLM eval |
| **lmms-eval** | lmms-lab | follow upstream | not vendored (editable under GPTAQ-MBQ `3rdparty/`) | submodule / pip for VLM eval |
| **timm / FastViT weights** | Hugging Face / timm | model-specific | weights outside repo | `ModelSpec` + HF Hub |

## Staging tree provenance

- `staging/ebss/` — engineering reimplementation of EBSS calibration → JSONL / imatrix (importance matrix) text bridge.
  Score form follows MoEQuant (`score = -(R+logP)/(t+1) + σ/τ`); cite MoEQuant.
- `staging/fastvit/` — FastViT PTQ scripts/layers derived from FIMA-Q (Apache-2.0).
  Keep `staging/fastvit/LICENSE`. File headers should retain upstream attribution
  when migrating into `src/ptqkit/algos/fimaq/`.
- `staging/vlm/` — Phase1 packaging: quant/eval/ablation scripts, docs (`QUANT_SPEC`,
  harness note, ablation plan), and patches `0001`–`0003` applied on the GPU-host
  GPTAQ-MBQ tree. Full MBQ/GPTAQ sources are **not** vendored here (MIT/Apache —
  see NOTICE). Results: `results/vlm-gptaq-mbq/`.

## Data we do **not** redistribute

- ImageNet (ILSVRC) — provide path via `IMAGENET_ROOT`.
- COCO / ShareGPT4V style caption dumps — download scripts only; check licenses.
- Model weights (Qwen, FastViT, …) — HF Hub / local `QS_MODEL_ROOT`.

## Checklist before any public release

1. Confirm license / redistribution clearance for any new third-party code.
2. `./tools/scrub_check.sh` PASS; gitleaks clean.
3. Every `third_party/*` submodule has its LICENSE pinned.
4. EBSS wording is “independent reimplementation”, with MoEQuant citation.
5. GPTAQ upstream license verified and listed here.

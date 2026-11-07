# Third-party notices

First-party PTQKit code is Apache-2.0 (see `LICENSE` and `NOTICE`).
This document lists upstream projects we depend on, cite, or adapt.

## Algorithms & papers we cite (not vendored as-is)

| Component | Upstream | License / note | How we use it |
|-----------|----------|----------------|---------------|
| **EBSS** (Expert-Balanced Self-Sampling) | [MoEQuant](https://arxiv.org/abs/2310.02410) paper (Frantar et al., 2023) | Cite paper; do **not** claim invention | Independent reimplementation in `staging/ebss/` → future `src/ptqkit/calib/ebss/` |
| **GPTQ** | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (Frantar et al.) | Apache-2.0 | Hessian / GPTQ primitives planned under `algos/gptq/` (not yet implemented) |
| **AdaRound / QDrop / related PTQ** | Academic lineage used inside FIMA-Q | Follow FIMA-Q NOTICE | Via FIMA-Q FastViT path |

## Software we adapt or will submodule

| Component | Upstream | License | Location today | Target |
|-----------|----------|---------|----------------|--------|
| **FIMA-Q** | [ShiheWang/FIMA-Q](https://github.com/ShiheWang/FIMA-Q) | Apache-2.0 | `staging/fastvit/` (adapted FastViT integration) | `third_party/fima-q` fork + thin adapter |
| **MBQ** | [thu-nics/MBQ](https://github.com/thu-nics/MBQ) | MIT | **Not included**: External GPU host `$QS_WORK/GPTAQ-MBQ`; patches in `staging/vlm/patches/` | `third_party/mbq` fork + adapter |
| **GPTAQ-for-vision** | External codebase (location TBD) | **License not verified; redistribution status unresolved** | **Not included**: External GPU host `$QS_WORK/GPTAQ-MBQ`; patches in `staging/vlm/patches/` apply to `qmllm/` and `quant/vision/` modules | Source repository and license verification REQUIRED before public release |
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
  harness note, ablation plan), and patches `0001`–`0005` for external GPU-host
  codebase. Patches modify `qmllm/models/`, `qmllm/methods/rtn/`, `qmllm/methods/mbq/`,
  and `quant/vision/gptaq_vision.py` in external environment. Full MBQ/GPTAQ sources
  are **not** vendored here. MBQ is MIT (thu-nics/MBQ, verified). GPTAQ-for-vision
  license and provenance **not verified**; redistribution status unresolved. Results: `results/vlm-gptaq-mbq/`.

## Data we do **not** redistribute

- ImageNet (ILSVRC) — provide path via `IMAGENET_ROOT`.
- COCO / ShareGPT4V style caption dumps — download scripts only; check licenses.
- Model weights (Qwen, FastViT, …) — HF Hub / local `QS_MODEL_ROOT`.

## MBQ Redistribution Notice

MBQ ([thu-nics/MBQ](https://github.com/thu-nics/MBQ)) is licensed under the MIT License:

```
MIT License

Copyright (c) 2023 THU-NICS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

PTQKit does not include MBQ source code. Patches in `staging/vlm/patches/` are derivative works under PTQKit's Apache-2.0 license, intended for use with user-supplied MBQ sources.

## Checklist before any public release

1. Confirm license / redistribution clearance for any new third-party code.
2. `./tools/scrub_check.sh` PASS; gitleaks clean.
3. Every `third_party/*` submodule has its LICENSE pinned.
4. EBSS wording is "independent reimplementation", with MoEQuant citation.
5. **GPTAQ-for-vision source repository and license verification REQUIRED**:
   - Patches apply to external codebase at GPU host `$QS_WORK/GPTAQ-MBQ`
   - Modified modules: `qmllm/models/qwen{2,3}_vl/`, `qmllm/methods/{rtn,mbq}/`, `quant/vision/gptaq_vision.py`
   - **License not verified**; redistribution status unresolved
   - Source repository location unknown (not found on public GitHub/arXiv as standalone project)
   - PTQKit's Apache-2.0 license does **not** grant redistribution rights to third-party-derived patches
   - Cannot release `staging/vlm/` patches without upstream license verification and permission
   - Attribution is **incomplete** until source and license are verified

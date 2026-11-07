# PTQKit

**A library for reproducing and improving post-training quantization (PTQ) methods.**
Multiple method lines (reproduce → ablate → improve). Calibration sets, quant states,
eval reports, and exports are first-class artifacts *(pipeline wiring still scaffold —
reproduce via staging scripts)*.

Repository: https://github.com/NeuraLiying/PTQKit

## Pipeline narrative

```text
Calibration  →  Search / Observe  →  Apply (FakeQuant)  →  Eval  →  Export
  ▲              ▲                    ▲
  │              │                    └─ VLM: vision GPTAQ + language MBQ (Plan)
  │              └─ FastViT: FIMA-Q calibrate + Fisher block reconstruction (Reconstruct)
  └─ MoE: Expert-Balanced Self-Sampling → importance matrix (imatrix) / GGUF (Calibration)
```

Three recipes, one DAG vocabulary — each line highlights a different PTQ lever.

| Lever | Recipe | Role in PTQ |
|-------|--------|-------------|
| **Calibration data** | `recipes/moe-ebss-gguf/` | Cover under-visited MoE experts |
| **Search / reconstruction** | `recipes/fastvit-w4a4/` | W4A4 needs rounding + block reconstruction, not scale-only |
| **Plan / split** | `recipes/vlm-gptaq-mbq/` | Vision vs language sensitivity differs |

## Results

### VLM GPTAQ + MBQ (Phase1, sealed)

Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`. Spec (Phase1): **ViT GPTAQ W4g32A8** + **LLM MBQ W4g128A16** (pseudo-quant). Same lmms-eval harness for BF16 and quant.

**Coverage honesty:** Phase1 LLM MBQ/RTN only fake-quantizes decoder **`nn.Linear` ≈ Attn q/k/v/o**. Packed MoE experts (`gate_up_proj` / `down_proj`) stay BF16. Effective quantized weight share is about **~5%** of total parameters (ViT ≈0.6B + Attn-Linear ≈0.9B); experts ≈28B are not quantized. Do **not** read the Phase1 table as full W4 MoE-expert quantization. On the Attn-Linear subset, MBQ vs RTN is within noise; ViT GPTAQ vs RTN was not compared in Phase1.

| Task | n | BF16 | GPTAQ+MBQ | Δpt |
|------|--:|-----:|----------:|----:|
| AI2D | 3088 | 85.62 | 84.91 | −0.71 |
| MMMU-val | 900 | 54.56 | 54.00 | −0.56 |
| RealWorldQA | 765 | 65.10 | 65.75 | +0.65 |
| ScienceQA | 4241 | 93.99 | 92.78 | −1.21 |
| TextVQA-val | 5000 | 82.31 | 79.00 | −3.31 |
| VizWiz-val | 4319 | 69.97 | 69.07 | −0.90 |
| MMBench-EN-dev | 4329 | 86.94 | 85.14 | −1.80 |

**Mean Δ ≈ −1.1 pt** across the 7 tasks (largest drop on TextVQA).

Ablations (same harness):

| Task | BF16 | Full | ViT-only | LLM-only | RTN |
|------|-----:|-----:|---------:|---------:|----:|
| AI2D | 85.62 | 84.91 | 85.23 | 85.10 | 84.52 |
| MMMU-val | 54.56 | 54.00 | 54.33 | 54.44 | 54.33 |
| RealWorldQA | 65.10 | 65.75 | 64.18 | 65.62 | — |
| ScienceQA | 93.99 | 92.78 | 93.59 | 93.42 | — |
| TextVQA-val | 82.31 | 79.00 | 79.59 | 82.08 | 78.91 |
| VizWiz-val | 69.97 | 69.07 | 69.76 | 69.64 | — |
| MMBench-EN-dev | 86.94 | 85.14 | 85.74 | 86.25 | — |

**Conclusions**

- Phase1 GPTAQ+MBQ holds within ~1 pt mean of same-harness BF16; TextVQA is the main outlier (−3.3 pt).
- TextVQA: ViT-only **79.59** ≈ Full **79.00**; LLM-only **82.08** ≈ BF16 **82.31** → drop is mostly **ViT-side**.
- LLM RTN/MBQ on this build covers **Attn-Linear** only (~3% of LLM params; ~5% of total with ViT); packed MoE experts are not pseudo-quantized (see [`ABLATION_STATUS`](results/vlm-gptaq-mbq/ABLATION_STATUS.md)). Stage2 adds expert coverage probes.
- Absolute MMMU/RWQA levels are **not** comparable to official Qwen tables — harness differs ([`HARNESS_VS_OFFICIAL`](staging/vlm/docs/HARNESS_VS_OFFICIAL.md)).

Full provenance: [`PHASE1_TABLE`](results/vlm-gptaq-mbq/PHASE1_TABLE.md) · [`summary.json`](results/vlm-gptaq-mbq/summary.json).

### FastViT W4A4 (FIMA-Q)

| Model | Setting | Metric | Value |
|-------|---------|--------|------:|
| fastvit_ma36 | W4A4 + fisher_dplr reconstruction, ImageNet val | Prec@1 | **83.23** |
| fastvit_ma36 | W4A4 + fisher_dplr reconstruction, ImageNet val | Prec@5 | 96.45 |
| fastvit_ma36 | W4A4 after MSE calibration only | Prec@1 | 72.36 |

Recipe: [`recipes/fastvit-w4a4/ma36.yaml`](recipes/fastvit-w4a4/ma36.yaml). Summary: [`results/fastvit-w4a4/summary.json`](results/fastvit-w4a4/summary.json).

### MoE EBSS → GGUF (Qwen3.5-35B-A3B)

Held-out WikiText-2 PPL (`llama-perplexity`, 580×512):

| Variant | PPL | ±stderr |
|---------|----:|--------:|
| bf16 (ref) | **6.5339** | 0.0416 |
| Q4_K_M + EBSS imatrix (importance matrix) | **6.6246** | 0.0424 |
| Q4_K_M + random imatrix (importance matrix) | **6.6206** | 0.0423 |

Expert-coverage σ (64×512 smoke run): EBSS 0.0060 · random 0.0018. Details: [`results/moe-ebss-gguf/`](results/moe-ebss-gguf/).

## Attribution

- **EBSS** algorithm is from **MoEQuant** (Frantar et al., 2023). This repo ships an **independent reimplementation** plus engineering. We do **not** claim to have proposed EBSS.
- **FIMA-Q** ([Apache-2.0](https://github.com/ShiheWang/FIMA-Q)) code adapted in `staging/fastvit/`.
- **MBQ** ([MIT](https://github.com/thu-nics/MBQ)) and **GPTAQ-for-vision** (license TBD) are **not included**; `staging/vlm/` contains patches for external sources only.

See [`NOTICE`](NOTICE) and [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) for complete licensing information.

## 30-minute get-started

Prereqs: Python ≥3.10. Model weights and ImageNet stay **outside** the repo (`QS_MODEL_ROOT`, `IMAGENET_ROOT`). Full VLM/EBSS reproduction needs a GPU host + conda `quantstage` — see [`docs/REPRODUCE.md`](docs/REPRODUCE.md) and [`recipes/env/`](recipes/env/).

```bash
git clone https://github.com/NeuraLiying/PTQKit.git ptqkit && cd ptqkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add extras: moe | vision | vlm when running those lines

# Scrub / compliance gate (must PASS before any push)
./tools/scrub_check.sh

# CLI scaffold
qs version
qs run recipes/moe-ebss-gguf/qwen35-35b-a3b.yaml   # returns 2 until DAG lands
```

**Until `qs run` is wired**, use staging scripts (examples):

```bash
export QS_MODEL_ROOT=/path/to/models
python staging/ebss/generate_calib.py \
  --model "${QS_MODEL_ROOT}/Qwen3.5-35B-A3B" \
  --out /tmp/ebss_smoke.pt --out_jsonl /tmp/ebss_smoke.jsonl \
  --num_seqs 64 --seqlen 512 --w 4 --device cuda

python staging/fastvit/fastvit/test_quant_fastvit.py \
  --model fastvit_ma36 --config staging/fastvit/configs/4bit/best.py \
  --dataset "${IMAGENET_ROOT}" --calibrate --quant-act --optimize
```

VLM Phase1 scripts/patches: [`staging/vlm/`](staging/vlm/). Results: [`results/vlm-gptaq-mbq/`](results/vlm-gptaq-mbq/).

## Layout

```text
src/ptqkit/     core Stage/Pipeline/store/fingerprint/manifest + CLI (scaffold)
recipes/        declarative DAGs + env/ conda pins
staging/        pre-migration working code (ebss, fastvit, vlm)
results/        small summary.json + sealed tables (no raw dumps / weights)
docs/           THIRD_PARTY, REPRODUCE
tools/scrub_*   compliance gate
```

## Constraints

- No `.pth` / `.pt` / `.dat` / `.gguf` in git.
- No Hugging Face tokens in tree or commit messages — use `huggingface-cli login` or env `HF_TOKEN`.
- Host paths via env / recipe only (`QS_WORK`, `QS_MODEL_ROOT`, `IMAGENET_ROOT`).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Before pushing: `./tools/scrub_check.sh` must PASS (no tokens, home paths, or weight dumps).

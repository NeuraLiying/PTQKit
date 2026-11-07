# Stage2 CLIFF-SPLIT Results (Qwen3-VL-30B-A3B)

**Status**: Observational evidence from external environment; not freshly reproduced in this tree.

## Configuration

| Component | R0 Baseline | CLIFF (Stage2) | Notes |
|-----------|-------------|----------------|-------|
| **ViT** | BF16 | BF16 | No change |
| **Attention** | RTN W3g128 | RTN W4g128 | **+1 bit** |
| **MoE Experts** | RTN W3g128 | RTN W3g128 | No change |
| **Activation** | 16-bit | 16-bit | No change |
| **Quantization** | Pseudo/fake | Pseudo/fake | No memory/speedup claim |

**NOT same-bpw comparison**: Attention changes W3→W4. This is a **mixed-precision diagnostic**, not an equal-budget method improvement.

## Results (Full Tasks, Zero-Shot)

| Task | Split | n | R0 (W3 Attn) | CLIFF (W4 Attn) | Δ pp | Metric |
|------|-------|--:|-------------:|----------------:|-----:|--------|
| TextVQA | validation | 5000 | 77.24 | **79.07** | **+1.832** | exact_match,none |
| MMMU | validation | 900 | 49.44 | **50.89** | **+1.445** | mmmu_acc,none |
| AI2D | test | 3088 | 82.16 | **84.36** | **+2.202** | exact_match,flexible-extract |

**Mean improvement**: +1.826 pp across 3 tasks.

### Raw Scores (from server export)
- TextVQA: 0.7724000000000034 → 0.7907200000000038
- MMMU: 0.49444 → 0.50889
- AI2D: 0.8215673575129534 → 0.8435880829015544

## Limitations & Caveats

**Comparison validity**:
- ✗ NOT same bits-per-weight: W3→W4 attention adds capacity
- ✗ NOT matched throughput: R0 used `max_memory_per_gpu=11511MiB`; CLIFF omitted it
- ✗ Evaluation times are metadata, NOT controlled benchmarks
- ✓ Same harness (lmms-eval), same tasks, same random seeds

**Quantization scope**:
- Pseudo/fake quantization only (accuracy study)
- No packed kernels, no deployment acceleration
- No measured memory benefit or speedup
- Parameter allocation numel not collected (log confirms 96 expert tensors observed)

**Model provenance**:
- Model: Qwen/Qwen3-VL-30B-A3B-Instruct (exact revision/hash not captured)
- No calibration in CLIFF queue (RTN path)
- Both configs: `device_map=auto`, `use_flash_attention_2=False`, `batch_size=1`

## Source Evidence

**Included in bundle**:
- Raw result JSONs (6 files: R0×3 + CLIFF×3, path-sanitized)
- CLIFF evaluation script (`cliff_split_eval_main.py`)
- Queue configuration (attention=4, experts=3, group=128, method=rtn, a_bit=16)
- Observation summary with hash manifest

**NOT included**:
- Model weights
- Calibration data
- Sample outputs
- Private server logs/paths

## Reproducibility

**Existing command recipe** (recorded, NOT executed during bundle creation):
```bash
export QS_WORK=/path/to/quantstage-work
export MODEL_PATH=$QS_WORK/models/Qwen3-VL-30B-A3B

# Activate environment
conda activate quantstage

# Run CLIFF evaluation (from existing server script)
bash scripts/run_stage2_cliff_split_queue.sh
```

Script sets: `--method rtn --a_bit 16 --pseudo_quant --batch_size 1 --log_samples`  
Attention: `--w_bit 4 --w_group 128`  
Experts: `--w_bit 3 --w_group 128`

**Requirements**:
- GPU environment with quantstage conda env (see `recipes/env/`)
- Model weights at `$MODEL_PATH`
- lmms-eval harness (editable install under GPTAQ-MBQ/3rdparty/)
- No fresh reproduction performed; existing results presented as evidence only

## Attribution & License Status

**Source code provenance**: External GPTAQ-MBQ codebase on GPU host  
**License status**: **UNRESOLVED**
- No git origin or top-level LICENSE found in export
- MBQ component: MIT (https://github.com/thu-nics/MBQ, verified)
- GPTAQ-for-vision component: **license not verified**
- PTQKit's Apache-2.0 does NOT grant redistribution rights to unverified third-party sources

**Cannot release full source until upstream license verification.**

## References

- Observation summary: [`cliff_results_observed.json`](stage2/cliff_results_observed.json)
- Phase1 results: [`PHASE1_TABLE.md`](PHASE1_TABLE.md)
- Quant spec: [`../staging/vlm/docs/QUANT_SPEC.md`](../../staging/vlm/docs/QUANT_SPEC.md)

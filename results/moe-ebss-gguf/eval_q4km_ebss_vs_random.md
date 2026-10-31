# Eval: Q4_K_M EBSS vs Random (+ bf16 ref) — Qwen3.5-35B-A3B

Phase1 benchmark scores for MoE path (gpu-host / conda `quantstage`).
Times are **Asia/Singapore (UTC+8)** on the GPU-host clock.

## Setup

| item | value |
|------|-------|
| host | gpu-host |
| conda | `quantstage` (torch 2.7.0+cu126, transformers 5.21.0) |
| llama.cpp | `0.8.0-dev` build 1, commit `7df8b87`, CUDA ARCHS=860 |
| binary | `$QS_WORK/llama.cpp/build/bin/llama-perplexity` |
| GPUs | Q4: single-GPU `-ngl 99`; bf16: multi-GPU as available |
| corpus | **WikiText-2 raw test** `eval_data/wikitext-2-raw/wiki.test.raw` (1 290 590 bytes, sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`) |
| held-out | yes — overlap probe vs `runs/imatrix/calib_{ebss,random}.txt`: **0/100** 100-char windows |
| PPL args | `-c 512 -b 512 -ub 512 -ngl 99 --chunks -1` → **580 chunks** |

**Not** the calibration set. Do not confuse **imatrix-PPL** (measured on each method's own calib text during `llama-imatrix`) with **eval PPL** below.

Coverage comparison: `runs/compare_ebss_vs_random_qwen35.md` — at 64×512 smoke, Random σ 0.0018 / EBSS σ 0.0060.

## WikiText-2 test perplexity (held-out)

| model | quant / calib | PPL | ± | tokens (HF tok) | chunks | wall (UTC+8) | log |
|-------|---------------|----:|--:|----------------:|-------:|-------------:|-----|
| Qwen3.5-35B-A3B | **bf16** (ref) | **6.5339** | 0.04157 | 297193 | 580 | 3:07.75 | `runs/eval_ppl/ppl_bf16.log` |
| Qwen3.5-35B-A3B | Q4_K_M + **EBSS** imatrix | **6.6246** | 0.04242 | 297193 | 580 | 1:42.57 | `runs/eval_ppl/ppl_q4km_ebss.log` |
| Qwen3.5-35B-A3B | Q4_K_M + **random** imatrix | **6.6206** | 0.04225 | 297193 | 580 | 1:44.10 | `runs/eval_ppl/ppl_q4km_random.log` |

Δ(Q4 EBSS − Q4 random) = **+0.0040** PPL (within ±0.04 stderr).
Δ(Q4 random − bf16) ≈ +0.087; Δ(Q4 EBSS − bf16) ≈ +0.091.

GGUF paths (outside git):
- `gguf/Qwen3.5-35B-A3B-bf16.gguf` (~67G)
- `gguf/Qwen3.5-35B-A3B-Q4_K_M-ebss.gguf` (~21G, BPW 4.89)
- `gguf/Qwen3.5-35B-A3B-Q4_K_M-random.gguf` (~21G, BPW 4.89)

## imatrix-PPL (calib text only — not comparable across methods)

From `runs/imatrix/imatrix_{ebss,random}.log` (reported by `llama-imatrix` on each calib corpus):

| calib | imatrix-PPL | note |
|-------|------------:|------|
| EBSS 64×512 | 1.4441 ± 0.02080 | easier/shorter self-sampled text |
| random 64×512 | 4.2701 ± 0.07737 | different text difficulty |

**Do not** treat imatrix-PPL gap as a quality ranking. Eval PPL on shared WikiText-2 is the comparable score.

## Generation smoke (sanity, temp=0)

Prompt: `The capital of France is` · `-n 48 --single-turn -no-cnv --temp 0`

| variant | output (greedy) | gen tok/s |
|---------|-----------------|----------:|
| Q4 EBSS | starts with ` Paris.` then repeats the sentence | ~164 |
| Q4 random | same pattern | ~164 |

Logs: `runs/eval_ppl/gen_q4km_{ebss,random}.{out,err}`.

## Commands (repro)

```bash
conda activate quantstage
cd $QS_WORK
BIN=llama.cpp/build/bin
EVAL=eval_data/wikitext-2-raw/wiki.test.raw
export LD_LIBRARY_PATH=$BIN:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 $BIN/llama-perplexity \
  -m gguf/Qwen3.5-35B-A3B-Q4_K_M-ebss.gguf -f $EVAL \
  -c 512 -b 512 -ub 512 -ngl 99 --chunks -1

CUDA_VISIBLE_DEVICES=1 $BIN/llama-perplexity \
  -m gguf/Qwen3.5-35B-A3B-Q4_K_M-random.gguf -f $EVAL \
  -c 512 -b 512 -ub 512 -ngl 99 --chunks -1

CUDA_VISIBLE_DEVICES=0,1,2 $BIN/llama-perplexity \
  -m gguf/Qwen3.5-35B-A3B-bf16.gguf -f $EVAL \
  -c 512 -b 512 -ub 512 -ngl 99 --chunks -1
```

## Phase1 takeaway

1. Full MoE pipeline artifacts exist: model → EBSS/random calib → imatrix → Q4_K_M GGUF → **held-out WikiText-2 PPL** + gen smoke.
2. At smoke scale, EBSS and random Q4 PPL are within stderr; coverage σ favors random (0.0018 vs 0.0060).
3. Q4 is ~0.09 PPL above bf16 on this corpus — sane quantization gap.
4. Phase2 optimization is not started here; this report closes Phase1 benchmark scores for the MoE line.

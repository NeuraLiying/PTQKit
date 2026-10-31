> PTQKit staging copy of FastViT PTQ code derived from **FIMA-Q** (Apache-2.0). Paths must use env vars / CLI args — no host absolutes. Weights stay out of git. See `docs/THIRD_PARTY.md`.
> **V1 status:** ImageNet W4A4 numbers packaged; **not** a V1 optimization target. See `results/fastvit-w4a4/`.

# FastViT W4A4 (FIMA-Q staging)

FIMA-Q-based FastViT quantization tooling: **calibration** and **block reconstruction**.

## Requirements

- Python 3.10+, PyTorch, torchvision, timm
- ImageNet at `IMAGENET_ROOT` (not shipped in git)
- See `recipes/env/` for the GPU-host conda pins

## Running `test_quant_fastvit.py`

Entry point:

```bash
python fastvit/test_quant_fastvit.py [options]
```

### Basic usage

**Full pipeline: calibrate → validate → optimize → validate**

```bash
python fastvit/test_quant_fastvit.py \
  --model fastvit_ma36 \
  --config configs/4bit/best.py \
  --dataset "${IMAGENET_ROOT}" \
  --calibrate --quant-act --optimize
```

**Load an optimized checkpoint and validate**

```bash
python fastvit/test_quant_fastvit.py \
  --model fastvit_ma36 \
  --config configs/4bit/best.py \
  --dataset "${IMAGENET_ROOT}" \
  --load-optimize-checkpoint /path/to/checkpoint.pth \
  --test-optimize-checkpoint
```

### Common flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | fastvit_t8 | timm model name (e.g. fastvit_t8 / fastvit_t12 / fastvit_s12) |
| `--config` | configs/4bit/best.py | Quantization config path |
| `--dataset` | /path/to/ImageNet_ILSVRC2012/ | ImageNet dataset root |
| `--val-batch-size` | 200 | Validation batch size |
| `--num-workers` | 8 | DataLoader workers |
| `--device` | cuda | Device (`cuda` / `cuda:0` / `cpu`) |
| `--skip-fp-eval` | - | Skip FP baseline ImageNet validation |
| `--quant-act` | False | Quantize activations (default is weight-only W4A32) |
| `--checkpoint` | None | Local FP pretrained weights for init |

### Calibration

| Flag | Meaning |
|------|---------|
| `--calibrate` | Run calibration |
| `--load-calibrate-checkpoint <path>` | Load a calibrated checkpoint (mutually exclusive with `--calibrate`) |
| `--test-calibrate-checkpoint` | Validate immediately after loading a calibration checkpoint |
| `--calib-size` | Calibration set size (config default if omitted) |
| `--calib-batch-size` | Calibration batch size |
| `--calib-metric` | Calibration metric: mse / mae |

### Block reconstruction

| Flag | Default | Meaning |
|------|---------|---------|
| `--optimize` | - | Run block reconstruction |
| `--load-optimize-checkpoint <path>` | - | Load an optimized checkpoint (mutually exclusive with `--optimize`) |
| `--test-optimize-checkpoint` | - | Validate after loading an optimized checkpoint |
| `--optim-size` | 1024 | Calibration-set size used for optimization |
| `--optim-batch-size` | - | Optimization batch size |
| `--optim-metric` | - | fisher_brecq / fisher_lr / fisher_diag / fisher_dplr / mse / mae |
| `--optim-mode` | - | qinp / rinp / qdrop |
| `--recon-iters` | 20000 | Block-reconstruction iterations |
| `--recon-batch-size` | 32 | Block-reconstruction batch size |
| `--recon-lr` | 4e-5 | Block-reconstruction learning rate |
| `--recon-weight` | 0.01 | Rounding regularization weight |
| `--recon-block-limit` | 0 | Reconstruct only the first N blocks (0 = all) |

### Other optional flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--w_bit` | - | Weight bits (overridden by config) |
| `--a_bit` | - | Activation bits (overridden by config) |
| `--drop-prob` | - | qdrop drop probability |
| `--k` | - | Fisher rank |
| `--p1`, `--p2` | - | Fisher low-rank / diagonal ratios |
| `--dis-mode` | - | Gradient mode: q / qf |
| `--print-freq` | 10 | Log print frequency |
| `--seed` | 3407 | Random seed |

### Output directory

Outputs go under `checkpoints/<subdir>/<model>-<timestamp>/`. Subdir is chosen from the quant config:

- `weight-only`: weight-only (W4A32, default)
- `W4A8`: W4A8
- `W4A4`: W4A4
- otherwise: `W4A{a_bit}`

Logs and checkpoints live in that directory. Packaged ImageNet numbers: `results/fastvit-w4a4/`. Recipe: `recipes/fastvit-w4a4/ma36.yaml`.

# Why same-harness BF16 MMMU ~54.6 / RealWorldQA ~65 ≠ published ~74 / ~77

Our **BF16** and **GPTAQ+MBQ** rows share one lmms-eval path. They are comparable **to each other**. They are **not** official Qwen3-VL leaderboard numbers.

## Likely drivers (ordered)

1. **`max_new_tokens` default = 128** in `lmms_eval/models/qwen3_vl.py` (`gen_kwargs.setdefault("max_new_tokens", 128)`). Official / Thinking-style MMMU runs often allow much longer generations (CoT). Truncation alone can crush multi-step MMMU.
2. **Instruct checkpoint, no Thinking mode.** We load `Qwen3-VL-30B-A3B-Instruct` without thinking/enable-thinking prompts. Public tables that quote ~74 MMMU are frequently Thinking or differently prompted.
3. **Answer extraction / task YAML** — lmms-eval `mmmu_val` flexible extract vs official scripts; MMBench uses `gpt_eval_score` path in our logs.
4. **`max_pixels` unset** — processor default, not necessarily the official high-res setting used in reports. OCR-heavy tasks (and some MMMU diagrams) are resolution-sensitive.
5. **Harness / seed / batch** — `batch_size=1`, `temperature=0`, empty `gen_kwargs` in results.json config.

## What we claim instead

- Relative retention vs **this** BF16 baseline (mean ≈ −1.1 pt on 7 tasks).
- Explicitly **do not** claim “near official accuracy.”

## Follow-up (P1, not blocking packaging)

Re-run BF16 MMMU with official-ish settings (higher `max_new_tokens`, documented `max_pixels`, Thinking if that is the published row) and only then compare absolute levels.

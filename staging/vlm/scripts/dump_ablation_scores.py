#!/usr/bin/env python3
"""Scan ablation eval dirs and dump scores JSON for Mac ingest.

Requires env QS_WORK pointing at the GPU-host quantstage-work root
(contains runs/vlm_qwen3vl_30b_a3b/eval/...).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

QS_WORK = pathlib.Path(os.environ.get("QS_WORK", ".")).expanduser().resolve()
RUN = QS_WORK / "runs" / "vlm_qwen3vl_30b_a3b"
EVAL = RUN / "eval"

TASK_MAP = {
    "ai2d": "ai2d",
    "mmmu_val": "mmmu_val",
    "mmmu": "mmmu_val",
    "realworldqa": "realworldqa",
    "scienceqa": "scienceqa",
    "textvqa_val": "textvqa_val",
    "textvqa": "textvqa_val",
    "vizwiz_vqa_val": "vizwiz_vqa_val",
    "vizwiz": "vizwiz_vqa_val",
    "mmbench_en_dev": "mmbench_en_dev",
    "mmbench": "mmbench_en_dev",
}


def infer_task(name: str):
    n = name.lower()
    for k, v in TASK_MAP.items():
        if k in n:
            return v
    return None


def main() -> int:
    if not RUN.exists():
        print(f"missing run dir: {RUN} (set QS_WORK)", file=sys.stderr)
        return 1

    out = {"llm_only": {}, "vit_only": {}, "rtn_probe": {}, "vit_only_archive": {}}

    for arm in ("llm_only", "vit_only", "rtn_probe"):
        root = EVAL / f"ablation_{arm}"
        if not root.exists():
            continue
        for rj in root.rglob("*results.json"):
            if "results.json" not in rj.name:
                continue
            task = None
            for part in rj.parts:
                if part.startswith(arm + "_"):
                    task = infer_task(part[len(arm) + 1 :])
                    break
            if task is None:
                task = infer_task(rj.as_posix())
            if task is None:
                continue
            data = json.loads(rj.read_text())
            run_id = rj.stem.replace("_results", "")
            out[arm][task] = {
                "path": str(rj.relative_to(RUN)),
                "run_id": run_id,
                "results": data,
            }

    arch = EVAL / "gptaq_mbq_vit_only_or_partial_20261030_024202"
    if arch.exists():
        for rj in arch.rglob("*results.json"):
            task = infer_task(rj.as_posix())
            if not task:
                continue
            data = json.loads(rj.read_text())
            out["vit_only_archive"][task] = {
                "path": str(rj.relative_to(RUN)),
                "run_id": rj.stem.replace("_results", ""),
                "results": data,
            }
            if task not in out["vit_only"]:
                out["vit_only"][task] = out["vit_only_archive"][task]

    for arm, tasks in out.items():
        for t, info in tasks.items():
            res = info["results"].get("results") or {}
            tk = next(iter(res), None)
            if not tk:
                continue
            m = res[tk]
            val = None
            for k in (
                "exact_match,flexible-extract",
                "exact_match,none",
                "mmmu_acc,none",
                "gpt_eval_score,none",
            ):
                if k in m and m[k] not in (None, "N/A"):
                    val = m[k]
                    break
            print(f"{arm:18s} {t:16s} {val}", file=sys.stderr)

    json.dump(out, sys.stdout)
    print(file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

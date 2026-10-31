"""Content-addressed fingerprints for stages (scaffold)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(
    stage_name: str,
    stage_version: str,
    cfg: Mapping[str, Any],
    cache_keys: Sequence[str],
    input_fps: Iterable[str],
) -> str:
    """sha256(stage, version, cfg ∩ cache_keys, sorted input fingerprints)."""
    filtered = {k: cfg[k] for k in cache_keys if k in cfg}
    payload = {
        "stage": stage_name,
        "version": stage_version,
        "cfg": filtered,
        "inputs": sorted(input_fps),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

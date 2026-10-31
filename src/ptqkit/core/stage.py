"""Stage protocol and artifact references.

Thin scaffold: real implementations land in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed handle to a stage output."""

    kind: str  # calib_set | quant_state | qmodel | eval_report | export
    fingerprint: str
    path: Path


class Stage(Protocol):
    """One node in a PTQKit pipeline DAG."""

    name: str
    version: str
    inputs: Mapping[str, str]
    output_kind: str
    cache_keys: Sequence[str]

    def run(
        self,
        ctx: object,
        cfg: Mapping[str, object],
        inputs: Mapping[str, ArtifactRef],
    ) -> ArtifactRef: ...

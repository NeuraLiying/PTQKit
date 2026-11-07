"""Pipeline DAG runner (scaffold).

Caches by fingerprint; skips stages whose artifacts already exist.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class Pipeline:
    """Ordered list of stages driven by a recipe YAML."""

    def __init__(self, stages: Sequence[Any], store: Any) -> None:
        self.stages = list(stages)
        self.store = store

    def run(self, cfg: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute stages; return mapping of stage name -> ArtifactRef.

        Not implemented in this scaffold — see recipes/ and staging/.
        """
        raise NotImplementedError(
            "Pipeline.run is scaffold-only; use staging scripts or `qs run` once Phase 1 lands."
        )

    def status(self, cfg: Mapping[str, Any]) -> Mapping[str, str]:
        """Return cached / stale / missing per stage (scaffold)."""
        raise NotImplementedError("Pipeline.status is scaffold-only.")

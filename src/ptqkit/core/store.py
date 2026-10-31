"""Artifact store under .qs_cache/ (scaffold)."""

from __future__ import annotations

from pathlib import Path


class ArtifactStore:
    """Content-addressed artifact layout: .qs_cache/<kind>/<fingerprint>/."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(".qs_cache")

    def path_for(self, kind: str, fingerprint: str) -> Path:
        return self.root / kind / fingerprint

    def exists(self, kind: str, fingerprint: str) -> bool:
        return self.path_for(kind, fingerprint).exists()

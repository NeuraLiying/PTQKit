"""Recipe YAML loading helpers (scaffold)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

_ENV_PATTERN = re.compile(r"\$\{env:([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_recipe(path: str | Path) -> Mapping[str, Any]:
    """Load a recipe YAML and expand ${env:VAR} placeholders."""
    if yaml is None:
        raise ImportError("PyYAML is required to load recipes; pip install pyyaml")
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"recipe must be a mapping: {path}")
    return _expand_env(data)

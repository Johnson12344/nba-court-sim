from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    config_path = path or PROJECT_ROOT / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return deepcopy(loaded)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


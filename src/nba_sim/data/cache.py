from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.root / f"{safe}.json"

    def save(self, key: str, payload: Any, metadata: dict[str, Any] | None = None) -> Path:
        target = self.path(key)
        if target.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = self.root / f"{target.stem}_{stamp}.json"
        envelope = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "payload": payload,
        }
        target.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        return target

    def load(self, key: str) -> dict[str, Any] | None:
        target = self.path(key)
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))


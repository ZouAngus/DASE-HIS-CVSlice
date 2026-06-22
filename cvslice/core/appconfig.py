"""Tiny persistent app config (last-opened directories etc.).

Stored as a single JSON in the user's home dir so it survives across runs and is
shared by both the ClipAnnotator and the Skeleton Corrector. Best-effort: any
read/write error is swallowed (the feature is a convenience, never critical).
"""
from __future__ import annotations

import json
import os

_PATH = os.path.join(os.path.expanduser("~"), ".cvslice_config.json")


def _load() -> dict:
    try:
        with open(_PATH, encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def get_dir(key: str) -> str | None:
    """Return the cached directory for *key* if it still exists, else None."""
    d = _load().get(key)
    return d if isinstance(d, str) and os.path.isdir(d) else None


def set_dir(key: str, path: str) -> None:
    """Persist *path* as the last directory for *key*."""
    if not path:
        return
    try:
        cfg = _load()
        cfg[key] = path
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

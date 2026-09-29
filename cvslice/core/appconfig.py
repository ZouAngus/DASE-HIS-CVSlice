"""Tiny persistent app config (last-opened directories etc.).

Stored as a single JSON in the user's home dir so it survives across runs and is
shared by both the ClipAnnotator and the Skeleton Corrector. Best-effort: any
read/write error is swallowed (the feature is a convenience, never critical).
"""
from __future__ import annotations

import json
import os
import tempfile

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


def get_str(key: str, default: str = "") -> str:
    """Return a plain string setting (no existence check), or *default*."""
    v = _load().get(key)
    return v if isinstance(v, str) else default


def set_str(key: str, value: str) -> None:
    """Persist a plain string setting."""
    set_value(key, value)


def get_value(key: str, default=None):
    """Read a JSON setting (e.g. a group of UI preferences)."""
    return _load().get(key, default)


def set_value(key: str, value) -> None:
    """Replace the config atomically so interrupted writes keep the old file."""
    tmp = None
    try:
        cfg = _load()
        cfg[key] = value
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=os.path.dirname(_PATH), delete=False) as f:
            tmp = f.name
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _PATH)
    except Exception:
        pass
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def set_dir(key: str, path: str) -> None:
    """Persist *path* as the last directory for *key*."""
    if not path:
        return
    set_value(key, path)

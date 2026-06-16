"""Annotation (offset) persistence as JSON.

This is the rate-unified ("fixed") tool: it locks one video->points rate per
scene, so its view offsets are NOT interchangeable with the original CVSlice's
per-camera ones. To avoid corrupting annotations made by the original tool, it
persists to a separate ``*_annotations_fixed.json`` file and never writes the
original ``*_annotations.json``. On first load it falls back to the original
file (read-only); ``main_window`` then migrates the per-scene view offsets to
the unified rate and saves them to the fixed file.

A scene's presence of a ``"pfps"`` key marks it as already saved/migrated by
this tool (so the migration runs at most once).
"""
import os
import json


def annotations_path(xlsx_path: str) -> str:
    """Return the JSON annotations path used by the rate-fixed tool."""
    base = os.path.splitext(xlsx_path)[0]
    return base + "_annotations_fixed.json"


def legacy_annotations_path(xlsx_path: str) -> str:
    """Return the original CVSlice annotations path (read-only fallback)."""
    base = os.path.splitext(xlsx_path)[0]
    return base + "_annotations.json"


def load_annotations(xlsx_path: str) -> dict:
    """Load annotations from the fixed JSON, falling back to the legacy file.

    Scenes loaded from the legacy file lack a ``"pfps"`` key in their scene
    data, which is what marks them for one-time view-offset migration.
    """
    fixed = annotations_path(xlsx_path)
    legacy = legacy_annotations_path(xlsx_path)
    for p in (fixed, legacy):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if p == legacy:
                    print(f"[fixed] no {os.path.basename(fixed)} yet; loaded "
                          f"legacy annotations from {os.path.basename(p)} "
                          f"(view offsets will be migrated to the unified rate)")
                return data
            except Exception:
                pass
    return {}


def save_annotations(xlsx_path: str, data: dict) -> None:
    """Write annotations dict to the fixed JSON (legacy file untouched)."""
    p = annotations_path(xlsx_path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: failed to save annotations: {e}")

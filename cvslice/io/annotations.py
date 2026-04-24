"""Annotation (offset) persistence as JSON."""
import os
import json


def annotations_path(xlsx_path: str) -> str:
    """Return the JSON annotations path derived from the Excel path."""
    base = os.path.splitext(xlsx_path)[0]
    return base + "_annotations.json"


def load_annotations(xlsx_path: str) -> dict:
    """Load annotations from JSON. Returns empty dict on failure."""
    p = annotations_path(xlsx_path)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_annotations(xlsx_path: str, data: dict) -> None:
    """Write annotations dict to JSON."""
    p = annotations_path(xlsx_path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: failed to save annotations: {e}")

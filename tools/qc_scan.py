"""Batch QC scan: score every clip of an actor/export folder, worst first.

Writes a ``qc_report.json`` into each scene folder; the Skeleton Corrector
picks it up automatically (score column, worst-first sort, N-key navigation).
Diagnostic only — never modifies skeleton data.

Usage:
    python tools/qc_scan.py <actor_or_scene_folder> [--mosh <mosh_dir>]

The folder convention matches the corrector: a scene folder holds the
exported ``*.csv`` per action; an actor folder holds scene subfolders.
With --mosh, matching ``*_stageii.pkl`` joint arrays are scored instead of
the 37-marker CSVs (score what you edit).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvslice.io.discovery import load_csv_as_pts3d, load_mosh_pkl, mosh_pkl_kind
from cvslice.vision import qc

DEFAULT_FPS = 60.0
REPORT_NAME = "qc_report.json"


def find_scenes(folder: str) -> list[str]:
    """Scene folders: the folder itself if it holds CSVs, else subfolders."""
    def has_csv(p):
        try:
            return any(f.lower().endswith(".csv") for f in os.listdir(p))
        except OSError:
            return False
    if has_csv(folder):
        return [folder]
    subs = [os.path.join(folder, d) for d in sorted(os.listdir(folder))
            if os.path.isdir(os.path.join(folder, d))]
    return [s for s in subs if has_csv(s)]


def index_mosh(mosh_dir: str | None) -> dict[str, str]:
    """stem (minus _stageii) -> pkl path, recursive."""
    if not mosh_dir or not os.path.isdir(mosh_dir):
        return {}
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(mosh_dir):
        for fn in files:
            if fn.endswith("_stageii.pkl"):
                out[fn[:-len("_stageii.pkl")]] = os.path.join(root, fn)
    return out


def load_clip(csv_path: str, pkl: str | None):
    """Prefer the mosh joint array (what gets edited); fall back to the CSV.
    Returns (pts3d (T, J, 3), fps, source_label) or None."""
    if pkl:
        try:
            if mosh_pkl_kind(pkl) == "joints":
                pts, fps = load_mosh_pkl(pkl, "joints")
                if pts is not None:
                    return pts, (fps or DEFAULT_FPS), "mosh_joints"
        except Exception as e:                       # noqa: BLE001
            print(f"  ! pkl 加载失败 {os.path.basename(pkl)}: {e}")
    pts, _valid, _was_nan, fps = load_csv_as_pts3d(csv_path)
    if pts is None:
        return None
    return pts, (fps or DEFAULT_FPS), "csv"


def scan_scene(scene: str, mosh_index: dict[str, str]) -> dict:
    csvs = sorted(f for f in os.listdir(scene) if f.lower().endswith(".csv")
                  and not f.endswith("_corrected.csv"))
    report: dict = {"_meta": {"scorer": "cvslice.vision.qc", "version": 1}}
    rows = []
    for fn in csvs:
        tag = os.path.splitext(fn)[0]
        pkl = next((p for s, p in mosh_index.items() if s.startswith(tag)),
                   None)
        loaded = load_clip(os.path.join(scene, fn), pkl)
        if loaded is None:
            print(f"  ! 跳过(加载失败): {tag}")
            continue
        pts, fps, src = loaded
        r = qc.score_clip(pts, fps)
        r["source"] = src
        report[tag] = r
        rows.append((r["score"], tag, src))
    out = os.path.join(scene, REPORT_NAME)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    rows.sort(reverse=True)
    print(f"\n== {os.path.basename(scene)}  ({len(rows)} clips) -> {out}")
    for score, tag, src in rows[:10]:
        print(f"  {score:6.1f}  {tag}  [{src}]")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="actor folder or single scene folder")
    ap.add_argument("--mosh", default=None, help="mosh dir with *_stageii.pkl")
    args = ap.parse_args()
    scenes = find_scenes(args.folder)
    if not scenes:
        sys.exit(f"未找到含 CSV 的场景文件夹: {args.folder}")
    mosh_index = index_mosh(args.mosh)
    for s in scenes:
        scan_scene(s, mosh_index)


if __name__ == "__main__":
    main()

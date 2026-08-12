"""Chunked align for 13-elsdon (resumable, budget per invocation).

Drives data/smpl/15/align_dataset.py's align_one_clip directly (same
validated machinery the other scenes went through), one clip at a time so
runs fit a remote 45s execution window. The 37 recovered clips have no
check/ overlays -> leads assumed 0 (align still normalizes length and
produces the stageii + 30hz pair); the 2 surviving walking clips have
check overlays and get真 measurement + self-verify.

Per-clip reports accumulate in <out_video>/alignment_report_partial.jsonl;
when nothing remains, they are merged into alignment_report.json
(replacing any older entries for the same clip).

Usage: python tools\\align_13_elsdon_chunked.py [--budget 600]
Run repeatedly until it prints ALIGN COMPLETE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN = os.path.join(REPO, "data", "smpl", "15", "align_dataset.py")
PKL_DIR = os.path.join(REPO, "data", "smpl", "13", "13-elsdon")
VID_DIR = os.path.join(REPO, "data", "13", "exported_13", "13-elsdon")
CAM_DIR = os.path.join(VID_DIR, "calibration")
OUT_V = os.path.join(REPO, "data", "13", "13_export_aligned", "13-elsdon")
OUT_P = os.path.join(REPO, "data", "smpl", "13_aligned", "13-elsdon")


def load_align():
    spec = importlib.util.spec_from_file_location("align_ds", ALIGN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=600.0)
    a = ap.parse_args()
    t0 = time.time()
    ad = load_align()
    viz = ad.viz
    table = viz.index_videos_by_view(VID_DIR, viz.ALL_VIEWS)
    part = os.path.join(OUT_V, "alignment_report_partial.jsonl")
    os.makedirs(OUT_V, exist_ok=True)
    os.makedirs(OUT_P, exist_ok=True)

    from pathlib import Path
    pkls = sorted(Path(PKL_DIR).glob("*_stageii.pkl"))
    done = remaining = skipped = 0
    for pf in pkls:
        base = pf.stem
        if os.path.exists(os.path.join(OUT_P, base + "_30hz.pkl")):
            skipped += 1
            continue
        if time.time() - t0 > a.budget:
            remaining += 1
            continue
        vv = viz.find_videos_for_pkl(pf, table, viz.ALL_VIEWS)
        if not vv:
            print(f"!! no videos for {base}")
            remaining += 1
            continue
        rep = ad.align_one_clip({
            "pkl_file": str(pf), "views_videos": vv, "camera_dir": CAM_DIR,
            "out_video_dir": OUT_V, "out_pkl_dir": OUT_P,
            # save_check=True: the mounted FS forbids os.remove, so the
            # temp-check path (save_check=False) would crash on clips that
            # DO have check overlays; keeping the trimmed check clip is fine
            "long_clip_warn": 1000, "save_check": True,
        })
        with open(part, "a", encoding="utf-8") as f:
            f.write(json.dumps(rep) + "\n")
        done += 1
        nw = len(rep.get("warnings", []))
        print(f"{rep['clip']}: OK" + (f" ({nw} warn)" if nw else ""))
    print(f"\naligned {done} | already done {skipped} | REMAINING {remaining}")
    if remaining == 0:
        # merge partial reports into the scene's alignment_report.json
        rp = os.path.join(OUT_V, "alignment_report.json")
        doc = {"pkl_dir": PKL_DIR, "video_dir": VID_DIR, "camera_dir": CAM_DIR,
               "note": "recovered-scene align: 37 clips without check/ use "
                       "leads=0; 2 walking clips measured normally", "clips": []}
        old = {}
        if os.path.exists(rp):
            try:
                for c in json.load(open(rp)).get("clips", []):
                    old[c["clip"]] = c
            except Exception:
                pass
        if os.path.exists(part):
            for line in open(part, encoding="utf-8"):
                c = json.loads(line)
                old[c["clip"]] = c
        doc["clips"] = [old[k] for k in sorted(old)]
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        nb = sum(len(c.get("warnings", [])) for c in doc["clips"])
        print(f"ALIGN COMPLETE: report merged, {len(doc['clips'])} clips,"
              f" {nb} warnings total")


if __name__ == "__main__":
    main()

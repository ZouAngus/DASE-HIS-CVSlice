"""Re-slice the 6 missing views of 13-boss shooting-rep1 from the raw session.

Run ON WINDOWS. The only existing view (topleft) is used as an anchor
template: its first frame is located in the raw boss_13_topleft.mp4 by
frame matching around the logged time (23:25), then the SAME frame window
[S, S+L) is cut from every other camera's raw recording (the 7 raw boss
files are size-identical to 0.03% => started/stopped together; any 1-2
frame residual per view is annotator-nudgeable via corrector view offsets).

Outputs 13-boss-{cam}-shooting-rep1.mp4 next to the existing topleft slice
(default: data\\13\\13_export_aligned\\13-boss). Never overwrites; the raw
files are read-only.

Usage: python tools\\cut_13_shooting.py [--hint-mmss 23:25] [--run]
       (dry-run prints the located window + match score without writing)
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

CAMS = ("topleft", "topcenter", "topright", "bottomleft", "bottomcenter",
        "bottomright", "diagonal")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gray_small(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (160, 90)).astype(np.float32)


def score(a, b):
    """zero-normalized correlation, 1.0 = identical"""
    a, b = a - a.mean(), b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float((a * b).sum() / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=r"E:\Data_2026\video\1.15_13")
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "data", "13", "13_export_aligned", "13-boss"))
    ap.add_argument("--template", default=None,
                    help="existing topleft slice (default: <out-dir>\\"
                         "13-boss-topleft-shooting-rep1.mp4)")
    ap.add_argument("--hint-mmss", default="23:25",
                    help="logged start time inside the raw recording")
    ap.add_argument("--search-sec", type=float, default=90.0,
                    help="search radius around the hint")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    tpl_path = a.template or os.path.join(
        a.out_dir, "13-boss-topleft-shooting-rep1.mp4")
    if not os.path.exists(tpl_path):
        alt = os.path.join(REPO, "data", "13", "exported_13", "13-boss",
                           "13-boss-topleft-shooting-rep1.mp4")
        if os.path.exists(alt):
            tpl_path = alt
        else:
            sys.exit(f"template not found: {tpl_path}")
    cap = cv2.VideoCapture(tpl_path)
    L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        sys.exit("cannot read template first frame")
    tpl = gray_small(f0)
    print(f"template: {os.path.basename(tpl_path)} ({L} frames)")

    raw_tl = os.path.join(a.raw_dir, "boss_13_topleft.mp4")
    cap = cv2.VideoCapture(raw_tl)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mm, ss = a.hint_mmss.split(":")
    hint = int((int(mm) * 60 + int(ss)) * fps)
    lo = max(0, hint - int(a.search_sec * fps))
    hi = min(n_raw - 1, hint + int(a.search_sec * fps))
    # pass 1: coarse scan every 5 frames; pass 2: refine +-6
    best, best_s = -2.0, lo
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    idx = lo
    while idx <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        if (idx - lo) % 5 == 0:
            sc = score(tpl, gray_small(fr))
            if sc > best:
                best, best_s = sc, idx
        idx += 1
    for cand in range(max(0, best_s - 6), min(n_raw - 1, best_s + 7)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cand)
        ok, fr = cap.read()
        if ok:
            sc = score(tpl, gray_small(fr))
            if sc > best:
                best, best_s = sc, cand
    cap.release()
    t = best_s / fps
    print(f"anchor: raw frame {best_s} ({int(t//60)}:{t%60:04.1f}) "
          f"corr={best:.3f}  -> window [{best_s}, {best_s + L})")
    if best < 0.90:
        print("!! low match confidence (<0.90) — check --hint-mmss; not cutting")
        if a.run:
            sys.exit(1)
    if not a.run:
        print("DRY-RUN: add --run to cut the 6 missing views.")
        return

    for cam in CAMS:
        dst = os.path.join(a.out_dir, f"13-boss-{cam}-shooting-rep1.mp4")
        if os.path.exists(dst):
            print(f"{cam}: exists — skip")
            continue
        src = os.path.join(a.raw_dir, f"boss_13_{cam}.mp4")
        cap = cv2.VideoCapture(src)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"),
                              cap.get(cv2.CAP_PROP_FPS) or fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_s)
        n = 0
        while n < L:
            ok, fr = cap.read()
            if not ok:
                break
            out.write(fr)
            n += 1
        cap.release()
        out.release()
        print(f"{cam}: wrote {n}/{L} frames -> {os.path.basename(dst)}")
    print("done. In the corrector, fine-tune each new view's offset if the"
          " overlay is 1-2 frames off (raw cams may differ by a frame or two).")


if __name__ == "__main__":
    main()

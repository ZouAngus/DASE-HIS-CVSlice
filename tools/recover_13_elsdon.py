"""Rebuild the 37 missing 13-elsdon action videos from the raw sword session.

The first-round export died after 2 actions, but its offsets.json kept the
full cut list (39 actions: raw_clip_start/end + pre/post padding + per-view
offsets). The raw footage is the SWORD session (sword==elsdon naming drift)
in data/13_raw_video/sword_13_{cam}.mp4.

Camera mapping verified at corr=1.00 against the 2 surviving slices:
  slice topleft   <- raw topcenter, FLIPPED
  slice topcenter <- raw topleft,   FLIPPED
  slice topright  <- raw topright,  FLIPPED
  bottoms/diagonal <- same name, no flip
Per-view frame window: [start - pre + off_v, end + post + off_v]; view
offsets from the surviving actions' records (session constants).

Resumable (.part safety, skip existing); --views a,b limits a run; --budget
seconds per invocation. Output: data/13/exported_13/13-elsdon/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, "data", "13", "exported_13", "13-elsdon")
RAW = os.path.join(REPO, "data", "13_raw_video")
MAP = {"topleft": ("topcenter", True), "topcenter": ("topleft", True),
       "topright": ("topright", True), "bottomleft": ("bottomleft", False),
       "bottomcenter": ("bottomcenter", False),
       "bottomright": ("bottomright", False), "diagonal": ("diagonal", False)}
OFF = {"topleft": 0, "topcenter": -3, "topright": -15, "bottomleft": -5,
       "bottomcenter": 3, "bottomright": -11, "diagonal": -8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default=None)
    ap.add_argument("--budget", type=float, default=600.0)
    a = ap.parse_args()
    t0 = time.time()
    acts = json.load(open(os.path.join(SCENE, "offsets.json")))["actions"]
    views = a.views.split(",") if a.views else list(MAP)
    done = remaining = 0
    for v in views:
        rawcam, flip = MAP[v]
        src = os.path.join(RAW, f"sword_13_{rawcam}.mp4")
        cap = None
        for act in sorted(acts, key=lambda x: x["raw_clip_start"]):
            tag = f"13-elsdon-{v}-{act['action']}-rep{act['rep']}"
            dst = os.path.join(SCENE, tag + ".mp4")
            if os.path.exists(dst):
                continue
            if time.time() - t0 > a.budget:
                remaining += 1
                continue
            off = act.get("view_offsets", {}).get(v, OFF[v])
            pre = act.get("pre_padding", 5)
            post = act.get("post_padding", 5)
            s = act["raw_clip_start"] - pre + off
            L = act["raw_clip_end"] - act["raw_clip_start"] + 1 + pre + post
            if cap is None:
                cap = cv2.VideoCapture(src)
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            out = cv2.VideoWriter(dst + ".part.mp4",
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, s))
            n = 0
            while n < L:
                ok, fr = cap.read()
                if not ok:
                    break
                out.write(cv2.flip(fr, 1) if flip else fr)
                n += 1
            out.release()
            if n == L:
                os.replace(dst + ".part.mp4", dst)
                done += 1
            else:
                print(f"!! {tag}: {n}/{L} frames only")
                remaining += 1
        if cap:
            cap.release()
    print(f"wrote {done} clips | REMAINING {remaining}")
    if remaining == 0:
        n = len([f for f in os.listdir(SCENE) if f.endswith(".mp4")])
        print(f"RECOVERY COMPLETE: {n} mp4 in 13-elsdon (expect 39x7=273)")


if __name__ == "__main__":
    main()

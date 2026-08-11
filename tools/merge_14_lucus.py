"""Merge Lucus's actor-14 delivery (data/14_lucus) into the standard layout.

Delivery format (verified 2026-08): rgb/ = 2065 ALIGNED trimmed videos
(flat, 295 tags x 7 cams, 30fps); skeleton_smpl24/ = 295 bare (T,24,3)
float32 arrays at native 240Hz with (T-1) == 8*(n_video-1) exactly — i.e.
the same aligned geometry our own align pipeline produces (video frame i
<-> skeleton row 8i). No _edited, no fps metadata.

Produces (originals MOVED, not copied — E: still holds the unaligned set):
  data/smpl/14_aligned/14-{scene}/{tag}_stageii.pkl
        bare array wrapped as {"joints": arr, "mocap_frame_rate": 240.0}
        so the corrector and build_dataset_v1 read fps correctly
  data/14/14_export_aligned/14-{scene}/{cam-video}.mp4     (moved from rgb/)
  data/14/14_export_aligned/14-{scene}/{tag}.csv           (copied from
        exported_14, corrector discovery needs them) + calibration/ +
        offsets.json

Resumable: already-done targets are skipped; run repeatedly until it
prints REMAINING 0. --budget N limits seconds per invocation (default 600).
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import time

CAMS = ("topleft", "topcenter", "topright", "bottomleft", "bottomcenter",
        "bottomright", "diagonal")
VID_RE = re.compile(r"^(14)-([a-z]+)-(%s)-(.+)\.mp4$" % "|".join(CAMS))
PKL_RE = re.compile(r"^(14)-([a-z]+)-(.+)\.pkl$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--budget", type=float, default=600.0)
    a = ap.parse_args()
    t0 = time.time()
    src = os.path.join(a.data, "14_lucus")
    exp_old = os.path.join(a.data, "14", "exported_14")
    out_v = os.path.join(a.data, "14", "14_export_aligned")
    out_s = os.path.join(a.data, "smpl", "14_aligned")
    done, remaining = 0, 0

    def budget_left():
        return time.time() - t0 < a.budget

    # 1) videos: move + split by scene
    rgb = os.path.join(src, "rgb")
    if os.path.isdir(rgb):
        for f in sorted(os.listdir(rgb)):
            m = VID_RE.match(f)
            if not m:
                continue
            dst = os.path.join(out_v, f"14-{m.group(2)}", f)
            if os.path.exists(dst):
                continue
            if not budget_left():
                remaining += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(os.path.join(rgb, f), dst)
            done += 1

    # 2) skeletons: wrap bare array -> dict with fps, write to 14_aligned
    sk = os.path.join(src, "skeleton_smpl24")
    if os.path.isdir(sk):
        for f in sorted(os.listdir(sk)):
            m = PKL_RE.match(f)
            if not m:
                continue
            dst = os.path.join(out_s, f"14-{m.group(2)}",
                               f[:-4] + "_stageii.pkl")
            if os.path.exists(dst):
                continue
            if not budget_left():
                remaining += 1
                continue
            with open(os.path.join(sk, f), "rb") as fh:
                arr = pickle.load(fh)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                pickle.dump({"joints": arr, "mocap_frame_rate": 240.0,
                             "source": "lucus_smpl24_aligned"}, fh, protocol=2)
            done += 1

    # 3) csvs + calibration + offsets from the unaligned export (small)
    if os.path.isdir(exp_old) and budget_left():
        for sc in sorted(os.listdir(exp_old)):
            sdir = os.path.join(exp_old, sc)
            ddir = os.path.join(out_v, sc)
            if not os.path.isdir(sdir) or not os.path.isdir(ddir):
                continue
            for f in os.listdir(sdir):
                if f.endswith(".csv") or f == "offsets.json":
                    dst = os.path.join(ddir, f)
                    if not os.path.exists(dst):
                        shutil.copy2(os.path.join(sdir, f), dst)
                        done += 1
            cal_s, cal_d = os.path.join(sdir, "calibration"), os.path.join(ddir, "calibration")
            if os.path.isdir(cal_s) and not os.path.isdir(cal_d):
                shutil.copytree(cal_s, cal_d)
                done += 1

    print(f"processed {done} items | REMAINING {remaining}")
    if remaining == 0:
        nv = sum(len([f for f in fs if f.endswith(".mp4")])
                 for _, _, fs in os.walk(out_v)) if os.path.isdir(out_v) else 0
        ns = sum(len([f for f in fs if f.endswith(".pkl")])
                 for _, _, fs in os.walk(out_s)) if os.path.isdir(out_s) else 0
        print(f"MERGE COMPLETE: {nv} aligned videos, {ns} skeleton pkls")


if __name__ == "__main__":
    main()

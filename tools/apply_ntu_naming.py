"""One-pass NTU-style renaming for the CAVEAT dataset directory.

Run this AFTER all actors have been built with build_dataset_v1.py, from (or
pointing at) the dataset root. It renames every file to the NTU convention
``SsssCcccPpppRrrrAaaa`` using ONLY what is already recorded in metadata/
(samples.csv ntu_id column + camera_map.json) — nothing is recomputed, so the
ids match what samples.csv has said all along:

  skeletons/15-boss/15-boss-jumping_center-rep1.npz
      -> skeletons/S001C000P015R001A011.npz           (C000 = shared skeleton)
  videos/15-boss/15-boss-topcenter-jumping_center-rep1.mp4
      -> videos/S001C002P015R001A011_rgb.mp4

Also rewrites videos/lists/mmaction_all.txt to the new paths and appends a
``renamed`` marker column state into metadata/naming_applied.json. Renaming is
a MOVE (os.replace) — run on the dataset copy, sources are never touched.
Idempotent: already-renamed files are skipped.

Usage:  python apply_ntu_naming.py [--root DIR] [--run]
        (default is a dry-run report; --run applies)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="dataset root (default: this script's directory)")
    ap.add_argument("--run", action="store_true", help="apply (default dry-run)")
    a = ap.parse_args()
    root = a.root
    md = os.path.join(root, "metadata")

    with open(os.path.join(md, "samples.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(md, "camera_map.json"), encoding="utf-8") as f:
        cam_id = json.load(f)

    moves = []          # (old_relpath, new_relpath)
    label_of = {}       # new video relpath -> label id (for the list rewrite)
    missing = []
    for r in rows:
        tag, sid = r["sample_id"], r["ntu_id"]
        if not sid or "C000" not in sid:
            missing.append(f"{tag}: bad ntu_id '{sid}'")
            continue
        adir = "{}-{}".format(r["actor"], r["scene"])
        # skeleton: tag.npz -> Sxxx C000 ...
        old = os.path.join("skeletons", adir, tag + ".npz")
        if os.path.exists(os.path.join(root, old)):
            moves.append((old, os.path.join("skeletons", sid + ".npz")))
        elif not os.path.exists(os.path.join(root, "skeletons", sid + ".npz")):
            missing.append(f"{tag}: skeleton file not found")
        # videos: per view, swap C000 -> C{cam}
        views = [v for v in (r.get("views") or "").split(";") if v]
        for cam in views:
            cid = cam_id.get(cam)
            if not cid:
                missing.append(f"{tag}: unknown camera '{cam}'")
                continue
            vsid = re.sub(r"C\d{3}", "C{:03d}".format(cid), sid, count=1)
            vname = "{}-{}-{}-{}-rep{}.mp4".format(
                r["actor"], r["scene"], cam, r["action"], r["rep"])
            old = os.path.join("videos", adir, vname)
            new = os.path.join("videos", vsid + "_rgb.mp4")
            if os.path.exists(os.path.join(root, old)):
                moves.append((old, new))
            elif not os.path.exists(os.path.join(root, new)):
                missing.append(f"{tag}/{cam}: video file not found")
            label_of[new.replace(os.sep, "/")] = r["label_id"]

    print(f"{len(rows)} samples in samples.csv -> {len(moves)} files to rename")
    if missing:
        print(f"--- {len(missing)} problems ---")
        for m in missing[:30]:
            print("  !", m)
    for old, new in moves[:5]:
        print("  e.g.", old, "->", new)
    if not a.run:
        print("\nDRY-RUN only. Re-run with --run to rename.")
        return

    for old, new in moves:
        dst = os.path.join(root, new)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(os.path.join(root, old), dst)
    # drop now-empty per-scene dirs (best effort)
    for sub in ("skeletons", "videos"):
        base = os.path.join(root, sub)
        if os.path.isdir(base):
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if os.path.isdir(p) and not os.listdir(p):
                    try:
                        os.rmdir(p)
                    except OSError:
                        pass
    # rewrite the MMAction2 list with the new names
    ld = os.path.join(root, "videos", "lists")
    os.makedirs(ld, exist_ok=True)
    with open(os.path.join(ld, "mmaction_all.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{p} {l}" for p, l in sorted(label_of.items())) + "\n")
    with open(os.path.join(md, "naming_applied.json"), "w", encoding="utf-8") as f:
        json.dump({"scheme": "NTU SsssCcccPpppRrrrAaaa", "renamed": len(moves)},
                  f, indent=2)
    print(f"RENAMED {len(moves)} files; mmaction_all.txt rewritten;"
          " metadata/naming_applied.json written.")


if __name__ == "__main__":
    main()

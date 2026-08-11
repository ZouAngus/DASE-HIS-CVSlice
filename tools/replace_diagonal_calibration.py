"""Propagate the authoritative diagonal calibration to every export dir.

Group decision 2026-08: the diagonal camera calibration refined in June
(living in data/15/15_export_37/15-boss/calibration/cali4_diagonal_*) is
the correct one for ALL sessions. This script overwrites the CONTENT of
every actor/scene export calibration's diagonal files with it, while
KEEPING each target's own filename/prefix (cali2_/cali3_/cali4_ — the
corrector discovers the prefix from the dir, so names must stay).

Rules:
- `*diagonal_extrinsics*.json`        <- source extrinsics content
- `*diagonal_intrinsic*.json`         <- source intrinsic content
  (except *_blank.json placeholders — untouched)
- Files already byte-identical to the source are skipped (idempotent).
- Every replaced file's original is first copied to
  _to_delete/diagonal_cal_backup/<actor-scene>/<name>.

Scope: data/*/exported_* , data/*/*_export* (incl. 13/14 aligned dirs and
15_export_37's six scenes). E: drive untouched.

Usage: python tools\\replace_diagonal_calibration.py [--run]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
SRC_DIR = os.path.join(DATA, "15", "15_export_37", "15-boss", "calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    src_ext = open(os.path.join(SRC_DIR, "cali4_diagonal_extrinsics.json"),
                   "rb").read()
    src_int = open(os.path.join(SRC_DIR,
                   "cali4_diagonal_intrinsic_1280x720.json"), "rb").read()
    bak_root = os.path.join(REPO, "_to_delete", "diagonal_cal_backup")

    cal_dirs = sorted(set(
        glob.glob(os.path.join(DATA, "*", "exported_*", "*", "calibration")) +
        glob.glob(os.path.join(DATA, "*", "*_export*", "*", "calibration"))))
    n_rep = n_same = n_skip = 0
    for cd in cal_dirs:
        scene = os.path.basename(os.path.dirname(cd))
        for f in sorted(os.listdir(cd)):
            fl = f.lower()
            if "diagonal" not in fl or not fl.endswith(".json"):
                continue
            if "_blank" in fl:
                n_skip += 1
                continue
            if "extrinsics" in fl:
                new = src_ext
            elif "intrinsic" in fl:
                new = src_int
            else:
                n_skip += 1
                continue
            p = os.path.join(cd, f)
            cur = open(p, "rb").read()
            if cur == new:
                n_same += 1
                continue
            n_rep += 1
            if a.run:
                bd = os.path.join(bak_root, scene)
                os.makedirs(bd, exist_ok=True)
                if not os.path.exists(os.path.join(bd, f)):
                    shutil.copy2(p, os.path.join(bd, f))
                with open(p, "wb") as fh:
                    fh.write(new)
            print(f"{'REPLACED' if a.run else 'would replace'}: "
                  f"{os.path.relpath(p, DATA)}")
    print(f"\ndirs scanned: {len(cal_dirs)} | replaced: {n_rep} | "
          f"already identical: {n_same} | skipped (blank/other): {n_skip}")
    if not a.run:
        print("DRY-RUN — add --run to apply (originals backed up to"
              " _to_delete/diagonal_cal_backup).")


if __name__ == "__main__":
    main()

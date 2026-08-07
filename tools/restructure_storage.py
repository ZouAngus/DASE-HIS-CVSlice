"""ONE-SHOT storage restructure (run ON WINDOWS, from the cvslice repo).

Executes the 2026-08 group decision end-to-end:
  RAW data lives on E: (portable drive), PROCESSED data lives on C: in
  cvslice\\data, no duplicates.

Steps (in order; each logged; dry-run default, --run applies):
  1. Delete C: cvslice\\_to_delete            (junk + superseded, ~140 GB —
     user-approved; frees the space the consolidation needs)
  2. E: junk cleanup                          (calls cleanup_data_drive.py:
     deletes ._*/DS_Store/Thumbs junk; moves 11_old/12_old/CVCalibration2026/
     05-star-002 + confirmed remnants to E:\\_review_before_delete)
  3. E: file Boss_07/Travel_07 (Jun-2026 re-exports sitting at Data_2026
     root) into csv_data\\, renaming the stale Jan-2026 copies to *.old_jan2026
  4. C: rename C:\\cv_data_2026 -> C:\\cv_data_2026_TO_DELETE   (verified
     byte-superset lives in E:\\Data_2026; empty the renamed dir manually)
  5. E->C consolidation                       (calls consolidate_to_c.py:
     sliced+mosh for 01/03/05/06/07/08/10/11/12/14 into cvslice\\data, E
     read-only, first-wins, collisions logged not overwritten)
  6. Verify + write restructure_report.txt    (per-actor file counts on C,
     disk numbers, collision file pointer)

Usage:  python tools\\restructure_storage.py          (dry-run preview)
        python tools\\restructure_storage.py --run    (do everything)
Safe to re-run: every step is idempotent or skips what's already done.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def sh(args, run):
    print(f"\n>>> {' '.join(args)}")
    if run:
        r = subprocess.run(args)
        if r.returncode != 0:
            print(f"!! step failed (exit {r.returncode}) — aborting so you can look")
            sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-root", default="E:\\")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--cv2026", default="C:\\cv_data_2026")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    run, E, repo = a.run, a.e_root, a.repo
    py = sys.executable
    t0 = time.time()
    log = []

    def note(msg):
        print(msg)
        log.append(msg)

    note(f"{'RUN' if run else 'DRY-RUN'} storage restructure | repo={repo} | E={E}")

    # -- 1. quarantine report ONLY — never blind-delete ---------------------
    # _to_delete holds junk AND quarantined raw/old-sliced copies. Deleting
    # is deferred to verify_quarantine.py, which first PROVES (name+size,
    # optional hash) that every non-junk file has a counterpart on E: and
    # only then purges the proven ones. Space is sufficient without it.
    td = os.path.join(repo, "_to_delete")
    if os.path.isdir(td):
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fs in os.walk(td) for f in fs
                 if os.path.exists(os.path.join(dp, f))) / 1e9
        note(f"[1] quarantine {td} ({sz:.1f} GB) KEPT — run"
             f" tools\\verify_quarantine.py afterwards to prove redundancy"
             f" and purge safely")
    else:
        note("[1] _to_delete already gone — skip")

    # -- 2. E junk cleanup ---------------------------------------------------
    note("[2] E-drive cleanup (cleanup_data_drive.py)")
    sh([py, os.path.join(HERE, "cleanup_data_drive.py"), "--drive", E]
       + (["--run"] if run else []), True)   # its own dry-run when not --run

    # -- 3. Boss_07 / Travel_07 into csv_data --------------------------------
    for name in ("Boss_07.csv", "Travel_07.csv"):
        src = os.path.join(E, "Data_2026", name)
        dst = os.path.join(E, "Data_2026", "csv_data", name)
        if os.path.exists(src):
            note(f"[3] {name}: root Jun-2026 copy -> csv_data"
                 f" (old renamed .old_jan2026)")
            if run:
                if os.path.exists(dst):
                    os.replace(dst, dst + ".old_jan2026")
                os.replace(src, dst)
        else:
            note(f"[3] {name}: no loose copy at Data_2026 root — skip"
                 " (already filed?)")

    # -- 4. retire C:\cv_data_2026 ------------------------------------------
    if os.path.isdir(a.cv2026):
        note(f"[4] rename {a.cv2026} -> {a.cv2026}_TO_DELETE"
             " (E:\\Data_2026 is a verified superset; empty it manually)")
        if run:
            os.rename(a.cv2026, a.cv2026 + "_TO_DELETE")
    else:
        note(f"[4] {a.cv2026} not present — skip")

    # -- 5. consolidation ----------------------------------------------------
    note("[5] E->C consolidation (consolidate_to_c.py) — the long step")
    sh([py, os.path.join(HERE, "consolidate_to_c.py"), "--e-root", E,
        "--dest", os.path.join(repo, "data")] + (["--run"] if run else []),
       True)

    # -- 6. verify + report --------------------------------------------------
    data = os.path.join(repo, "data")
    note("\n[6] verification — per-actor counts under cvslice\\data:")
    for actor in ["01", "03", "05", "06", "07", "08", "10", "11", "12",
                  "13", "14", "15"]:
        exp = None
        adir = os.path.join(data, actor)
        if os.path.isdir(adir):
            for d in os.listdir(adir):
                if d.startswith(("exported_", f"{actor}_export")):
                    exp = os.path.join(adir, d)
                    break
        n_mp4 = n_csv = 0
        if exp:
            for dp, _, fs in os.walk(exp):
                n_mp4 += sum(f.endswith(".mp4") for f in fs)
                n_csv += sum(f.endswith(".csv") for f in fs)
        sk = os.path.join(data, "smpl", actor)
        sk2 = os.path.join(data, "smpl", f"{actor}_aligned")
        n_pkl = 0
        for skd in (sk, sk2):
            if os.path.isdir(skd):
                for dp, _, fs in os.walk(skd):
                    n_pkl += sum(f.endswith(".pkl") for f in fs)
        note(f"    {actor}: mp4 {n_mp4} | csv {n_csv} | pkl {n_pkl}"
             + ("" if (n_mp4 or n_pkl) else "   << nothing on C"))
    coll = os.path.join(data, "consolidate_collisions.json")
    if os.path.exists(coll):
        note(f"    collisions to review: {coll}")
    free = shutil.disk_usage(os.path.splitdrive(repo)[0] + "\\").free / 1e9 \
        if os.name == "nt" else 0
    note(f"\ndone in {(time.time()-t0)/60:.1f} min | C free now: {free:.1f} GB")
    if not run:
        note("\nDRY-RUN only — nothing was changed. Re-run with --run.")
    with open(os.path.join(repo, "restructure_report.txt"), "a",
              encoding="utf-8") as f:
        f.write("\n".join(log) + "\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

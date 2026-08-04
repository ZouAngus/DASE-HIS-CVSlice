"""One-shot cleanup for the E:\\ data drive (run ON WINDOWS, needs no deps).

Two phases, both previewed by the default dry-run:

1. JUNK — deleted outright (safe, zero information content):
   - AppleDouble stubs ``._*`` (~10k, created by macOS on exFAT)
   - ``.DS_Store``, ``Thumbs.db``
   - macOS system dirs: ``.Spotlight-V100``, ``.Trashes``, ``.TemporaryItems``,
     ``.fseventsd``

2. REVIEW — **moved** (not deleted) to ``E:\\_review_before_delete\\`` so a
   human can double-check and empty the folder later:
   - ``sliced_videos_and_csvs\\11_old`` / ``12_old``   (superseded by 11 / 12;
     NOTE: ``11_changed_csv`` / ``12_changed_csv`` patch sets are NOT touched)
   - ``CVCalibration2026``                     (byte-duplicate of CV Calibration2/3)
   - ``sliced_videos_and_csvs\\05\\05-star-002`` (aborted export, 0-byte files)

Everything else (2025 archive, NTU zip, retake dirs, tool checkouts) is left
in place — see benchmark/data_inventory_2026-08.md for the full inventory and
per-subject problem list.

Usage:  python cleanup_data_drive.py            (dry-run report)
        python cleanup_data_drive.py --run      (apply)
A log of every action is appended to E:\\cleanup_log.txt when running.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DRIVE = "E:\\"
ROOTS = ["mosh-results", "sliced_videos_and_csvs", "Data_2026", "2025",
         "04_data", "08-boss", "CVFrame2-v4", "Demoday",
         "CV Calibration", "CV Calibration2", "CV Calibration3",
         "CV Calibration4", "CVCalibration2026"]
JUNK_FILES = (".DS_Store", "Thumbs.db")
JUNK_DIRS = (".Spotlight-V100", ".Trashes", ".TemporaryItems", ".fseventsd")
REVIEW = [os.path.join("sliced_videos_and_csvs", "11_old"),
          os.path.join("sliced_videos_and_csvs", "12_old"),
          "CVCalibration2026",
          os.path.join("sliced_videos_and_csvs", "05", "05-star-002")]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default=DRIVE)
    ap.add_argument("--run", action="store_true", help="apply (default dry-run)")
    a = ap.parse_args()
    drive = a.drive
    review_dst = os.path.join(drive, "_review_before_delete")
    log = open(os.path.join(drive, "cleanup_log.txt"), "a",
               encoding="utf-8") if a.run else None

    def note(msg):
        print(msg)
        if log:
            log.write(msg + "\n")

    n_files, n_dirs, n_bytes = 0, 0, 0
    for root_name in ROOTS:
        root = os.path.join(drive, root_name)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            for d in list(dirnames):
                if d in JUNK_DIRS:
                    p = os.path.join(dirpath, d)
                    n_dirs += 1
                    if a.run:
                        shutil.rmtree(p, ignore_errors=True)
                        note(f"RMDIR {p}")
                    dirnames.remove(d)
            for fn in filenames:
                if fn.startswith("._") or fn in JUNK_FILES:
                    p = os.path.join(dirpath, fn)
                    try:
                        n_bytes += os.path.getsize(p)
                    except OSError:
                        pass
                    n_files += 1
                    if a.run:
                        try:
                            os.remove(p)
                        except OSError as e:
                            note(f"FAILED {p}: {e}")
    print(f"\nphase 1 junk: {n_files} files + {n_dirs} dirs"
          f" ({n_bytes / 1e6:.1f} MB){' deleted' if a.run else ' would be deleted'}")

    moved = 0
    for rel in REVIEW:
        src = os.path.join(drive, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(review_dst, rel.replace(os.sep, "__"))
        moved += 1
        if a.run:
            os.makedirs(review_dst, exist_ok=True)
            try:
                os.rename(src, dst)
                note(f"MOVED {src} -> {dst}")
            except OSError as e:
                note(f"FAILED move {src}: {e}")
        else:
            print(f"would move {src} -> {dst}")
    print(f"phase 2 review: {moved} folders"
          f"{' moved to' if a.run else ' would move to'} {review_dst}")
    if not a.run:
        print("\nDRY-RUN only. Re-run with --run to apply.")
    elif log:
        log.close()


if __name__ == "__main__":
    main()

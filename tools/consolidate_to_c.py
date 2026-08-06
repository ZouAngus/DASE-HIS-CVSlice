"""Consolidate all PROCESSED data from E:\\ into the repo's data/ on C:.

Run ON WINDOWS. E: is read-only throughout — nothing on E is modified or
deleted. Default is a dry-run report; --run copies.

Layout produced (the 13/15 standard, ready for align -> correct -> build):
    data/{actor}/exported_{actor}/{actor}-{scene}/   sliced videos + csvs +
                                                     check/ + calibration/ +
                                                     offsets.json
    data/smpl/{actor}/{actor}-{scene}/               mosh stageii (+_edited)

Rules
-----
- "._*", ".DS_Store", "Thumbs.db" junk is never copied.
- Mosh scene dirs are name-normalized (Boss_01_24points / 02-boss-moshed /
  Travel_002_05_24points all become {actor}-{scene}); retake dirs (_002..)
  merge into the base scene.
- FIRST-WINS priority, nothing is ever overwritten: preferred sources are
  copied first (11/12 _changed_csv patches, 08-boss redo mosh), then the
  base source only fills files that don't exist yet. A same-name file with a
  DIFFERENT size is skipped and logged in the collision report for human
  confirmation (per group rule: never silently drop conflicting versions).
- Same name + same size at destination = already copied, skipped silently
  (safe re-run / resume).
- Skipped entirely: 11_old/12_old (superseded), 05-star-002 (0-byte dead
  export), 04_data (duplicate of 04), 2025 (previous year), raw sources.
- Actor 10 has mosh only (no sliced anywhere); 14 has sliced only (no mosh);
  08 sliced covers just the boss scene. Copied as-is; gaps stay documented in
  benchmark/data_inventory_2026-08.md.

Usage:  python tools\\consolidate_to_c.py           (dry-run)
        python tools\\consolidate_to_c.py --run     (copy)
        [--e-root E:\\] [--dest <repo>\\data]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

JUNK = ("Thumbs.db", ".DS_Store")
SCENE_RE = re.compile(r"(boss|elsdon|gallery|star|travel|trove|sword)", re.I)
SLICED_SRC = {  # actor -> subdir of E:\sliced_videos_and_csvs
    "01": "01", "02": "02", "03": "03", "04": "04", "05": "05",
    "06": "06_updated", "07": "07", "08": "08", "10": "10",
    "11": "11", "12": "12", "14": "14",
}
MOSH_SRC = {  # actor -> subdir of E:\mosh-results (08/10 renamed *-edited
    # in Aug-2026 drop; those supersede the old E:\08-boss redo)
    "01": "01", "02": "02", "03": "03", "04": "04", "05": "05",
    "06": "06", "07": "07", "08": "08-edited", "10": "10-edited",
    "11": "11", "12": "12",
}
SKIP_DIR_TOKENS = ("_old", "05-star-002")


def is_junk(name: str) -> bool:
    return name.startswith("._") or name in JUNK or name.endswith(".bak")


def scene_of(dirname: str):
    m = SCENE_RE.search(dirname)
    return m.group(1).lower() if m else None


class Copier:
    def __init__(self, run: bool):
        self.run = run
        self.n_copied = self.n_dup = self.n_patched = 0
        self.bytes_copied = 0
        self.collisions = []   # (src, dst, src_size, dst_size)
        self.errors = []
        self.expected = set()  # dest paths where a preferred source already
                               # won on purpose (patches / redo) — a later
                               # differing base copy is EXPECTED, not a clash

    def copy_tree(self, src_root: str, dst_root: str, label: str,
                  register: bool = False, skip_pkl: bool = False):
        """Copy src_root/** into dst_root, first-wins, junk-filtered.
        skip_pkl: for sliced sources — mosh pkls misplaced there (e.g. the
        54 stageii sitting in sliced 08-star) are NOT videos, don't copy."""
        for dirpath, dirnames, filenames in os.walk(src_root):
            dirnames[:] = [d for d in dirnames if not is_junk(d)]
            rel = os.path.relpath(dirpath, src_root)
            for fn in filenames:
                if is_junk(fn) or (skip_pkl and fn.endswith(".pkl")):
                    continue
                sp = os.path.join(dirpath, fn)
                dp = os.path.join(dst_root, rel, fn) if rel != "." \
                    else os.path.join(dst_root, fn)
                try:
                    ssz = os.path.getsize(sp)
                    if os.path.exists(dp):
                        dsz = os.path.getsize(dp)
                        if dsz == ssz:
                            self.n_dup += 1
                        elif dp in self.expected:
                            self.n_patched += 1   # preferred source won
                        else:
                            self.collisions.append((label, sp, dp, ssz, dsz))
                        continue
                    if register:
                        self.expected.add(dp)
                    self.n_copied += 1
                    self.bytes_copied += ssz
                    if self.run:
                        os.makedirs(os.path.dirname(dp), exist_ok=True)
                        shutil.copy2(sp, dp)
                except OSError as e:
                    self.errors.append(f"{sp}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-root", default="E:\\")
    ap.add_argument("--dest", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--run", action="store_true", help="copy (default dry-run)")
    a = ap.parse_args()
    E, dest = a.e_root, a.dest

    def do_pass(run: bool):
        cp = Copier(run)
        plan = build(cp, E, dest)
        return cp, plan

    # pass 1: always dry — measure before any byte is written
    cp, plan = do_pass(False)
    gb = cp.bytes_copied / 1e9
    drive = (os.path.splitdrive(dest)[0] + "\\") if os.name == "nt" else "/"
    free = shutil.disk_usage(drive).free / 1e9
    fits = gb <= free - 20   # keep 20 GB headroom
    if a.run and fits:
        cp, plan = do_pass(True)   # pass 2: the real copy
    report(cp, plan, gb, free, fits, dest, a.run)
    if a.run and not fits:
        sys.exit(1)


def build(cp: "Copier", E: str, dest: str):
    plan = []   # human-readable mapping lines

    # ---------- sliced videos + csvs + calibration ----------
    for actor, sub in sorted(SLICED_SRC.items()):
        src = os.path.join(E, "sliced_videos_and_csvs", sub)
        if not os.path.isdir(src):
            print(f"!! missing on E: {src}")
            continue
        exp = os.path.join(dest, actor, f"exported_{actor}")
        # patches FIRST (first-wins => patch version prevails)
        patch = os.path.join(E, "sliced_videos_and_csvs", f"{sub}_changed_csv")
        if os.path.isdir(patch):
            for d in sorted(os.listdir(patch)):
                sc = scene_of(d)
                pd = os.path.join(patch, d)
                if sc and os.path.isdir(pd):
                    cp.copy_tree(pd, os.path.join(exp, f"{actor}-{sc}"),
                                 f"{actor}:patch", register=True)
            plan.append(f"{actor}: patches {os.path.basename(patch)} first")
        scenes = {}
        for d in sorted(os.listdir(src)):
            p = os.path.join(src, d)
            if not os.path.isdir(p) or is_junk(d):
                continue
            if any(t in d for t in SKIP_DIR_TOKENS):
                plan.append(f"{actor}: SKIP {d}")
                continue
            sc = scene_of(d)
            if not sc:
                plan.append(f"{actor}: SKIP (no scene token) {d}")
                continue
            scenes.setdefault(sc, []).append(p)
        for sc, dirs in sorted(scenes.items()):
            # base dir (shortest name) first, retakes fill the gaps
            for p in sorted(dirs, key=lambda x: len(os.path.basename(x))):
                # flatten accidental double-nesting (08\08-boss\08-boss\...)
                inner = os.path.join(p, os.path.basename(p))
                if os.path.isdir(inner):
                    p = inner
                cp.copy_tree(p, os.path.join(exp, f"{actor}-{sc}"),
                             f"{actor}:{sc}", skip_pkl=True)
            if len(dirs) > 1:
                plan.append(f"{actor}: merged {len(dirs)} dirs -> {actor}-{sc}")

    # ---------- mosh skeletons ----------
    # (E:\08-boss old redo intentionally NOT used — superseded by the
    #  Aug-2026 mosh-results\08-edited drop, which has 62 edited boss pkls)
    for actor, sub in sorted(MOSH_SRC.items()):
        src = os.path.join(E, "mosh-results", sub)
        if not os.path.isdir(src):
            print(f"!! missing on E: {src}")
            continue
        for d in sorted(os.listdir(src)):
            p = os.path.join(src, d)
            if not os.path.isdir(p) or is_junk(d):
                continue
            sc = scene_of(d)
            if not sc:
                plan.append(f"{actor}: SKIP mosh dir (no scene token) {d}")
                continue
            cp.copy_tree(p, os.path.join(dest, "smpl", actor,
                                         f"{actor}-{sc}"), f"{actor}:mosh")

    return plan


def report(cp: "Copier", plan, gb, free, fits, dest, ran):
    print("\n".join(plan))
    verb = "copied" if ran and fits else "to copy"
    print(f"\n{verb} : {cp.n_copied} files, {gb:.1f} GB"
          f"   (already present: {cp.n_dup};"
          f" superseded by patch/redo as intended: {cp.n_patched})")
    print(f"free on {os.path.splitdrive(dest)[0] or '/'} : {free:.1f} GB"
          + ("  OK" if fits else "  << NOT ENOUGH (need 20 GB headroom), no copy done"))
    if cp.collisions:
        print(f"\n--- {len(cp.collisions)} same-name-DIFFERENT-size collisions"
              " (skipped, NEED HUMAN CONFIRMATION) ---")
        for label, sp, dp, ssz, dsz in cp.collisions[:40]:
            print(f"  [{label}] {sp} ({ssz}) vs existing {dp} ({dsz})")
        with open(os.path.join(dest, "consolidate_collisions.json"), "w",
                  encoding="utf-8") as f:
            json.dump([{"label": l, "src": s, "dst": d, "src_size": ss,
                        "dst_size": ds} for l, s, d, ss, ds in cp.collisions],
                      f, indent=2, ensure_ascii=False)
        print(f"  full list -> {os.path.join(dest, 'consolidate_collisions.json')}")
    if cp.errors:
        print(f"\n--- {len(cp.errors)} errors ---")
        for e in cp.errors[:20]:
            print("  !", e)
    if not ran:
        print("\nDRY-RUN only. Re-run with --run to copy"
              " (safe to interrupt & re-run: it resumes).")


if __name__ == "__main__":
    main()

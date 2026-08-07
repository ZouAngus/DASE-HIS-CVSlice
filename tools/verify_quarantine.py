"""Prove the quarantine (cvslice\\_to_delete) is redundant, THEN purge it.

Run ON WINDOWS. Never deletes anything unproven:

  JUNK      = files under appledouble_junk/ or git_stale_locks/, or named
              ._* / .DS_Store / *.lock — metadata stubs we created, zero
              dataset content. Always purgeable.
  VERIFIED  = every other file whose (filename, byte size) — and, with
              --hash, MD5 — matches a file somewhere under the E: data roots
              (Data_2026, sliced_videos_and_csvs, mosh-results, 2025) or,
              for 15_export_37.zip, whose every zip member matches the live
              data\\15\\15_export_37 directory on C.
  UNVERIFIED= everything else. NEVER deleted — listed for human review.

Usage:  python tools\\verify_quarantine.py            report only
        python tools\\verify_quarantine.py --hash     + MD5 check (slower)
        python tools\\verify_quarantine.py --purge    delete JUNK+VERIFIED
                                                      (UNVERIFIED stays)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
E_ROOTS = ["Data_2026", "sliced_videos_and_csvs", "mosh-results", "2025",
           "04_data", "08-boss", "_review_before_delete"]
JUNK_DIRS = ("appledouble_junk", "git_stale_locks")


def md5(path, limit=None):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-root", default="E:\\")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--hash", action="store_true",
                    help="also MD5-compare (slow but absolute)")
    ap.add_argument("--purge", action="store_true",
                    help="delete JUNK + VERIFIED files (UNVERIFIED kept)")
    a = ap.parse_args()
    td = os.path.join(a.repo, "_to_delete")
    if not os.path.isdir(td):
        print("no _to_delete — nothing to do")
        return

    print("indexing E: ...")
    index = {}   # (name.lower(), size) -> full path
    for root in E_ROOTS:
        base = os.path.join(a.e_root, root)
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.startswith("._"):
                    continue
                p = os.path.join(dp, f)
                try:
                    index[(f.lower(), os.path.getsize(p))] = p
                except OSError:
                    pass
    print(f"  {len(index)} (name,size) keys")

    junk, verified, unverified = [], [], []
    zip_special = None
    for dp, _, fs in os.walk(td):
        parts = os.path.relpath(dp, td).split(os.sep)
        in_junk = parts[0] in JUNK_DIRS
        for f in fs:
            p = os.path.join(dp, f)
            if in_junk or f.startswith("._") or f in (".DS_Store",) \
                    or f.endswith(".lock") or f.startswith("L") and parts[0] in JUNK_DIRS:
                junk.append(p)
                continue
            if f == "15_export_37.zip":
                zip_special = p
                continue
            try:
                key = (f.lower(), os.path.getsize(p))
            except OSError:
                unverified.append((p, "unreadable"))
                continue
            hit = index.get(key)
            if not hit:
                unverified.append((p, "no name+size match on E"))
                continue
            if a.hash and md5(p) != md5(hit):
                unverified.append((p, f"MD5 differs vs {hit}"))
                continue
            verified.append(p)

    # zip: every member must exist in the live export dir with same size
    if zip_special:
        live = os.path.join(a.repo, "data", "15", "15_export_37")
        livemap = {}
        for dp, _, fs in os.walk(live):
            for f in fs:
                livemap[f.lower()] = os.path.getsize(os.path.join(dp, f))
        missing = 0
        try:
            with zipfile.ZipFile(zip_special) as z:
                for zi in z.infolist():
                    if zi.is_dir() or os.path.basename(zi.filename).startswith("._"):
                        continue
                    n = os.path.basename(zi.filename).lower()
                    if livemap.get(n) != zi.file_size:
                        missing += 1
        except Exception as e:
            missing = -1
            print(f"zip unreadable: {e}")
        if missing == 0:
            verified.append(zip_special)
            print("15_export_37.zip: every member matches the live export dir -> VERIFIED")
        else:
            unverified.append((zip_special, f"{missing} zip members not in live dir"))

    tot = lambda L: sum(os.path.getsize(x) for x in L if os.path.exists(x)) / 1e9
    print(f"\nJUNK      : {len(junk)} files ({tot(junk):.1f} GB) — always safe")
    print(f"VERIFIED  : {len(verified)} files ({tot(verified):.1f} GB) — proven on E"
          + (" (MD5)" if a.hash else " (name+size; add --hash for MD5)"))
    print(f"UNVERIFIED: {len(unverified)} files — KEPT no matter what")
    for p, why in unverified[:30]:
        print(f"   ? {os.path.relpath(p, td)}: {why}")
    if not a.purge:
        print("\nreport only — nothing deleted. Add --purge to delete JUNK+VERIFIED.")
        return
    n = 0
    for p in junk + verified:
        try:
            os.remove(p)
            n += 1
        except OSError as e:
            print(f"  ! {p}: {e}")
    # drop now-empty dirs
    for dp, dns, fs in os.walk(td, topdown=False):
        try:
            os.rmdir(dp)
        except OSError:
            pass
    print(f"PURGED {n} files. Anything left under _to_delete is UNVERIFIED —"
          " review by hand.")


if __name__ == "__main__":
    main()

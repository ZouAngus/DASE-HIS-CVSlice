"""Restructure annotated CVSlice output into the benchmark dataset layout.

Reads one actor's annotation results:
  - export dir   : data/{actor}/…/{actor}-{scene}/   (per-action videos + ref CSVs
                   + offsets.json + corrector_progress.json + calibration/)
  - skeleton dir : data/smpl/{actor}_aligned/{actor}-{scene}/*_stageii.pkl
                   (+ *_stageii_edited.pkl manual corrections, preferred)

and writes the dataset structure agreed in benchmark/ docs (joints.md SMPL-22,
split_v0.json, 规划文档 P2.2/2.3):

  OUT/
  ├── README.md
  ├── metadata/
  │   ├── label_map.json            action string -> class id (created once, stable)
  │   ├── samples.csv               master sample table (incl. offsets + quality)
  │   ├── graph.py                  PARENTS_22 / EDGES_22 / adjacency (joints.md)
  │   ├── splits/split_v0.json      copied verbatim (authoritative X-Sub splits)
  │   └── calibration/{actor}-{scene}/*.json
  ├── skeletons/{actor}-{scene}/{tag}.npz      pts float32 (T,22,3) world-metres,
  │                                            native mocap rate (~120 Hz)
  └── videos/{actor}-{scene}/{...}.mp4         7 views/action, copied as-is
      └── lists/mmaction_all.txt               "relpath label_id" per view-sample

Decisions encoded (2026-08, Angus, final): skeletons keep the NATIVE mocap
rate — the npz stores the native-rate slice covering the aligned window plus
(fps, video_fps, ratio) so dataloaders map video frame i -> skeleton frame
round(i * ratio); per-view videos are trimmed by their corrector offsets at
export (frame-locked to the window; sources untouched); videos are COPIED
(self-contained dir); corrected == *_edited.pkl present (else pristine source,
sample marked silver). Output root: <repo>/data/release/CAVEAT_v0.

Alignment per action (mirrors the corrector's play-window logic): the video
window [lo, hi] is the intersection over views of the offset-valid range, and
the skeleton must cover round((i + skel_offset) * ratio) without clamping.
Views needing a head shift (offset != 0 or lo > 0) are re-cut frame-exactly
(sequential decode); head-aligned views are copied as-is — they may keep a few
surplus TAIL frames beyond n_frames, which dataloaders should ignore.

Safety: the *_edited.pkl provenance sidecar (.meta.json) is honoured — a
sidecar naming a different action means the file is a mis-attribution leftover;
the script falls back to the pristine source and reports it.

Usage (from repo root; numpy required, cv2 optional for video frame counts):
  python tools/build_dataset_v1.py --actor 15 --dry-run
  python tools/build_dataset_v1.py --actor 15 --run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import sys

import numpy as np

CAMS = ("topleft", "topcenter", "topright",
        "bottomleft", "bottomcenter", "bottomright", "diagonal")
# NTU-style IDs: SsssCcccPpppRrrrAaaa (S=scene/setup, C=camera, P=performer,
# R=replication, A=action class). Camera numbering is FIXED by CAMS order
# (C001..C007); C000 marks the shared (all-view) skeleton file. Scene and
# action numberings live in metadata/scene_map.json / label_map.json and are
# append-only — once assigned, an id never changes.
CAM_ID = {c: i + 1 for i, c in enumerate(CAMS)}
# Scene aliases (group decision 2026-08): some actors' exports used a
# different name for the same physical scene. Normalized BEFORE scene-id
# assignment, so e.g. 02-sword and 13-elsdon share one S-id / X-Scene bucket.
# File/tag names keep their original spelling.
SCENE_ALIAS = {"sword": "elsdon"}


def ntu_id(scene_id: int, cam_id: int, actor, rep: int, act_id: int) -> str:
    return "S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}".format(
        scene_id, cam_id, int(actor), rep, act_id)
NAME_RE = re.compile(
    r"^(?P<actor>\d+)-(?P<scene>[A-Za-z]+)-"
    r"(?:(?P<cam>" + "|".join(CAMS) + r")-)?"
    r"(?P<action>.+)-rep(?P<rep>\d+)$")

# joints.md v1 (authoritative): SMPL-22 official joint set.
PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
              9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
N_JOINTS = 22

GRAPH_PY = '''"""CAVEAT SMPL-22 graph (generated from benchmark/joints.md v1)."""
import numpy as np

PARENTS_22 = %r
EDGES_22 = [(j, PARENTS_22[j]) for j in range(1, 22)]   # 21 undirected edges


def adjacency_22(self_loops: bool = True) -> np.ndarray:
    A = np.zeros((22, 22))
    for a, b in EDGES_22:
        A[a, b] = A[b, a] = 1.0
    if self_loops:
        A += np.eye(22)
    return A
''' % (PARENTS_22,)


def load_pkl_pts(path):
    """(T,J,3) float array + fps from a stageii/edited pkl (dict or bare array)."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    fps = 0.0
    if isinstance(d, dict):
        for k in ("mocap_frame_rate", "mocap_framerate", "frame_rate", "fps"):
            if k in d:
                try:
                    fps = float(np.asarray(d[k]).flatten()[0])
                except Exception:
                    pass
                break
        for k in ("joints", "Jtr", "keypoints3d", "pts3d"):
            if k in d:
                d = d[k]
                break
    arr = np.asarray(d, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unexpected shape {arr.shape} in {path}")
    return arr, fps


def parse_name(stem):
    m = NAME_RE.match(stem)
    return m.groupdict() if m else None


def scan_actor(repo, actor):
    """Collect per-action samples from the export dir + skeleton dir."""
    data_root = os.path.join(repo, "data", actor)
    export_root = None
    for root, dirs, files in os.walk(data_root):
        if "offsets.json" in files and os.path.basename(root).startswith(actor + "-"):
            export_root = os.path.dirname(root)
            break
    if not export_root:
        sys.exit(f"no export dir with offsets.json found under {data_root}")
    skel_root = os.path.join(repo, "data", "smpl", f"{actor}_aligned")

    samples, problems = {}, []
    for scene_dir in sorted(os.listdir(export_root)):
        sd = os.path.join(export_root, scene_dir)
        if not os.path.isdir(sd) or not scene_dir.startswith(actor + "-"):
            continue
        try:
            with open(os.path.join(sd, "corrector_progress.json"), encoding="utf-8") as f:
                progress = json.load(f)
        except Exception:
            progress = {}
        for fn in sorted(os.listdir(sd)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in (".csv", ".mp4") or fn.endswith(".bak"):
                continue
            info = parse_name(stem)
            if not info:
                continue
            tag = "{}-{}-{}-rep{}".format(actor, info["scene"], info["action"], info["rep"])
            s = samples.setdefault(tag, {
                "actor": actor, "scene": info["scene"], "action": info["action"],
                "rep": int(info["rep"]), "scene_dir": sd, "views": {}, "csv": None})
            if ext.lower() == ".mp4" and info["cam"]:
                s["views"][info["cam"]] = os.path.join(sd, fn)
            elif ext.lower() == ".csv" and not info["cam"]:
                s["csv"] = os.path.join(sd, fn)
        # attach per-tag corrector progress (offsets tell alignment state)
        for tag, s in samples.items():
            if s["scene_dir"] != sd:
                continue
            p = progress.get(tag, {})
            s["skel_offset"] = int(p.get("skel_offset", 0) or 0)
            vo = p.get("view_offsets", {}) or {}
            s["view_offsets"] = {k: int(v) for k, v in vo.items() if int(v or 0) != 0}

    # skeletons: prefer *_edited.pkl (manual correction), honour .meta.json
    for tag, s in samples.items():
        sdir = os.path.join(skel_root, "{}-{}".format(actor, s["scene"]))
        src = os.path.join(sdir, tag + "_stageii.pkl")
        edt = os.path.join(sdir, tag + "_stageii_edited.pkl")
        s["skel_src"], s["corrected"] = None, False
        if os.path.exists(edt):
            meta_p = edt + ".meta.json"
            owner = None
            if os.path.exists(meta_p):
                try:
                    with open(meta_p, encoding="utf-8") as f:
                        owner = json.load(f).get("tag")
                except Exception:
                    owner = None
            if owner not in (None, tag):
                problems.append(f"{tag}: edited pkl belongs to [{owner}] — using source")
            else:
                s["skel_src"], s["corrected"] = edt, True
        if s["skel_src"] is None and os.path.exists(src):
            s["skel_src"] = src
        if s["skel_src"] is None:
            problems.append(f"{tag}: no skeleton pkl found")
        s["src_pkl"] = src if os.path.exists(src) else None
    return export_root, samples, problems


def video_props(path):
    """(n_frames, fps) via cv2; (-1, 0.0) if unavailable."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return -1, 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()
        return n, fps
    except Exception:
        return -1, 0.0


def trim_video(src, dst, start, end, fps):
    """Frame-exact re-cut of frames [start..end] (sequential decode, like the
    corrector's _trim_video). Returns frames written."""
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps if fps > 0 else 30.0, (w, h))
    skipped = 0
    while skipped < max(0, start):
        if not cap.grab():
            vw.release(); cap.release()
            return 0
        skipped += 1
    cur, n = max(0, start), 0
    while cur <= end:
        ret, frm = cap.read()
        if not ret:
            break
        vw.write(frm)
        cur += 1
        n += 1
    vw.release()
    cap.release()
    return n


def align_window(n_skel, so, ratio, view_tots, view_offs):
    """Video-frame window [lo, hi] valid for every view AND the skeleton
    (mirrors the corrector's _recalc_play_range)."""
    lo, hi = 0, min(view_tots.values()) - 1
    for cam, tot in view_tots.items():
        off = view_offs.get(cam, 0)
        lo = max(lo, -off)
        hi = min(hi, tot - 1 - off)

    def skel_ok(i):
        idx = int(round((i + so) * ratio))
        return 0 <= idx <= n_skel - 1
    while lo <= hi and not skel_ok(lo):
        lo += 1
    while hi >= lo and not skel_ok(hi):
        hi -= 1
    return lo, hi


def skel_window(pts, so, ratio, lo, hi):
    """Native-rate skeleton slice covering video window [lo..hi] (+ start idx).
    Video frame i maps to sliced index round((i+so)*ratio) - slo."""
    n = pts.shape[0]
    slo = min(n - 1, max(0, int(round((lo + so) * ratio))))
    shi = min(n - 1, max(slo, int(round((hi + so) * ratio))))
    return pts[slo:shi + 1], slo


def build(repo, actor, out, run, check_videos):
    export_root, samples, problems = scan_actor(repo, actor)
    tags = sorted(samples)
    print(f"export dir : {export_root}")
    print(f"samples    : {len(tags)} actions, "
          f"{sum(len(s['views']) for s in samples.values())} view-videos")

    # ---- label map (stable: never renumber an existing file) ----------------
    lm_path = os.path.join(out, "metadata", "label_map.json")
    label_map = {}
    if os.path.exists(lm_path):
        with open(lm_path, encoding="utf-8") as f:
            label_map = json.load(f)
    for a in sorted({s["action"] for s in samples.values()}):
        if a not in label_map:
            label_map[a] = len(label_map)

    # scene map (append-only, S-ids start at 1)
    sm_path = os.path.join(out, "metadata", "scene_map.json")
    scene_map = {}
    if os.path.exists(sm_path):
        with open(sm_path, encoding="utf-8") as f:
            scene_map = json.load(f)
    for sc in sorted({SCENE_ALIAS.get(s["scene"], s["scene"])
                      for s in samples.values()}):
        if sc not in scene_map:
            scene_map[sc] = len(scene_map) + 1

    rows, n_corrected, n_offsets = [], 0, 0
    for tag in tags:
        s = samples[tag]
        n_skel, fps = -1, 0.0
        if s["skel_src"]:
            try:
                pts, fps = load_pkl_pts(s["skel_src"])
                if not fps and s.get("src_pkl") and s["src_pkl"] != s["skel_src"]:
                    _, fps = load_pkl_pts(s["src_pkl"])   # bare edited arr: fps from source
                n_skel = pts.shape[0]
                if pts.shape[1] < N_JOINTS:
                    problems.append(f"{tag}: only {pts.shape[1]} joints (<22)")
                    s["skel_src"] = None
            except Exception as e:
                problems.append(f"{tag}: skeleton load failed ({e})")
                s["skel_src"] = None
        n_vid = (video_props(next(iter(s["views"].values())))[0]
                 if (check_videos and s["views"]) else -1)
        if s["corrected"]:
            n_corrected += 1
        if s["skel_offset"] or s["view_offsets"]:
            n_offsets += 1
            problems.append(f"{tag}: non-zero offsets "
                            f"(skel={s['skel_offset']}, views={s['view_offsets']}) "
                            f"— these views will be re-cut at export to frame-lock")
        if len(s["views"]) < len(CAMS):
            missing = sorted(set(CAMS) - set(s["views"]))
            problems.append(f"{tag}: missing views {missing}")
        sid = ntu_id(scene_map[SCENE_ALIAS.get(s["scene"], s["scene"])], 0,
                     actor, s["rep"], label_map[s["action"]] + 1)
        s["sid"] = sid
        rows.append({
            "sample_id": tag, "ntu_id": sid, "actor": actor, "scene": s["scene"],
            "action": s["action"], "label_id": label_map[s["action"]],
            "rep": s["rep"], "n_frames": -1, "fps": 0.0,
            "n_frames_skel": -1,
            "n_frames_skel_native": n_skel, "fps_skel_native": fps,
            "n_frames_video": n_vid, "n_views": len(s["views"]),
            "views": ";".join(sorted(s["views"])),
            "views_recut": "", "corrected": s["corrected"],
            "quality": "gold" if s["corrected"] else "silver",
            "skel_offset": s["skel_offset"],
            "view_offsets": json.dumps(s["view_offsets"]) if s["view_offsets"] else "",
        })

    print(f"corrected  : {n_corrected}/{len(tags)} actions have *_edited.pkl (gold)")
    print(f"offsets    : {n_offsets} actions still carry non-zero offsets")
    print(f"classes    : {len(label_map)} distinct action strings")
    print(f"scenes     : {len(scene_map)} -> " +
          ", ".join(f"S{v:03d}={k}" for k, v in sorted(scene_map.items(), key=lambda kv: kv[1])))
    if tags:
        ex = samples[tags[0]]
        print("naming     : files keep the readable tag names for now; future"
              " NTU id precomputed, e.g. {} -> {}".format(tags[0], ex["sid"]))
    if problems:
        print(f"\n--- {len(problems)} problems/warnings ---")
        for p in problems:
            print("  !", p)
    if not run:
        print("\nDRY-RUN only. Re-run with --run to write the dataset.")
        return

    # ---- write ---------------------------------------------------------------
    md = os.path.join(out, "metadata")
    os.makedirs(os.path.join(md, "splits"), exist_ok=True)
    with open(lm_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(md, "graph.py"), "w", encoding="utf-8") as f:
        f.write(GRAPH_PY)
    sv = os.path.join(repo, "benchmark", "split_v0.json")
    if os.path.exists(sv):
        shutil.copy2(sv, os.path.join(md, "splits", "split_v0.json"))

    with open(sm_path, "w", encoding="utf-8") as f:
        json.dump(scene_map, f, indent=2, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(md, "camera_map.json"), "w", encoding="utf-8") as f:
        json.dump(CAM_ID, f, indent=2)

    n_skel_w, n_vid_w, n_recut = 0, 0, 0
    list_rows = []
    row_by_id = {r["sample_id"]: r for r in rows}
    for tag in tags:
        s = samples[tag]
        row = row_by_id[tag]
        adir = "{}-{}".format(actor, s["scene"])
        if not s["skel_src"] or not s["views"]:
            problems.append(f"{tag}: skipped (needs both skeleton and videos)")
            continue
        pts, src_fps = load_pkl_pts(s["skel_src"])
        if not src_fps and s.get("src_pkl") and s["src_pkl"] != s["skel_src"]:
            _, src_fps = load_pkl_pts(s["src_pkl"])
        props = {cam: video_props(vp) for cam, vp in s["views"].items()}
        view_tots = {c: p[0] for c, p in props.items() if p[0] > 0}
        if not view_tots:
            problems.append(f"{tag}: skipped (no readable video)")
            continue
        vfps = max((p[1] for p in props.values()), default=0.0) or 30.0
        n_skel, min_vtot = pts.shape[0], min(view_tots.values())
        ratio = ((n_skel - 1) / (min_vtot - 1)
                 if n_skel > 1 and min_vtot > 1 else 1.0)
        so = s["skel_offset"]
        lo, hi = align_window(n_skel, so, ratio, view_tots, s["view_offsets"])
        if hi < lo:
            problems.append(f"{tag}: skipped (empty aligned window)")
            continue
        t_out = hi - lo + 1

        # skeleton: NATIVE-rate slice covering the aligned window, SMPL-22.
        # Dataloaders map video frame i -> skeleton index round(i*ratio).
        # Readable tag names for now; NTU renaming happens later in one pass
        # via apply_ntu_naming.py (ntu_id precomputed in samples.csv).
        sk22, slo = skel_window(pts[:, :N_JOINTS], so, ratio, lo, hi)
        sk_dir = os.path.join(out, "skeletons", adir)
        os.makedirs(sk_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(sk_dir, tag + ".npz"),
            pts=np.ascontiguousarray(sk22, dtype=np.float32),
            fps=np.float32(src_fps), video_fps=np.float32(vfps),
            ratio=np.float64(ratio), n_frames_video=np.int64(t_out),
            corrected=bool(s["corrected"]), ntu_id=s["sid"], tag=tag,
            actor=actor, scene=s["scene"], action=s["action"], rep=s["rep"])
        n_skel_w += 1

        # videos: re-cut views needing a head shift; copy head-aligned ones
        v_dir = os.path.join(out, "videos", adir)
        os.makedirs(v_dir, exist_ok=True)
        recut = []
        for cam, vp in sorted(s["views"].items()):
            dst = os.path.join(v_dir, os.path.basename(vp))
            off = s["view_offsets"].get(cam, 0)
            if off != 0 or lo > 0:
                n = trim_video(vp, dst, lo + off, hi + off, vfps)
                if n != t_out:
                    problems.append(f"{tag}/{cam}: re-cut wrote {n}/{t_out} frames")
                recut.append(cam)
                n_recut += 1
            elif not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(vp):
                shutil.copy2(vp, dst)
            n_vid_w += 1
            list_rows.append("videos/{}/{} {}".format(
                adir, os.path.basename(vp), label_map[s["action"]]))
        row.update({"n_frames": t_out, "fps": round(vfps, 3),
                    "n_frames_skel": int(sk22.shape[0]),
                    "views_recut": ";".join(recut)})
        cal_src = os.path.join(s["scene_dir"], "calibration")
        cal_dst = os.path.join(md, "calibration", adir)
        if os.path.isdir(cal_src) and not os.path.isdir(cal_dst):
            shutil.copytree(cal_src, cal_dst,
                            ignore=shutil.ignore_patterns("_*backup*"))

    with open(os.path.join(md, "samples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ld = os.path.join(out, "videos", "lists")
    os.makedirs(ld, exist_ok=True)
    with open(os.path.join(ld, "mmaction_all.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(list_rows) + "\n")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("# CAVEAT dataset (build v1, work-in-progress)\n\n"
                "- Joint set: SMPL-22 (benchmark/joints.md v1); world coords, metres.\n"
                "- skeletons/: one .npz per action — pts float32 (T,22,3) at the"
                " NATIVE mocap rate (fps), sliced to the aligned window; map video"
                " frame i -> skeleton index round(i * ratio) (ratio + video_fps +"
                " n_frames_video stored in the npz).\n"
                "- videos/: per-view mp4 clips, offset-aligned at export (view label"
                " in filename); head-aligned copies may keep surplus TAIL frames"
                " beyond samples.csv n_frames — ignore them.\n"
                "- lists/mmaction_all.txt = `relpath label_id` per view-sample.\n"
                "- metadata/: label_map.json, samples.csv (gold/silver, applied"
                " offsets, re-cut views), graph.py, splits/.\n"
                "- NAMING: files currently use readable tags"
                " ({actor}-{scene}-[{cam}-]{action}-rep{N}). The final NTU-style"
                " ids (SsssCcccPpppRrrrAaaa) are PRECOMPUTED in samples.csv"
                " (ntu_id column; C000 = shared skeleton) + scene_map.json /"
                " camera_map.json. Once ALL actors are in, run"
                " `python apply_ntu_naming.py --run` (in this directory) to"
                " rename everything in one pass.\n"
                "- Generated by tools/build_dataset_v1.py — do not hand-edit.\n")
    # ship the future renaming tool inside the dataset (self-contained archive)
    renamer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "apply_ntu_naming.py")
    if os.path.exists(renamer):
        shutil.copy2(renamer, os.path.join(out, "apply_ntu_naming.py"))
    if problems:
        print(f"\n--- post-run problems ({len(problems)} total incl. scan) ---")
        for p in problems[-20:]:
            print("  !", p)
    print(f"\nWROTE: {n_skel_w} skeleton npz, {n_vid_w} videos "
          f"({n_recut} re-cut for alignment), {len(rows)} samples.csv rows -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--actor", default="15")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None,
                    help="output root (default: <repo>/data/release/CAVEAT_v0)")
    ap.add_argument("--run", action="store_true", help="actually write (default dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry-run (default)")
    ap.add_argument("--check-videos", action="store_true",
                    help="open each first-view video to count frames (needs cv2, slow)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.repo, "data", "release", "CAVEAT_v0")
    build(a.repo, a.actor, out, run=a.run and not a.dry_run,
          check_videos=a.check_videos)


if __name__ == "__main__":
    main()

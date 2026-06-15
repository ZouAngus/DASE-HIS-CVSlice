"""Data discovery: find CSVs, video folders, and cameras for a scene."""
import os
import re
import numpy as np
import pandas as pd
from ..core.constants import CAMERA_NAMES


_SCENE_ALIASES = {
    "sword": {"sword", "elsdon"},
    "elsdon": {"sword", "elsdon"},
}


def _normalize_scene_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def scene_keys(name: str | None) -> set[str]:
    """Return normalized scene keys including known aliases."""
    key = _normalize_scene_key(name or "")
    if not key:
        return set()
    return set(_SCENE_ALIASES.get(key, {key}))


def scene_name_matches(candidate: str, scene_name: str | None) -> bool:
    """True if *candidate* matches *scene_name* or one of its aliases."""
    cand = _normalize_scene_key(candidate)
    if not cand:
        return False
    keys = scene_keys(scene_name)
    if not keys:
        return True
    return any(k in cand or cand in k for k in keys)


def find_data_subfolder(data_root: str, sheet_name: str) -> str | None:
    """Find the data subfolder matching a scene name."""
    if not data_root or not os.path.isdir(data_root):
        return None
    keys = scene_keys(sheet_name)
    # Exact/alias match
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and _normalize_scene_key(entry) in keys:
            return full
    # Fuzzy match with aliases
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and scene_name_matches(entry, sheet_name):
            return full
    return None


def find_csv_in_folder(folder: str) -> str | None:
    """Find the first 'extracted*.csv' in a folder."""
    if not folder or not os.path.isdir(folder):
        return None
    for fn in sorted(os.listdir(folder)):
        if fn.lower().startswith("extracted") and fn.lower().endswith(".csv"):
            return os.path.join(folder, fn)
    return None


def find_csv_for_scene(data_root: str, sheet_name: str) -> tuple[str | None, str | None]:
    """Find CSV + video folder for a scene.

    Returns (csv_path | None, video_folder | None).
    """
    subfolder = find_data_subfolder(data_root, sheet_name)
    # 1. CSV inside subfolder
    if subfolder:
        csv_path = find_csv_in_folder(subfolder)
        if csv_path:
            return csv_path, subfolder
    # 2. CSV in data root matching scene name
    csv_path = None
    if data_root and os.path.isdir(data_root):
        for fn in sorted(os.listdir(data_root)):
            if not fn.lower().endswith(".csv"):
                continue
            if not fn.lower().startswith("extracted"):
                continue
            fk = os.path.splitext(fn)[0].replace("extracted", "").strip("_")
            if scene_name_matches(fk, sheet_name):
                csv_path = os.path.join(data_root, fn)
                break
    return csv_path, subfolder


def find_cameras_in_folder(folder: str, scene_hint: str | None = None) -> list[str]:
    """Detect available camera names by scanning for matching .mp4 files.

    If *scene_hint* is given, only consider files whose name contains the
    normalised scene key (e.g. 'boss' inside 'boss_15_topleft.mp4').
    """
    if not folder or not os.path.isdir(folder):
        return []
    scene_hint_present = bool(scene_hint)
    cams = []
    for cn in CAMERA_NAMES:
        for fn in os.listdir(folder):
            fl = fn.lower()
            if not fl.endswith(".mp4"):
                continue
            if cn not in fl:
                continue
            if scene_hint_present and not scene_name_matches(fl, scene_hint):
                continue
            cams.append(cn)
            break
    return cams


def load_csv_as_pts3d(csv_path: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    """Load extracted CSV -> (T, J, 3) array with NaN interpolation.

    Returns (pts3d_array, valid_mask, was_nan_mask, fps):
        - pts3d_array: (T, J, 3) with NaN replaced by interpolated values
        - valid_mask: (T,) bool — True if frame has at least one originally valid joint
        - was_nan_mask: (T, J) bool — True where original data was NaN (now interpolated)
        - fps: float — Export Frame Rate from CSV header (default 240.0 if not found)
    """
    from ..vision.interpolation import interpolate_joints

    # Read CSV header to extract frame rate
    fps = 240.0  # default
    try:
        with open(csv_path, "r") as f:
            first_line = f.readline()
            if "Export Frame Rate" in first_line:
                parts = first_line.split(",")
                for i, part in enumerate(parts):
                    if "Export Frame Rate" in part and i + 1 < len(parts):
                        try:
                            fps = float(parts[i + 1])
                            break
                        except:
                            pass
    except:
        pass


    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    nc = df.shape[1]
    if nc % 3 != 0:
        return None, None, None, fps
    pts_raw = df.values.reshape(-1, nc // 3, 3)

    # Interpolate NaN gaps
    pts_filled, was_nan = interpolate_joints(pts_raw)

    # Valid mask: frame has at least one originally non-NaN joint
    valid = ~np.all(was_nan, axis=1)
    return pts_filled, valid, was_nan, fps


def load_mosh_pkl(pkl_path: str, which: str = "sim"
                  ) -> tuple[np.ndarray | None, float]:
    """Load a MoSh++ ``*_stageii.pkl`` as (T, J, 3) marker positions.

    MoSh++ stores per-frame 37-marker data inside
    ``stageii_debug_details``:
        - ``markers_orig`` (T, 37, 3): the raw triangulated markers fed into
          MoSh — identical to the CVSlice export CSV.
        - ``markers_sim``  list of (37, 3): the markers re-simulated from the
          fitted SMPL-X body (cleaner, no NaN/jitter).

    Both live in the same world frame as the calibration, so they project
    correctly with the exported intrinsics/extrinsics.

    Args:
        pkl_path: path to a ``*_stageii.pkl`` file.
        which: ``"sim"`` for the fitted markers, ``"orig"`` for the raw input.

    Returns:
        (pts3d (T, J, 3) float64 | None, fps).  fps comes from
        ``mocap_frame_rate`` (default 240.0).
    """
    import pickle

    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
    except Exception:
        return None, 240.0
    if not isinstance(d, dict):
        return None, 240.0
    sd = d.get("stageii_debug_details", {}) or {}
    try:
        fps = float(sd.get("mocap_frame_rate", 240.0) or 240.0)
    except Exception:
        fps = 240.0

    primary = "markers_sim" if which == "sim" else "markers_orig"
    fallback = "markers_orig" if which == "sim" else "markers_sim"
    arr = sd.get(primary)
    if arr is None:
        arr = sd.get(fallback)
    if arr is None:
        return None, fps

    try:
        if isinstance(arr, list):
            pts = np.stack([np.asarray(a, dtype=np.float64) for a in arr], 0)
        else:
            pts = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None, fps
    if pts.ndim != 3 or pts.shape[2] != 3:
        return None, fps
    return pts, fps

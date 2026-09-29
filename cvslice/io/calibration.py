"""Camera calibration file loading."""
import os
import json
from ..core.constants import CAMERA_NAMES


def _intrinsic_score(fn: str, prefer_size, declared):
    """rank intrinsic candidates: resolution match first, blanks last"""
    fl = fn.lower()
    match = 0
    if prefer_size and declared and tuple(declared) == tuple(prefer_size):
        match = 2
    elif not declared:
        match = 1
    return (match, 0 if "blank" in fl else 1, fn)


def load_calibration(cal_dir: str, cam_name: str, prefer_size=(1280, 720)):
    """Load intrinsic + extrinsic JSON for a single camera.

    A calibration folder can hold several intrinsics for the same camera
    (``*_intrinsic.json``, ``*_intrinsic_1280x720.json``,
    ``*_intrinsic_1280_720_blank.json``) and they are NOT interchangeable:
    some are the raw 1920x1080 / 3840x2160 calibration, others the rescaled
    720p one. Picking the wrong one scales the projected skeleton (a 1080p
    intrinsic on a 720p frame draws it ~1.5x too big) and shifts it by the
    principal-point difference. So choose the intrinsic whose declared
    ``image_size`` matches the video we are about to draw on (720p exports by
    default), prefer non-``blank`` files, and fall back to a stable
    alphabetical order -- never to whatever os.listdir happens to yield last.

    Returns (intrinsic_dict, extrinsic_dict) or (None, None).
    """
    cands, extr = [], None
    for fn in sorted(os.listdir(cal_dir)):
        fl = fn.lower()
        if cam_name not in fl:
            continue
        fp = os.path.join(cal_dir, fn)
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        if "intrinsic" in fl:
            cands.append((_intrinsic_score(fn, prefer_size,
                                           d.get("image_size")), d))
        elif "extrinsic" in fl:
            extr = d
    intr = max(cands, key=lambda c: c[0])[1] if cands else None
    return intr, extr


def load_all_calibrations(cal_dir: str) -> dict:
    """Load calibration for all known cameras.

    Returns {cam_name: (intrinsic, extrinsic)} for cameras with both files.
    """
    calibs = {}
    for cn in CAMERA_NAMES:
        intr, extr = load_calibration(cal_dir, cn)
        if intr and extr:
            calibs[cn] = (intr, extr)
    return calibs

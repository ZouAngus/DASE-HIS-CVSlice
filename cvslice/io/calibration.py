"""Camera calibration file loading."""
import os
import json
from ..core.constants import CAMERA_NAMES


def load_calibration(cal_dir: str, cam_name: str):
    """Load intrinsic + extrinsic JSON for a single camera.

    Returns (intrinsic_dict, extrinsic_dict) or (None, None).
    """
    intr = extr = None
    for fn in os.listdir(cal_dir):
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
            intr = d
        elif "extrinsic" in fl:
            extr = d
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

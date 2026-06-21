"""Parity tests: SkeletonSource load/save must match the exact old loaders."""
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvslice.io.discovery import load_csv_as_pts3d, load_mosh_pkl
from cvslice.io.skeleton_sources import (
    CsvSource, MoshPklSource, available_sources, preferred_key,
)

CSV = "data/15/15_export_37/15-boss/15-boss-bending_down_center-rep1.csv"
PKL = "data/smpl/15/15-boss_renamed/15-boss-bending_down_center-rep1_stageii.pkl"
_HAVE = os.path.exists(CSV) and os.path.exists(PKL)


def test_csv_source_load_matches():
    pts_ref, _v, wn_ref, fps_ref = load_csv_as_pts3d(CSV)
    pts, wn, fps = CsvSource(CSV).load()
    assert np.array_equal(np.nan_to_num(pts), np.nan_to_num(pts_ref))
    assert fps == fps_ref
    assert np.array_equal(wn, wn_ref)


def test_joint_source_load_matches():
    ref, fps_ref = load_mosh_pkl(PKL, "sim")
    pts, wn, fps = MoshPklSource(PKL, "sim", "mosh_joints", "x", "joints").load()
    assert np.array_equal(pts, ref) and fps == fps_ref and wn is None


def test_availability_and_preference():
    act = {"csv": CSV, "pkl": PKL}
    keys = [s.key for s in available_sources(act, mosh_kind="joints")]
    assert keys == ["mosh_joints", "csv"]          # mosh first, csv reference
    assert preferred_key(keys) == "mosh_joints"    # default = SMPL
    keys_m = [s.key for s in available_sources(act, mosh_kind="markers")]
    assert keys_m == ["mosh_sim", "mosh_orig", "csv"]
    assert [s.key for s in available_sources({"csv": CSV})] == ["csv"]
    assert preferred_key(["csv"]) == "csv"


def test_csv_save_roundtrip():
    pts, _wn, _fps = CsvSource(CSV).load()
    tmp = os.path.join(tempfile.gettempdir(), "rt.csv")
    CsvSource(CSV).save(pts, tmp)
    back, _, _, _ = load_csv_as_pts3d(tmp)
    assert np.allclose(np.nan_to_num(back), np.nan_to_num(pts), atol=1e-4)
    os.remove(tmp)


def test_pkl_save_roundtrip():
    src = MoshPklSource(PKL, "sim", "mosh_joints", "x", "joints")
    pts, _, _ = src.load()
    pts = pts.copy(); pts[5, 3] += 0.01
    tmp = os.path.join(tempfile.gettempdir(), "rt.pkl")
    src.save(pts, tmp)
    back = pickle.load(open(tmp, "rb"))
    assert back.dtype == np.float32 and back.shape == pts.shape
    assert np.allclose(back, pts.astype(np.float32))
    os.remove(tmp)


if __name__ == "__main__":
    if not _HAVE:
        print("SKIP: sample data not present")
    else:
        for fn in (test_csv_source_load_matches, test_joint_source_load_matches,
                   test_availability_and_preference, test_csv_save_roundtrip,
                   test_pkl_save_roundtrip):
            fn()
        print("skeleton_sources parity OK")

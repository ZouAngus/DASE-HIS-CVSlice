"""Unit tests for cvslice.vision.qc (clip quality scoring)."""
import numpy as np

from cvslice.vision import qc

FPS = 60.0


def _clean_clip(T=240, J=24, seed=0):
    """A moving, well-formed skeleton: rigid translation of a fixed pose."""
    rng = np.random.default_rng(seed)
    pose = rng.normal(scale=0.3, size=(J, 3))
    drift = np.cumsum(np.full((T, 3), [0.01, 0.0, 0.0]), axis=0)  # 0.6 m/s
    return pose[None, :, :] + drift[:, None, :]


def test_clean_clip_scores_low():
    r = qc.score_clip(_clean_clip(), FPS)
    assert r["score"] < 10
    assert r["suspect_frames"] == []
    assert r["frozen_joints"] == []


def test_frozen_joint_detected_while_body_moves():
    pts = _clean_clip()
    pts[60:180, 20] = pts[60, 20]          # freeze left wrist for 2 s
    r = qc.score_clip(pts, FPS)
    assert 20 in r["frozen_joints"]
    assert r["components"]["frozen"] > 0.3
    assert any(60 <= f < 180 for f in r["suspect_frames"])


def test_teleport_spike_detected():
    pts = _clean_clip()
    pts[100, 20] += np.array([2.0, 0.0, 0.0])   # 2 m teleport for one frame
    r = qc.score_clip(pts, FPS)
    assert r["components"]["spike"] > 0
    assert any(99 <= f <= 101 for f in r["suspect_frames"])


def test_bone_stretch_detected():
    pts = _clean_clip()
    # Stretch the left forearm (wrist 20 away from elbow 18) mid-clip.
    pts[120:200, 20] += (pts[120:200, 20] - pts[120:200, 18]) * 2.0
    r = qc.score_clip(pts, FPS)
    assert r["components"]["bone"] > 0.05
    assert any(120 <= f < 200 for f in r["suspect_frames"])


def test_nan_fraction_counted():
    pts = _clean_clip()
    pts[:120, 5] = np.nan
    r = qc.score_clip(pts, FPS)
    assert abs(r["components"]["nan"] - 0.5 / 24) < 0.01


def test_ranking_orders_worst_first():
    good = qc.score_clip(_clean_clip(seed=1), FPS)["score"]
    bad_pts = _clean_clip(seed=2)
    bad_pts[30:210, 21] = bad_pts[30, 21]       # long frozen right wrist
    bad = qc.score_clip(bad_pts, FPS)["score"]
    assert bad > good


def test_default_pairs_topologies():
    p24 = qc.default_pairs(24)
    assert (22, 20) in p24 and (4, 1) in p24 and len(p24) == 23
    p22 = qc.default_pairs(22)
    assert (20, 18) in p22 and (22, 20) not in p22 and len(p22) == 21

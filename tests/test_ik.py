"""Unit tests for cvslice.vision.ik (two-bone limb IK for the corrector).

These encode the zero-ambiguity behaviour contract:
  * bone lengths are always preserved exactly;
  * a target beyond reach gives a perfectly STRAIGHT limb, clamped —
    a straight arm must never be folded;
  * mid-joint (elbow/knee) drags stay on the swivel circle, effector fixed;
  * degenerate cases refuse (return None) instead of guessing.
"""
import numpy as np
import pytest

from cvslice.vision import ik

L1, L2 = 0.30, 0.25          # upper arm / forearm, metres
ROOT = np.array([1.0, 2.0, 3.0])


def _len(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ------------------------------------------------------------ solve_effector
def test_effector_reachable_preserves_lengths_and_hits_target():
    target = ROOT + np.array([0.35, 0.10, 0.05])       # inside reach
    hint = ROOT + np.array([0.1, 0.2, 0.0])
    mid, eff, clamped = ik.solve_effector(ROOT, target, L1, L2, hint)
    assert not clamped
    assert _len(ROOT, mid) == pytest.approx(L1, abs=1e-9)
    assert _len(mid, eff) == pytest.approx(L2, abs=1e-9)
    assert eff == pytest.approx(target, abs=1e-9)


def test_effector_beyond_reach_gives_straight_clamped_limb():
    direction = np.array([0.0, 1.0, 0.0])
    target = ROOT + direction * (L1 + L2) * 2.0        # far beyond reach
    mid, eff, clamped = ik.solve_effector(ROOT, target, L1, L2,
                                          ROOT + np.array([0.1, 0.1, 0.0]))
    assert clamped
    # Effector clamped onto the reach sphere, limb perfectly straight.
    assert _len(ROOT, eff) == pytest.approx(L1 + L2, abs=1e-9)
    assert _len(ROOT, mid) == pytest.approx(L1, abs=1e-9)
    assert _len(mid, eff) == pytest.approx(L2, abs=1e-9)
    # Collinearity: mid lies exactly on the root->eff segment (never folded).
    cross = np.cross(eff - ROOT, mid - ROOT)
    assert np.linalg.norm(cross) == pytest.approx(0.0, abs=1e-9)


def test_effector_exactly_at_reach_is_straight_but_not_clamped():
    direction = np.array([1.0, 0.0, 0.0])
    target = ROOT + direction * (L1 + L2)
    mid, eff, clamped = ik.solve_effector(ROOT, target, L1, L2, None)
    assert not clamped
    assert eff == pytest.approx(target, abs=1e-9)
    assert _len(ROOT, mid) == pytest.approx(L1, abs=1e-9)


def test_effector_too_close_clamps_outward():
    target = ROOT + np.array([0.001, 0.0, 0.0])        # inside |l1-l2|
    mid, eff, clamped = ik.solve_effector(ROOT, target, L1, L2, None)
    assert clamped
    assert _len(ROOT, eff) == pytest.approx(abs(L1 - L2), abs=1e-9)
    assert _len(ROOT, mid) == pytest.approx(L1, abs=1e-9)
    assert _len(mid, eff) == pytest.approx(L2, abs=1e-9)


def test_effector_prefers_hint_swivel_plane():
    target = ROOT + np.array([0.35, 0.0, 0.0])
    hint = ROOT + np.array([0.15, 0.9, 0.0])           # bend toward +Y
    mid, _, _ = ik.solve_effector(ROOT, target, L1, L2, hint)
    assert (mid - ROOT)[1] > 0                          # elbow went +Y
    hint2 = ROOT + np.array([0.15, -0.9, 0.0])
    mid2, _, _ = ik.solve_effector(ROOT, target, L1, L2, hint2)
    assert (mid2 - ROOT)[1] < 0                         # elbow went -Y


def test_effector_falls_back_to_prev_mid_when_hint_collinear():
    target = ROOT + np.array([0.35, 0.0, 0.0])
    collinear_hint = ROOT + np.array([0.2, 0.0, 0.0])   # on the axis: useless
    prev = ROOT + np.array([0.1, 0.0, 0.8])             # bend toward +Z
    mid, _, _ = ik.solve_effector(ROOT, target, L1, L2, collinear_hint, prev)
    assert (mid - ROOT)[2] > 0


def test_effector_target_at_root_returns_none():
    assert ik.solve_effector(ROOT, ROOT.copy(), L1, L2, None) is None


# -------------------------------------------------------------- solve_swivel
def test_swivel_stays_on_circle_and_preserves_everything():
    eff = ROOT + np.array([0.30, 0.20, 0.0])            # reachable, bent
    drag = ROOT + np.array([0.1, 0.0, 5.0])             # pull elbow toward +Z
    mid = ik.solve_swivel(ROOT, eff, L1, L2, drag)
    assert mid is not None
    assert _len(ROOT, mid) == pytest.approx(L1, abs=1e-9)
    assert _len(mid, eff) == pytest.approx(L2, abs=1e-9)
    assert (mid - ROOT)[2] > 0                          # followed the drag


def test_swivel_refused_when_limb_straight():
    eff = ROOT + np.array([L1 + L2, 0.0, 0.0])          # fully extended
    assert ik.solve_swivel(ROOT, eff, L1, L2, ROOT + 1.0) is None


def test_swivel_refused_when_drag_on_axis():
    eff = ROOT + np.array([0.30, 0.20, 0.0])
    circ = ik.swivel_circle(ROOT, eff, L1, L2)
    assert circ is not None
    center, _, _ = circ
    assert ik.solve_swivel(ROOT, eff, L1, L2, center) is None


# ---------------------------------------------------------- reference_lengths
def test_reference_lengths_median_robust_to_outlier_frames():
    T = 20
    pts = np.zeros((T, 24, 3))
    c = _CH24_LEFT_ARM = ik.limb_chains(24)[0]
    pts[:, c.root] = ROOT
    pts[:, c.mid] = ROOT + np.array([L1, 0, 0])
    pts[:, c.eff] = ROOT + np.array([L1 + L2, 0, 0])
    pts[3, c.eff] = ROOT + np.array([9.0, 0, 0])        # one wild frame
    pts[5, c.mid] = np.nan                              # one missing frame
    ref = ik.reference_lengths(pts, c)
    assert ref is not None
    assert ref[0] == pytest.approx(L1, abs=1e-9)
    assert ref[1] == pytest.approx(L2, abs=1e-9)


def test_reference_lengths_refuses_with_too_few_frames():
    pts = np.full((10, 24, 3), np.nan)
    c = ik.limb_chains(24)[0]
    pts[0, [c.root, c.mid, c.eff]] = [ROOT, ROOT + 1, ROOT + 2]
    assert ik.reference_lengths(pts, c) is None


# ------------------------------------------------------------------ topology
def test_chain_maps_smpl24_and_h36m17_and_unsupported():
    eff24, mid24 = ik.chain_maps(24)
    assert set(eff24) == {20, 21, 7, 8}                 # wrists + ankles
    assert set(mid24) == {18, 19, 4, 5}                 # elbows + knees
    assert eff24[20].rider == 22 and eff24[21].rider == 23
    eff22, _ = ik.chain_maps(22)                        # SMPL-22 also valid
    assert set(eff22) == {20, 21, 7, 8}
    eff17, mid17 = ik.chain_maps(17)
    assert set(eff17) == {3, 6, 13, 16}
    assert set(mid17) == {2, 5, 12, 15}
    assert ik.limb_chains(37) == []                     # raw markers: no IK


def test_sample_circle_points_lie_on_circle():
    center = np.array([0.0, 1.0, 2.0])
    axis = np.array([0.0, 0.0, 1.0])
    pts = ik.sample_circle(center, 0.5, axis, 12)
    assert pts.shape == (12, 3)
    d = np.linalg.norm(pts - center, axis=1)
    assert np.allclose(d, 0.5, atol=1e-9)
    assert np.allclose((pts - center) @ axis, 0.0, atol=1e-9)


def test_root_map_for_rigid_limb_translation():
    r24 = ik.root_map(24)
    assert set(r24) == {16, 17, 1, 2}                   # shoulders + hips
    assert r24[16].eff == 20 and r24[16].rider == 22
    r17 = ik.root_map(17)
    assert set(r17) == {11, 14, 1, 4}
    assert ik.root_map(37) == {}

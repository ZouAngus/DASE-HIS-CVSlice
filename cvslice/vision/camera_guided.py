"""Camera-guided in-between filling for the Skeleton Corrector.

Between two hand-placed keyframes the *source* SMPL trajectory may be wrong, and
a blind interpolation can't tell a real motion from a source error: over a large
gap a true limb swing and a 20 cm source drift are the same large deviation from
the keyframe chord. The multi-view videos break that tie. We triangulate the
subject's 2D pose across the calibrated cameras to recover the TRUE in-between
motion, then warp it to pass exactly through the user's keyframes:

    result = camera_trajectory + smooth_keyframe_anchored_residual

* the cameras supply the *shape* of the motion in between (real, not guessed);
* the keyframes stay hard anchors (the residual is exact there);
* the surface(COCO)-vs-rotation-center(SMPL) offset and any slow calibration
  drift are a smooth, low-frequency residual -> absorbed, not injected;
* where the cameras can't see a joint (occluded / <2 confident views) we fall
  back to the original source baseline for that joint/frame -> never worse than
  today.

This is exactly the existing offset-space interpolation with the camera track
substituted for the source baseline, so it reuses ``interpolate_offsets_all_joints``
verbatim. Temporal coherence (no frame-to-frame jitter) comes from a One-Euro
pass on the triangulated track; boundary continuity from a velocity-matched
cosine ease into the untouched source on both sides.
"""
from __future__ import annotations

import numpy as np

from .propagation import interpolate_offsets_all_joints, one_euro

# COCO-17 body keypoint index -> SMPL-24 joint index. Only the 12 limb joints
# map cleanly; COCO body has no face-internal / hand / foot / spine joints.
COCO_TO_SMPL24 = {
    5: 16, 6: 17,    # shoulders  (L, R)
    7: 18, 8: 19,    # elbows
    9: 20, 10: 21,   # wrists
    11: 1, 12: 2,    # hips
    13: 4, 14: 5,    # knees
    15: 7, 16: 8,    # ankles
}
SMPL_PELVIS, SMPL_LHIP, SMPL_RHIP = 0, 1, 2

# SMPL joints the cameras can drive: the 12 mapped + a derived pelvis (hip mid).
CAMERA_DRIVEN = sorted(set(COCO_TO_SMPL24.values()) | {SMPL_PELVIS})


def coco_to_smpl24(coco_xyz: np.ndarray) -> np.ndarray:
    """(T,17,3) triangulated COCO points -> (T,24,3) in SMPL-24 layout.

    Unmapped SMPL joints are NaN. Pelvis is derived as the hip midpoint on
    frames where both hips were triangulated.
    """
    T = coco_xyz.shape[0]
    out = np.full((T, 24, 3), np.nan, dtype=np.float64)
    for c, s in COCO_TO_SMPL24.items():
        out[:, s] = coco_xyz[:, c]
    lh, rh = coco_xyz[:, 11], coco_xyz[:, 12]
    both = np.isfinite(lh).all(1) & np.isfinite(rh).all(1)
    out[both, SMPL_PELVIS] = 0.5 * (lh[both] + rh[both])
    return out


def _fill_and_smooth(traj: np.ndarray, dt: float,
                     min_cutoff: float, beta: float):
    """Fill interior NaNs of a (T,3) track (linear), One-Euro smooth, and report
    which frames lie inside the observed span (so we never trust extrapolation).

    Returns (smoothed (T,3), inside (T,) bool).
    """
    T = len(traj)
    idx = np.arange(T)
    finite = np.isfinite(traj).all(1)
    if finite.sum() < 2:
        return traj.copy(), finite
    fi = idx[finite]
    filled = traj.copy()
    for d in range(3):
        filled[:, d] = np.interp(idx, fi, traj[finite, d])
    sm = one_euro(filled, dt, min_cutoff=min_cutoff, beta=beta, zero_phase=True)
    inside = (idx >= fi[0]) & (idx <= fi[-1])
    return sm, inside


def _ease_boundary(out: np.ndarray, source: np.ndarray, edited: np.ndarray,
                   fa: int, fb: int, margin: int) -> np.ndarray:
    """Velocity-matched cosine ease so the corrected segment joins the untouched
    source smoothly. For frames just *outside* [fa,fb] we add a decaying share of
    the boundary correction (edited - source at the keyframe). At the keyframe
    side the step equals the source's own velocity (C1-ish); at the outer edge it
    fades to the pristine source.
    """
    T = out.shape[0]
    L = min(margin, fa)
    if L > 0:
        delta = edited[fa] - source[fa]                      # (J,3)
        for k in range(1, L + 1):                            # i = fa-1 .. fa-L
            w = 0.5 * (1.0 + np.cos(np.pi * k / (L + 1)))    # ~1 near kf -> ~0
            out[fa - k] = source[fa - k] + w * delta
    L2 = min(margin, T - 1 - fb)
    if L2 > 0:
        delta = edited[fb] - source[fb]
        for k in range(1, L2 + 1):                           # i = fb+1 .. fb+L2
            w = 0.5 * (1.0 + np.cos(np.pi * k / (L2 + 1)))
            out[fb + k] = source[fb + k] + w * delta
    return out


def fuse(source: np.ndarray, edited: np.ndarray, cam_smpl: np.ndarray,
         keyframes, frame_a: int, frame_b: int, dt: float,
         method: str = "pchip", min_cutoff: float = 1.0, beta: float = 3.0,
         smooth: float = 0.0, margin: int = 8,
         edited_joints=None, min_observed_frac: float = 0.5):
    """Camera-guided fill of [frame_a..frame_b].

    Args:
        source:  (T,J,3) pristine baseline (used where cameras can't see).
        edited:  (T,J,3) current poses; the keyframes hold the hand edits.
        cam_smpl:(T,J,3) camera-triangulated track in SMPL layout, NaN unseen.
        keyframes: skel-frame knot indices.
        frame_a, frame_b: inclusive corrected range (first..last keyframe).
        dt: seconds per skeleton frame (1/pfps) for the One-Euro smoother.
        method/smooth: forwarded to the offset interpolation.
        margin: frames outside [a,b] used for the boundary ease.
        edited_joints: joints the user dragged; any that the cameras can't drive
            stay in absolute "replace" mode (today's behaviour for those).
        min_observed_frac: a camera-driven joint must be observed on at least
            this fraction of the gap (and on every keyframe) to be trusted; else
            it falls back to the source baseline.

    Returns (out (T,J,3), cam_used dict{smpl_joint: observed_frac}).
    """
    T, J, _ = source.shape
    fa = max(0, min(frame_a, frame_b))
    fb = min(T - 1, max(frame_a, frame_b))
    kfs = [int(k) for k in (keyframes or []) if fa <= k <= fb]

    # 1) Per-joint baseline: the camera track where a joint is camera-driven and
    #    confidently observed (incl. all keyframes), else the original source.
    baseline = source.copy()
    cam_used: dict[int, float] = {}
    span = max(1, fb - fa)
    for j in CAMERA_DRIVEN:
        if j >= J:
            continue
        sm, inside = _fill_and_smooth(cam_smpl[:, j], dt, min_cutoff, beta)
        kf_ok = all(inside[k] for k in kfs) if kfs else False
        frac = float(inside[fa:fb + 1].mean()) if fb > fa else 0.0
        if kf_ok and frac >= min_observed_frac:
            col = baseline[:, j].copy()
            col[inside] = sm[inside]            # ride the camera where it sees
            baseline[:, j] = col
            cam_used[j] = frac

    # Dragged joints the cameras can't drive keep absolute replace mode (their
    # drifting source middle is dropped, as today). Camera-driven joints use the
    # camera baseline in offset mode instead.
    edited_joints = set(edited_joints or [])
    replace = {j for j in edited_joints if j not in cam_used}

    # 2) Offset-anchor the baseline to the keyframes (exact at the keyframes).
    res = interpolate_offsets_all_joints(
        edited, baseline, fa, fb, method,
        keyframes=kfs, replace_joints=replace, smooth=smooth)
    out = edited.copy()
    out[fa:fb + 1] = res

    # 3) Smoothly connect to the untouched source outside [fa,fb].
    out = _ease_boundary(out, source, edited, fa, fb, margin)
    return out, cam_used

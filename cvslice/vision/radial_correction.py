"""Empirical radial de-warp for a single camera.

When a wide-angle / fisheye camera's distortion is under-modelled, the
projected 3D skeleton lands at the wrong radius near the image edges (measured:
predicted points sit too far out, error growing with radius). A proper fix is a
board re-calibration with a fisheye model; when that isn't possible, this
fits the *residual* radial error directly and cancels it at projection time.

It is a display-time correction (applied to 3D->2D projected points), NOT a
physical model — it only fixes what was measured, and is clamped (held flat)
beyond the largest radius that had data, since we can't correct where we never
observed the subject (e.g. the extreme corners).
"""
from __future__ import annotations

import numpy as np

MIN_PER_BIN = 15      # need enough samples per bin to beat ~13px detection noise
MIN_KNOTS = 3


def fit(pred_xy, true_xy, cx: float, cy: float, n_bins: int = 10) -> dict | None:
    """Fit a radial correction from matched (predicted, true) 2D points.

    pred_xy: (N,2) where the calibration *projects* a joint.
    true_xy: (N,2) where that joint actually is in the image (detected).
    Returns a model dict (cx, cy, r knots, delta knots, r_max) or None if the
    data is too sparse. delta[k] = (true_radius - predicted_radius) median in
    that radius bin — the signed amount to move a projected point radially.
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)
    if len(pred) < MIN_PER_BIN * 2:
        return None
    rp = np.hypot(pred[:, 0] - cx, pred[:, 1] - cy)
    rt = np.hypot(true[:, 0] - cx, true[:, 1] - cy)
    delta = rt - rp                      # move predicted radius by this to hit true
    rmax = float(rp.max())
    knots = np.linspace(0.0, rmax, n_bins + 1)
    r_c, d_c = [0.0], [0.0]              # anchor: no correction at the centre
    for a, b in zip(knots[:-1], knots[1:]):
        m = (rp >= a) & (rp < b)
        if int(m.sum()) >= MIN_PER_BIN:
            r_c.append(float((a + b) / 2))
            d_c.append(float(np.median(delta[m])))   # robust per-bin
    if len(r_c) < MIN_KNOTS:
        return None
    # Enforce a monotone non-increasing pull-in: the measured bias is "projected
    # too far out, growing with radius", so the correction should only ever pull
    # inward and never reverse. This also kills per-bin noise blips.
    for k in range(1, len(d_c)):
        d_c[k] = min(d_c[k], d_c[k - 1])
    return {"cx": float(cx), "cy": float(cy),
            "r": r_c, "delta": d_c, "r_max": float(r_c[-1]),
            "n_points": int(len(pred))}


def apply(xy, model: dict | None):
    """Move projected points radially by the fitted delta(r). Vectorised;
    flat-extrapolated (clamped) beyond the last fitted radius."""
    if not model:
        return xy
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cx, cy = model["cx"], model["cy"]
    rr = np.asarray(model["r"], dtype=np.float64)
    dd = np.asarray(model["delta"], dtype=np.float64)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    r = np.hypot(dx, dy)
    delta = np.interp(r, rr, dd)          # np.interp clamps to dd[0]/dd[-1] at ends
    scale = np.where(r > 1e-6, (r + delta) / r, 1.0)
    out = np.empty_like(pts)
    out[:, 0] = cx + dx * scale
    out[:, 1] = cy + dy * scale
    return out.reshape(np.asarray(xy).shape)

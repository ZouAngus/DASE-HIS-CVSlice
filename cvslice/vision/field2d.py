"""Empirical 2D (position-dependent) projection correction for one camera.

``radial_correction`` models the residual as a function of radius only — it is
symmetric about the principal point. Measured on the wide-angle "diagonal"
camera, the distortion is in fact ASYMMETRIC (decentering): at equal radius the
left half of the image has ~2x the residual of the right half. A radius-only
model can only fit a left/right compromise, so it under-corrects the worse side.

This fits the residual as a smooth function of image *position*: a grid of
median (true - predicted) offset vectors, binned by where the calibration
*projects* a joint. It naturally captures left/right (and top/bottom)
asymmetry. Like the radial version it is a display-time correction applied to
projected points, fit only from observed data, and it fades to zero where no
subject was ever seen (the extreme corners), since we can't correct there.
"""
from __future__ import annotations

import numpy as np

MIN_PER_CELL = 8       # min samples for a cell to be trusted
FILL_ITERS = 6         # diffusion steps to extend the field past sparse cells


def fit(pred_xy, true_xy, image_size: tuple[int, int],
        cell: int = 80) -> dict | None:
    """Fit a 2D offset field from matched (predicted, true) 2D points.

    pred_xy: (N,2) where the calibration *projects* a joint.
    true_xy: (N,2) where that joint actually is (detected).
    image_size: (W, H) in pixels. cell: grid cell size in pixels.

    Returns a model dict (grid node coords + dx/dy node arrays) or None if too
    sparse. delta = true - pred is the amount to move a projected point.
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)
    if len(pred) < MIN_PER_CELL * 4:
        return None
    W, H = int(image_size[0]), int(image_size[1])
    delta = true - pred                              # (N,2) move pred by this
    nx = max(2, int(np.ceil(W / cell)))
    ny = max(2, int(np.ceil(H / cell)))

    # Bin points into cells by their PREDICTED position (that's what we correct).
    ix = np.clip((pred[:, 0] / W * nx).astype(int), 0, nx - 1)
    iy = np.clip((pred[:, 1] / H * ny).astype(int), 0, ny - 1)
    dx = np.full((ny, nx), np.nan)
    dy = np.full((ny, nx), np.nan)
    cnt = np.zeros((ny, nx), dtype=int)
    for cyi in range(ny):
        for cxi in range(nx):
            m = (ix == cxi) & (iy == cyi)
            n = int(m.sum())
            cnt[cyi, cxi] = n
            if n >= MIN_PER_CELL:
                dx[cyi, cxi] = np.median(delta[m, 0])
                dy[cyi, cxi] = np.median(delta[m, 1])
    if np.isfinite(dx).sum() < 4:
        return None

    valid = np.isfinite(dx)
    dx, dy = _diffuse_fill(dx), _diffuse_fill(dy)
    # Anything still unfilled (far from any data) gets zero correction.
    dx = np.nan_to_num(dx)
    dy = np.nan_to_num(dy)

    gx = (np.arange(nx) + 0.5) * (W / nx)            # node x centers
    gy = (np.arange(ny) + 0.5) * (H / ny)            # node y centers
    return {"W": W, "H": H, "gx": gx.tolist(), "gy": gy.tolist(),
            "dx": dx.tolist(), "dy": dy.tolist(),
            "valid": valid.tolist(), "n_points": int(len(pred)),
            "n_cells": int(valid.sum())}


def _diffuse_fill(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells by averaging finite 4-neighbours, a few iterations, then
    one light smoothing pass. Spreads the correction smoothly into sparse cells
    (e.g. the left edge) without inventing data far from any observation."""
    g = grid.copy()
    for _ in range(FILL_ITERS):
        nan = ~np.isfinite(g)
        if not nan.any():
            break
        acc = np.zeros_like(g)
        wsum = np.zeros_like(g)
        for dyi, dxi in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(g, dyi, axis=0), dxi, axis=1)
            fin = np.isfinite(sh)
            acc[fin] += sh[fin]
            wsum[fin] += 1.0
        fill = nan & (wsum > 0)
        g[fill] = acc[fill] / wsum[fill]
    # light 3x3 box smooth on the finite field
    fin = np.isfinite(g)
    if fin.all():
        acc = g.copy(); w = np.ones_like(g)
        for dyi in (-1, 0, 1):
            for dxi in (-1, 0, 1):
                if dyi == 0 and dxi == 0:
                    continue
                acc += np.roll(np.roll(g, dyi, axis=0), dxi, axis=1)
                w += 1.0
        g = acc / w
    return g


def apply(xy, model: dict | None):
    """Move projected points by the fitted 2D offset (bilinear sample of the
    node grid). Clamped to the grid extent; zero where the field is zero."""
    if not model:
        return xy
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    gx = np.asarray(model["gx"], dtype=np.float64)
    gy = np.asarray(model["gy"], dtype=np.float64)
    dx = np.asarray(model["dx"], dtype=np.float64)
    dy = np.asarray(model["dy"], dtype=np.float64)

    fx = np.interp(pts[:, 0], gx, np.arange(len(gx)))   # fractional col index
    fy = np.interp(pts[:, 1], gy, np.arange(len(gy)))   # fractional row index
    x0 = np.clip(np.floor(fx).astype(int), 0, len(gx) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(gy) - 1)
    x1 = np.clip(x0 + 1, 0, len(gx) - 1)
    y1 = np.clip(y0 + 1, 0, len(gy) - 1)
    tx = np.clip(fx - x0, 0.0, 1.0)
    ty = np.clip(fy - y0, 0.0, 1.0)

    def bilerp(g):
        return ((g[y0, x0] * (1 - tx) + g[y0, x1] * tx) * (1 - ty) +
                (g[y1, x0] * (1 - tx) + g[y1, x1] * tx) * ty)

    out = pts.copy()
    out[:, 0] += bilerp(dx)
    out[:, 1] += bilerp(dy)
    return out.reshape(np.asarray(xy).shape)

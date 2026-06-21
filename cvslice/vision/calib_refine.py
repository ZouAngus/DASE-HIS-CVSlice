"""Calibration refinement from manually-aligned 2D-3D correspondences.

The Skeleton Corrector lets the user drag a projected joint to its true pixel
location. Each such drag yields one correspondence:

    (3D world point  ->  observed 2D pixel)   for a given camera and frame.

Because a camera's pose is static across frames, correspondences pooled from
many frames over-constrain the camera, so we can re-solve its parameters by
minimising reprojection error — i.e. bundle adjustment. We use OpenCV's
Levenberg-Marquardt solvers (no SciPy dependency):

  * ``"extrinsic"`` mode: keep K/distortion, re-solve pose with ``solvePnP``
    (+ LM refine). Lowest risk; fixes a mis-estimated camera pose.
  * ``"full"`` mode: also refine intrinsics + distortion with
    ``calibrateCamera`` (each frame is a view of the moving 3D structure),
    then re-solve a single pose with the refined intrinsics. Best for edge
    inaccuracy, which is usually a distortion-model limitation.

The 3D points are treated as fixed observations. They were triangulated from
the same calibration, so refining against them is mildly circular; we mitigate
that by (a) requiring many, well-spread correspondences and (b) never writing a
result that does not reduce reprojection error.
"""
from __future__ import annotations

import cv2
import numpy as np


# Minimum correspondences to attempt any solve, and minimum frames ("views")
# before we trust an intrinsics/distortion refine.
MIN_POINTS = 6
MIN_VIEWS_FOR_INTRINSICS = 4


def _reproj_rmse(obj: np.ndarray, img: np.ndarray, K, dist, rvec, tvec) -> float:
    """Root-mean-square reprojection error (pixels) over all points."""
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    d = proj - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _point_residuals(obj, img, K, dist, rvec, tvec) -> np.ndarray:
    """Per-point reprojection error (pixels)."""
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec,
                                K, np.asarray(dist).reshape(1, -1))
    return np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)


def refine_camera(obj_by_frame: list[np.ndarray],
                  img_by_frame: list[np.ndarray],
                  K0: np.ndarray, dist0: np.ndarray,
                  rvec0: np.ndarray, tvec0: np.ndarray,
                  image_size: tuple[int, int],
                  mode: str = "full", rational: bool = False) -> dict:
    """Refine one camera from per-frame 2D-3D correspondences.

    Args:
        obj_by_frame: list of (Ni, 3) world points, one array per frame.
        img_by_frame: list of (Ni, 2) observed pixels, matching obj_by_frame.
        K0, dist0: current intrinsics (3x3) and distortion (1x5 or 5,).
        rvec0, tvec0: current extrinsic as Rodrigues vec (3,) and tvec (3,).
        image_size: (width, height) in pixels.
        mode: "full" (K+dist+pose) or "extrinsic" (pose only).
        rational: in "full" mode, also try the 8-param rational distortion
            model (k1..k6) — needed to fit wide-angle / fisheye edge
            distortion that the 5-param Brown model can't. The best-RMSE
            candidate (extrinsic / full / full-rational) is kept.

    Returns a dict:
        ok, reason, mode_used, n_points, n_frames,
        rmse_before, rmse_after, improved,
        K, dist, rvec, tvec   (refined; numpy arrays)
    """
    obj_by_frame = [np.asarray(o, dtype=np.float64).reshape(-1, 3)
                    for o in obj_by_frame if len(o) > 0]
    img_by_frame = [np.asarray(i, dtype=np.float64).reshape(-1, 2)
                    for i in img_by_frame if len(i) > 0]
    K0 = np.asarray(K0, dtype=np.float64).reshape(3, 3)
    dist0 = np.asarray(dist0, dtype=np.float64).reshape(-1)
    if dist0.size < 5:
        dist0 = np.pad(dist0, (0, 5 - dist0.size))
    rvec0 = np.asarray(rvec0, dtype=np.float64).reshape(3, 1)
    tvec0 = np.asarray(tvec0, dtype=np.float64).reshape(3, 1)

    obj_all = np.concatenate(obj_by_frame, axis=0) if obj_by_frame else np.empty((0, 3))
    img_all = np.concatenate(img_by_frame, axis=0) if img_by_frame else np.empty((0, 2))
    n_points = len(obj_all)
    n_frames = len(obj_by_frame)

    result = {
        "ok": False, "reason": "", "mode_used": mode,
        "n_points": n_points, "n_frames": n_frames,
        "rmse_before": float("nan"), "rmse_after": float("nan"),
        "improved": False,
        "K": K0.copy(), "dist": dist0.copy(),
        "rvec": rvec0.copy(), "tvec": tvec0.copy(),
    }

    if n_points < MIN_POINTS:
        result["reason"] = f"correspondences too few ({n_points} < {MIN_POINTS})"
        return result

    # --- Robust inlier selection -----------------------------------------
    # Auto 2D detections produce outliers (left/right swaps, occluded joints).
    # An initial pose lets us drop gross outliers (MAD rule) so the refine
    # isn't dragged by them — this is what lets the distortion fit actually
    # help instead of fighting noise.
    ok0, rvi, tvi = cv2.solvePnP(
        obj_all.reshape(-1, 1, 3), img_all.reshape(-1, 1, 2),
        K0, dist0.reshape(1, -1), rvec0.copy(), tvec0.copy(),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok0:
        res0 = _point_residuals(obj_all, img_all, K0, dist0, rvi, tvi)
        med = float(np.median(res0))
        mad = float(np.median(np.abs(res0 - med))) + 1e-6
        keep = res0 <= max(med + 2.5 * mad, 8.0)
        if keep.sum() >= MIN_POINTS and keep.mean() >= 0.3:
            new_obj, new_img, off = [], [], 0
            for o, im in zip(obj_by_frame, img_by_frame):
                m = keep[off:off + len(o)]
                off += len(o)
                if m.any():
                    new_obj.append(o[m])
                    new_img.append(im[m])
            obj_by_frame, img_by_frame = new_obj, new_img
            obj_all = np.concatenate(obj_by_frame, axis=0)
            img_all = np.concatenate(img_by_frame, axis=0)
            n_points, n_frames = len(obj_all), len(obj_by_frame)
            result["n_points"], result["n_frames"] = n_points, n_frames

    rmse_before = _reproj_rmse(obj_all, img_all, K0, dist0, rvec0, tvec0)
    result["rmse_before"] = rmse_before

    def _solve_pose(K, dist):
        """solvePnP (+ LM refine) over all pooled points; return (rvec, tvec,
        rmse) or None on failure."""
        ok, rvec, tvec = cv2.solvePnP(
            obj_all.reshape(-1, 1, 3), img_all.reshape(-1, 1, 2),
            K, dist.reshape(1, -1), rvec0.copy(), tvec0.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_all.reshape(-1, 1, 3), img_all.reshape(-1, 1, 2),
                K, dist.reshape(1, -1), rvec, tvec)
        except cv2.error:
            pass
        return rvec, tvec, _reproj_rmse(obj_all, img_all, K, dist, rvec, tvec)

    # Build candidate solutions and keep whichever minimises reprojection.
    # The extrinsic-only solve is always safe (solvePnP can't do worse than
    # the initial guess); the full intrinsics refit can over-fit sparse,
    # moving-subject data (esp. fisheye), so we never blindly trust it.
    candidates = []  # (mode_used, K, dist, rvec, tvec, rmse)

    r = _solve_pose(K0, dist0)
    if r is not None:
        candidates.append(("extrinsic", K0, dist0, r[0], r[1], r[2]))

    if mode == "full" and n_frames >= MIN_VIEWS_FOR_INTRINSICS:
        objs = [o.astype(np.float32).reshape(-1, 1, 3) for o in obj_by_frame]
        imgs = [i.astype(np.float32).reshape(-1, 1, 2) for i in img_by_frame]
        imsz = tuple(int(s) for s in image_size)

        def _calib(flags, ncoef, label):
            d0 = np.pad(dist0, (0, max(0, ncoef - dist0.size)))[:ncoef]
            try:
                _ret, Kc, distc, _rv, _tv = cv2.calibrateCamera(
                    objs, imgs, imsz, K0.copy(), d0.reshape(1, -1), flags=flags)
            except cv2.error:
                return
            if not (np.all(np.isfinite(Kc)) and np.all(np.isfinite(distc))):
                return
            rf = _solve_pose(Kc, distc.reshape(-1))
            if rf is not None:
                candidates.append((label, Kc, distc.reshape(-1),
                                   rf[0], rf[1], rf[2]))

        _calib(cv2.CALIB_USE_INTRINSIC_GUESS, 5, "full")
        if rational:
            _calib(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL,
                   8, "full+rational")

    if not candidates:
        result["reason"] = "solvePnP failed"
        return result

    mode_used, K, dist, rvec, tvec, rmse_after = min(
        candidates, key=lambda c: c[5])
    result.update({
        "ok": True, "mode_used": mode_used,
        "rmse_after": rmse_after,
        "improved": rmse_after < rmse_before,
        "K": np.asarray(K, dtype=np.float64), "dist": np.asarray(dist).reshape(-1),
        "rvec": np.asarray(rvec).reshape(3), "tvec": np.asarray(tvec).reshape(3),
    })
    return result


def extrinsic_matrix_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> list[list[float]]:
    """Build a 3x4 [R|t] list (the ``best_extrinsic`` format) from rvec/tvec."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R, t])
    return Rt.tolist()

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


def refine_camera(obj_by_frame: list[np.ndarray],
                  img_by_frame: list[np.ndarray],
                  K0: np.ndarray, dist0: np.ndarray,
                  rvec0: np.ndarray, tvec0: np.ndarray,
                  image_size: tuple[int, int],
                  mode: str = "full") -> dict:
    """Refine one camera from per-frame 2D-3D correspondences.

    Args:
        obj_by_frame: list of (Ni, 3) world points, one array per frame.
        img_by_frame: list of (Ni, 2) observed pixels, matching obj_by_frame.
        K0, dist0: current intrinsics (3x3) and distortion (1x5 or 5,).
        rvec0, tvec0: current extrinsic as Rodrigues vec (3,) and tvec (3,).
        image_size: (width, height) in pixels.
        mode: "full" (K+dist+pose) or "extrinsic" (pose only).

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

    result["rmse_before"] = _reproj_rmse(obj_all, img_all, K0, dist0, rvec0, tvec0)

    K = K0.copy()
    dist = dist0.copy()
    mode_used = mode

    # --- Optional intrinsics + distortion refine -------------------------
    if mode == "full":
        if n_frames < MIN_VIEWS_FOR_INTRINSICS:
            # Not enough independent views to trust intrinsics; fall back.
            mode_used = "extrinsic"
        else:
            objs = [o.astype(np.float32).reshape(-1, 1, 3) for o in obj_by_frame]
            imgs = [i.astype(np.float32).reshape(-1, 1, 2) for i in img_by_frame]
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            try:
                ret, Kc, distc, _rv, _tv = cv2.calibrateCamera(
                    objs, imgs, tuple(int(s) for s in image_size),
                    K0.copy(), dist0.copy().reshape(1, -1), flags=flags)
                if np.all(np.isfinite(Kc)) and np.all(np.isfinite(distc)):
                    K, dist = Kc, distc.reshape(-1)
            except cv2.error:
                mode_used = "extrinsic"  # solver failed; keep K/dist

    # --- Pose refine (always) -------------------------------------------
    ok, rvec, tvec = cv2.solvePnP(
        obj_all.reshape(-1, 1, 3), img_all.reshape(-1, 1, 2),
        K, dist.reshape(1, -1), rvec0.copy(), tvec0.copy(),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        result["reason"] = "solvePnP failed"
        return result
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_all.reshape(-1, 1, 3), img_all.reshape(-1, 1, 2),
            K, dist.reshape(1, -1), rvec, tvec)
    except cv2.error:
        pass

    rmse_after = _reproj_rmse(obj_all, img_all, K, dist, rvec, tvec)
    result.update({
        "ok": True, "mode_used": mode_used,
        "rmse_after": rmse_after,
        "improved": rmse_after < result["rmse_before"],
        "K": K, "dist": dist.reshape(-1),
        "rvec": np.asarray(rvec).reshape(3), "tvec": np.asarray(tvec).reshape(3),
    })
    return result


def extrinsic_matrix_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> list[list[float]]:
    """Build a 3x4 [R|t] list (the ``best_extrinsic`` format) from rvec/tvec."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R, t])
    return Rt.tolist()

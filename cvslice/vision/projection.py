"""3D-to-2D projection and skeleton rendering."""
import cv2
import numpy as np
from ..core.constants import JOINT_PAIRS_MAP, PT_COLOR

# Module-level projection cache: {id(extr): (rvec, tvec, camera_matrix, dist_coeffs)}
_proj_cache: dict = {}


def clear_projection_cache():
    """Clear the cached projection parameters (call when calibration changes)."""
    _proj_cache.clear()


def _rvec_tvec(extr: dict):
    """Extract rotation vector and translation vector from extrinsic dict."""
    ext = None
    for k in ("best_extrinsic", "extrinsic", "extrinsics"):
        if k not in extr:
            continue
        v = extr[k]
        if k == "extrinsics" and isinstance(v, list) and v:
            v = v[0]
        ext = np.array(v, dtype=float)
        break
    if ext is None:
        return None, None
    if ext.shape == (4, 4):
        ext = ext[:3, :]
    if ext.shape != (3, 4):
        return None, None
    R = ext[:, :3]
    t = ext[:, 3].reshape(3, 1)
    rv, _ = cv2.Rodrigues(R)
    return rv, t


def project_pts(pts3d: np.ndarray, intr: dict, extr: dict,
                flip_x=False, flip_y=False, flip_z=False) -> np.ndarray | None:
    """Project 3D joint positions to 2D pixel coordinates.

    Uses a per-extrinsic cache for rvec/tvec/camera_matrix/dist_coeffs.
    """
    pts = pts3d.copy()
    if flip_x:
        pts[:, 0] *= -1
    if flip_y:
        pts[:, 1] *= -1
    if flip_z:
        pts[:, 2] *= -1

    cache_key = id(extr)
    if cache_key in _proj_cache:
        rv, tv, cm, dc = _proj_cache[cache_key]
    else:
        rv, tv = _rvec_tvec(extr)
        if rv is None:
            return None
        cm = np.array(intr["camera_matrix"], dtype=np.float64)
        dc_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
        dc = (np.array(dc_raw, dtype=np.float64).reshape(-1)
              if dc_raw is not None
              else np.zeros(5, dtype=np.float64))
        _proj_cache[cache_key] = (rv, tv, cm, dc)

    proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rv, tv, cm, dc)
    return proj.squeeze().astype(np.int32)


def draw_skel(frame: np.ndarray, proj: np.ndarray,
              color: tuple = PT_COLOR) -> None:
    """Draw skeleton joints and bones on a frame."""
    h, w = frame.shape[:2]
    n = len(proj)
    bc = tuple(int(c * 0.7) for c in color)
    for pt in proj:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 4, color, -1)
    for i, j in JOINT_PAIRS_MAP.get(n, []):
        if i < n and j < n:
            x1, y1 = int(proj[i][0]), int(proj[i][1])
            x2, y2 = int(proj[j][0]), int(proj[j][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), bc, 2)

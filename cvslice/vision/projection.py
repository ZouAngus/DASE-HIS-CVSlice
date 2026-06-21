"""3D-to-2D projection and skeleton rendering."""
import cv2
import numpy as np
from ..core.constants import (
    CENTER_COLOR, JOINT_PAIRS_MAP, LEFT_COLOR, LEFT_JOINTS, PT_COLOR,
    RIGHT_COLOR, RIGHT_JOINTS,
)
from . import radial_correction, field2d

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
    proj = proj.reshape(-1, 2)
    # Optional empirical de-warp for under-modelled wide-angle cameras. The 2D
    # position field handles asymmetric (decentering) distortion; the radial
    # model is the older symmetric fallback. Apply whichever is present.
    if isinstance(extr, dict):
        corr = extr.get("radial_correction")
        if corr:
            proj = radial_correction.apply(proj, corr)
        field = extr.get("correction_field")
        if field:
            proj = field2d.apply(proj, field)
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


# Colors for interpolated (low-confidence) joints
_INTERP_PT_COLOR = (0, 200, 255)    # orange-yellow for interpolated joints
_INTERP_BONE_COLOR = (0, 140, 180)  # darker orange for interpolated bones


def _joint_side_color(idx: int, left: set, right: set, default: tuple) -> tuple:
    """SMPL-style side color for a single joint index."""
    if idx in left:
        return LEFT_COLOR
    if idx in right:
        return RIGHT_COLOR
    if left or right:          # known topology, but a center joint
        return CENTER_COLOR
    return default


def _bone_side_color(i: int, j: int, left: set, right: set,
                     default: tuple) -> tuple:
    """Side color for a bone. Limb-colored only when BOTH ends are on that
    limb; connector bones (e.g. pelvis->hip, spine->collar->shoulder) stay
    center, matching the reference figure's blue torso."""
    if i in left and j in left:
        return LEFT_COLOR
    if i in right and j in right:
        return RIGHT_COLOR
    if left or right:
        return CENTER_COLOR
    return tuple(int(c * 0.7) for c in default)


def draw_skel_with_confidence(frame: np.ndarray, proj: np.ndarray,
                              nan_mask: np.ndarray | None = None,
                              color: tuple = PT_COLOR,
                              side_colors: bool = True,
                              skip_dots: set | None = None) -> None:
    """Draw skeleton with SMPL-style left/right side coloring.

    Args:
        frame: BGR image to draw on.
        proj: (J, 2) projected 2D joint positions.
        nan_mask: (J,) boolean — True if this joint was originally NaN
                  (interpolated). Interpolated joints override the side color
                  with a hollow warning marker. If None, all joints confident.
        color: Fallback color for confident joints on unknown topologies.
        side_colors: color left/right limbs differently (per JOINT side maps).
        skip_dots: joint indices whose *dot* is not drawn (bones still are).
                   Used by calibration-refine to hide a corrected joint's old
                   point while keeping the skeleton's connecting lines.
    """
    h, w = frame.shape[:2]
    n = len(proj)
    skip = skip_dots or set()
    left = LEFT_JOINTS.get(n, set()) if side_colors else set()
    right = RIGHT_JOINTS.get(n, set()) if side_colors else set()

    # Draw joints
    for idx, pt in enumerate(proj):
        if idx in skip:
            continue
        x, y = int(pt[0]), int(pt[1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        if nan_mask is not None and idx < len(nan_mask) and nan_mask[idx]:
            # Interpolated joint: hollow circle in warning color
            cv2.circle(frame, (x, y), 5, _INTERP_PT_COLOR, 2)
        else:
            cv2.circle(frame, (x, y), 4,
                       _joint_side_color(idx, left, right, color), -1)

    # Draw bones (always — independent of skip_dots)
    for i, j in JOINT_PAIRS_MAP.get(n, []):
        if i >= n or j >= n:
            continue
        x1, y1 = int(proj[i][0]), int(proj[i][1])
        x2, y2 = int(proj[j][0]), int(proj[j][1])
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue
        either_interp = (nan_mask is not None and
                         ((i < len(nan_mask) and nan_mask[i]) or
                          (j < len(nan_mask) and nan_mask[j])))
        if either_interp:
            # Dashed-style: draw thinner, different color
            cv2.line(frame, (x1, y1), (x2, y2), _INTERP_BONE_COLOR, 1)
        else:
            cv2.line(frame, (x1, y1), (x2, y2),
                     _bone_side_color(i, j, left, right, color), 2)

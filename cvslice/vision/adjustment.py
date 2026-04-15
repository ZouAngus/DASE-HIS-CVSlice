"""3D joint manual adjustment via 2D drag with depth-preserving unprojection.

When a user drags a joint in the 2D view, we:
1. Keep the joint's depth in camera coordinates (z_cam) fixed
2. Compute the new 2D position from the drag
3. Unproject (u', v', z_cam) back to 3D world coordinates
4. Update the 3D point in-place

This is geometrically exact for the current camera view — the joint moves
along the camera's image plane at its current depth.
"""
import cv2
import numpy as np


def _undistort_point(u: float, v: float,
                     K: np.ndarray, dist_coeffs: np.ndarray | None
                     ) -> tuple[float, float]:
    """Undistort a single 2D pixel coordinate.

    Returns the ideal (undistorted) pixel coordinates that can be safely
    used with K_inv for unprojection.
    """
    if dist_coeffs is None or np.allclose(dist_coeffs, 0):
        return u, v
    pts = np.array([[[u, v]]], dtype=np.float64)
    out = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def unproject_2d_to_3d(u: float, v: float, z_cam: float,
                       K: np.ndarray, R: np.ndarray, t: np.ndarray,
                       dist_coeffs: np.ndarray | None = None) -> np.ndarray:
    """Unproject a 2D pixel (u, v) to 3D world coordinates at a given camera depth.

    If dist_coeffs is provided, the pixel is first undistorted so that the
    unprojection is accurate even near image edges with strong lens distortion.

    Args:
        u, v: Pixel coordinates (possibly distorted).
        z_cam: Depth in camera coordinate frame.
        K: (3, 3) camera intrinsic matrix.
        R: (3, 3) rotation matrix (world -> camera).
        t: (3,) translation vector (world -> camera).
        dist_coeffs: Distortion coefficients (same format as OpenCV).

    Returns:
        (3,) 3D point in world coordinates.
    """
    u_corr, v_corr = _undistort_point(u, v, K, dist_coeffs)
    K_inv = np.linalg.inv(K)
    p_cam = z_cam * K_inv @ np.array([u_corr, v_corr, 1.0])
    p_world = R.T @ (p_cam - t)
    return p_world


def get_camera_depth(pt3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """Get the depth of a 3D world point in camera coordinates."""
    p_cam = R @ pt3d + t
    return float(p_cam[2])


def extract_R_t(extr: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract R, t from an extrinsic dict."""
    ext = None
    for k in ("best_extrinsic", "extrinsic", "extrinsics"):
        if k not in extr:
            continue
        v = extr[k]
        if k == "extrinsics" and isinstance(v, list) and v:
            v = v[0]
        ext = np.array(v, dtype=np.float64)
        break
    if ext is None:
        return None
    if ext.shape == (4, 4):
        ext = ext[:3, :]
    if ext.shape != (3, 4):
        return None
    R = ext[:, :3]
    t = ext[:, 3]
    return R, t


def compute_ray(u: float, v: float,
                K: np.ndarray, R: np.ndarray, t: np.ndarray,
                dist_coeffs: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 3D ray from a 2D pixel coordinate.

    If dist_coeffs is provided, the pixel is first undistorted.

    Returns:
        origin: (3,) camera center in world coordinates.
        direction: (3,) unit direction vector in world coordinates.
    """
    u_corr, v_corr = _undistort_point(u, v, K, dist_coeffs)
    K_inv = np.linalg.inv(K)
    origin = -R.T @ t
    d_cam = K_inv @ np.array([u_corr, v_corr, 1.0])
    d_world = R.T @ d_cam
    d_world = d_world / np.linalg.norm(d_world)
    return origin, d_world


def triangulate_two_rays(o1: np.ndarray, d1: np.ndarray,
                         o2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Find the midpoint of the closest approach of two 3D rays.

    Each ray: P = o + t*d.  Returns the 3D point that best satisfies both.
    """
    # Solve for t1, t2 that minimize |o1 + t1*d1 - o2 - t2*d2|
    w0 = o1 - o2
    a = float(d1 @ d1)  # always 1 if normalized, but be safe
    b = float(d1 @ d2)
    c = float(d2 @ d2)
    d = float(d1 @ w0)
    e = float(d2 @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        # Rays are parallel — fall back to midpoint at closest approach
        t1 = 0.0
        t2 = e / c if abs(c) > 1e-12 else 0.0
    else:
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom
    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    return 0.5 * (p1 + p2)


# Joint selection radius in pixels
PICK_RADIUS = 15


def find_nearest_joint(click_x: int, click_y: int,
                       proj: np.ndarray) -> int | None:
    """Find the joint index nearest to (click_x, click_y) within PICK_RADIUS.

    Returns joint index or None.
    """
    if proj is None or len(proj) == 0:
        return None
    dists = np.sqrt((proj[:, 0] - click_x) ** 2 + (proj[:, 1] - click_y) ** 2)
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= PICK_RADIUS:
        return min_idx
    return None

"""Two-view triangulation and a circularity-free calibration check.

Given two calibrated cameras and matched 2D detections of the *same* physical
points (e.g. a 2D pose model's keypoints in both views at one frame), we
triangulate to 3D and measure how well the two cameras agree:

* reprojection residual — project the triangulated 3D back into each view and
  measure the pixel distance to the original detection;
* (optional) epipolar distance — point-to-epipolar-line distance.

Crucially this never uses the MoSh/SMPL 3D — only the two cameras' own 2D
observations — so it scores the *relative* calibration (extrinsics + intrinsics
+ distortion) directly, with no circular dependency on the skeleton that was
itself derived from these cameras.
"""
from __future__ import annotations

import cv2
import numpy as np


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """3x4 pinhole projection matrix P = K [R|t]."""
    return K @ np.hstack([R, t.reshape(3, 1)])


def _undistort_to_pixels(pts: np.ndarray, K: np.ndarray,
                         dist: np.ndarray) -> np.ndarray:
    """Map distorted pixel detections to ideal (pinhole) pixel coords, so a
    plain P = K[R|t] is valid for triangulation."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2)


def triangulate_pair(pts1: np.ndarray, pts2: np.ndarray,
                     K1, d1, R1, t1, K2, d2, R2, t2) -> np.ndarray:
    """Triangulate matched 2D points from two calibrated views.

    pts1/pts2: (N, 2) pixel detections of the same N points in each view.
    Returns (N, 3) world-frame 3D points.
    """
    K1 = np.asarray(K1, float).reshape(3, 3)
    K2 = np.asarray(K2, float).reshape(3, 3)
    d1 = np.asarray(d1, float).reshape(-1)
    d2 = np.asarray(d2, float).reshape(-1)
    R1 = np.asarray(R1, float).reshape(3, 3)
    R2 = np.asarray(R2, float).reshape(3, 3)
    t1 = np.asarray(t1, float).reshape(3, 1)
    t2 = np.asarray(t2, float).reshape(3, 1)

    p1 = _undistort_to_pixels(pts1, K1, d1)
    p2 = _undistort_to_pixels(pts2, K2, d2)
    P1 = projection_matrix(K1, R1, t1)
    P2 = projection_matrix(K2, R2, t2)
    Xh = cv2.triangulatePoints(P1, P2, p1.T, p2.T)   # 4 x N homogeneous
    w = Xh[3]
    w[np.abs(w) < 1e-12] = 1e-12
    return (Xh[:3] / w).T


def triangulate_multiview(pts2d, cams) -> np.ndarray:
    """Linear DLT triangulation of ONE 3D point from V>=2 calibrated views.

    pts2d: (V, 2) pixel detections of the same point. cams: list of V tuples
    (K, dist, R, t). Returns the (3,) world point. Distortion is removed
    per view before building the DLT system.
    """
    rows = []
    for (uv, cam) in zip(pts2d, cams):
        K, dist, R, t = cam
        K = np.asarray(K, float).reshape(3, 3)
        R = np.asarray(R, float).reshape(3, 3)
        t = np.asarray(t, float).reshape(3, 1)
        p = _undistort_to_pixels(np.asarray(uv, float).reshape(1, 2),
                                 K, np.asarray(dist, float).reshape(-1))[0]
        P = projection_matrix(K, R, t)
        rows.append(p[0] * P[2] - P[0])
        rows.append(p[1] * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        X[3] = 1e-12
    return X[:3] / X[3]


def reprojection_residuals(X: np.ndarray, pts: np.ndarray,
                           K, dist, R, t) -> np.ndarray:
    """Per-point reprojection error (pixels) of 3D points X into one view."""
    X = np.asarray(X, float).reshape(-1, 1, 3)
    K = np.asarray(K, float).reshape(3, 3)
    dist = np.asarray(dist, float).reshape(-1)
    rvec, _ = cv2.Rodrigues(np.asarray(R, float).reshape(3, 3))
    proj, _ = cv2.projectPoints(X, rvec, np.asarray(t, float).reshape(3, 1),
                                K, dist)
    proj = proj.reshape(-1, 2)
    return np.linalg.norm(proj - np.asarray(pts, float).reshape(-1, 2), axis=1)


def pair_consistency(pts1: np.ndarray, pts2: np.ndarray,
                     cam1: tuple, cam2: tuple) -> dict:
    """Triangulate matched points and report per-view reprojection residuals.

    cam1/cam2: (K, dist, R, t). pts1/pts2: (N, 2) matched detections.
    Returns: X (N,3), res1/res2 (N,) per-view px residuals, in_front (N,) bool
    cheirality mask (point in front of both cameras).
    """
    K1, d1, R1, t1 = cam1
    K2, d2, R2, t2 = cam2
    X = triangulate_pair(pts1, pts2, K1, d1, R1, t1, K2, d2, R2, t2)
    res1 = reprojection_residuals(X, pts1, K1, d1, R1, t1)
    res2 = reprojection_residuals(X, pts2, K2, d2, R2, t2)
    # cheirality: depth (Z in camera frame) positive in both views
    R1 = np.asarray(R1, float).reshape(3, 3); t1 = np.asarray(t1, float).reshape(3)
    R2 = np.asarray(R2, float).reshape(3, 3); t2 = np.asarray(t2, float).reshape(3)
    z1 = (X @ R1[2]) + t1[2]
    z2 = (X @ R2[2]) + t2[2]
    in_front = (z1 > 0) & (z2 > 0)
    return {"X": X, "res1": res1, "res2": res2, "in_front": in_front}

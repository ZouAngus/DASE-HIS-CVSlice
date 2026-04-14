"""Multi-frame 3D joint propagation via keyframe anchors.

Supports:
1. Anchor-based spline interpolation: user sets anchors at specific frames,
   intermediate frames get smooth interpolated corrections.
2. Bulk offset: apply a constant delta to a joint across a frame range.
3. Smooth offset: apply a delta that tapers off toward range edges.
"""
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AnchorSet:
    """Manages keyframe anchors for a single joint.

    Each anchor stores (frame_index, target_xyz) — the desired 3D position
    at that frame after user editing.
    """

    def __init__(self):
        # {joint_idx: {frame_idx: np.array([x, y, z])}}
        self._anchors: dict[int, dict[int, np.ndarray]] = {}

    def set_anchor(self, joint: int, frame: int, xyz: np.ndarray):
        self._anchors.setdefault(joint, {})[frame] = xyz.copy()

    def remove_anchor(self, joint: int, frame: int):
        if joint in self._anchors:
            self._anchors[joint].pop(frame, None)
            if not self._anchors[joint]:
                del self._anchors[joint]

    def clear_joint(self, joint: int):
        self._anchors.pop(joint, None)

    def clear_all(self):
        self._anchors.clear()

    def get_anchors(self, joint: int) -> dict[int, np.ndarray]:
        """Return {frame: xyz} for a joint."""
        return dict(self._anchors.get(joint, {}))

    def all_joints(self) -> list[int]:
        return sorted(self._anchors.keys())

    def anchor_count(self) -> int:
        return sum(len(v) for v in self._anchors.values())

    def summary(self) -> list[tuple[int, int, np.ndarray]]:
        """Return [(joint, frame, xyz), ...] sorted by joint then frame."""
        out = []
        for j in sorted(self._anchors):
            for f in sorted(self._anchors[j]):
                out.append((j, f, self._anchors[j][f]))
        return out


def interpolate_anchors(pts3d: np.ndarray, joint: int,
                        anchors: dict[int, np.ndarray],
                        frame_start: int, frame_end: int,
                        method: str = "spline") -> np.ndarray:
    """Compute interpolated 3D positions for a joint between anchors.

    Args:
        pts3d: (T, J, 3) full point array (read-only, used for boundary values).
        joint: Joint index.
        anchors: {frame_idx: target_xyz} — at least 1 anchor required.
        frame_start: First frame of the range to interpolate (inclusive).
        frame_end: Last frame of the range to interpolate (inclusive).
        method: "spline" (cubic, needs scipy) or "linear".

    Returns:
        (N, 3) array of interpolated positions for frames [frame_start..frame_end].
    """
    if not anchors:
        return pts3d[frame_start:frame_end + 1, joint].copy()

    T = pts3d.shape[0]
    frame_start = max(0, frame_start)
    frame_end = min(T - 1, frame_end)
    n_frames = frame_end - frame_start + 1

    # Build knot points: anchors + boundary frames (original values)
    sorted_frames = sorted(anchors.keys())
    knot_frames = []
    knot_values = []

    # Add left boundary if not an anchor
    if frame_start not in anchors and frame_start < sorted_frames[0]:
        knot_frames.append(frame_start)
        knot_values.append(pts3d[frame_start, joint].copy())

    for f in sorted_frames:
        if frame_start <= f <= frame_end:
            knot_frames.append(f)
            knot_values.append(anchors[f].copy())

    # Add right boundary if not an anchor
    if frame_end not in anchors and frame_end > sorted_frames[-1]:
        knot_frames.append(frame_end)
        knot_values.append(pts3d[frame_end, joint].copy())

    knot_frames = np.array(knot_frames)
    knot_values = np.array(knot_values)  # (K, 3)

    if len(knot_frames) < 2:
        # Single anchor — hold its value across the range
        result = np.tile(knot_values[0], (n_frames, 1))
        return result

    target_frames = np.arange(frame_start, frame_end + 1)

    if method == "spline" and HAS_SCIPY and len(knot_frames) >= 2:
        result = np.zeros((n_frames, 3))
        for d in range(3):
            if len(knot_frames) >= 4:
                cs = CubicSpline(knot_frames, knot_values[:, d],
                                 bc_type='natural')
            else:
                # Not enough points for natural cubic — use clamped or linear
                cs = CubicSpline(knot_frames, knot_values[:, d],
                                 bc_type='clamped')
            result[:, d] = cs(target_frames)
        return result
    else:
        # Linear interpolation per axis
        result = np.zeros((n_frames, 3))
        for d in range(3):
            result[:, d] = np.interp(target_frames, knot_frames,
                                     knot_values[:, d])
        return result


def apply_bulk_offset(pts3d: np.ndarray, joint: int,
                      frame_start: int, frame_end: int,
                      delta: np.ndarray,
                      taper: str = "none") -> np.ndarray:
    """Apply a constant or tapered offset to a joint across a frame range.

    Args:
        pts3d: (T, J, 3) — will NOT be modified in-place.
        joint: Joint index.
        frame_start, frame_end: Inclusive range.
        delta: (3,) offset vector.
        taper: "none" (constant), "linear" (fade at edges), "cosine" (smooth fade).

    Returns:
        (N, 3) new positions for frames [frame_start..frame_end].
    """
    T = pts3d.shape[0]
    frame_start = max(0, frame_start)
    frame_end = min(T - 1, frame_end)
    n = frame_end - frame_start + 1

    original = pts3d[frame_start:frame_end + 1, joint].copy()

    if taper == "none":
        weights = np.ones(n)
    elif taper == "linear":
        # Triangle: 0 at edges, 1 at center
        half = n / 2.0
        weights = np.minimum(np.arange(n), np.arange(n - 1, -1, -1)) / half
        weights = np.clip(weights, 0, 1)
    elif taper == "cosine":
        # Smooth cosine taper
        t = np.linspace(0, np.pi, n)
        weights = 0.5 * (1 - np.cos(t))
        # Normalize so peak = 1
        if weights.max() > 0:
            weights /= weights.max()
    else:
        weights = np.ones(n)

    return original + delta[None, :] * weights[:, None]


def interpolate_all_joints(pts3d: np.ndarray,
                           frame_a: int, frame_b: int,
                           method: str = "spline",
                           keyframes: list[int] | None = None) -> np.ndarray:
    """Interpolate ALL joints between keyframes.

    Treats each keyframe as an anchor (its current position is correct),
    and smoothly interpolates every joint for all frames in between.

    Args:
        pts3d: (T, J, 3) full point array (read-only).
        frame_a: First frame of range (inclusive).
        frame_b: Last frame of range (inclusive).
        method: "spline" or "linear".
        keyframes: Optional list of intermediate keyframe indices.
            If None, only frame_a and frame_b are used as anchors.

    Returns:
        (N, J, 3) interpolated positions for frames [frame_a..frame_b].
    """
    T, J, _ = pts3d.shape
    fa, fb = max(0, min(frame_a, frame_b)), min(T - 1, max(frame_a, frame_b))
    n = fb - fa + 1
    result = pts3d[fa:fb + 1].copy()
    if n <= 2:
        return result

    # Build sorted unique knot frames within [fa, fb]
    kf_set = {fa, fb}
    if keyframes:
        for k in keyframes:
            if fa <= k <= fb:
                kf_set.add(k)
    knot_f = np.array(sorted(kf_set), dtype=float)

    target = np.arange(fa, fb + 1)

    for j in range(J):
        knot_v = np.array([pts3d[int(f), j] for f in knot_f])  # (K, 3)
        if method == "spline" and HAS_SCIPY and len(knot_f) >= 2:
            for d in range(3):
                cs = CubicSpline(knot_f, knot_v[:, d], bc_type='clamped')
                result[:, j, d] = cs(target)
        else:
            for d in range(3):
                result[:, j, d] = np.interp(target, knot_f, knot_v[:, d])
    return result


def apply_bulk_offset_all_joints(pts3d: np.ndarray,
                                 frame_start: int, frame_end: int,
                                 deltas: np.ndarray,
                                 taper: str = "none") -> np.ndarray:
    """Apply per-joint offsets to ALL joints across a frame range.

    Args:
        pts3d: (T, J, 3) — will NOT be modified in-place.
        frame_start, frame_end: Inclusive range.
        deltas: (J, 3) offset vector per joint.
        taper: "none", "linear", or "cosine".

    Returns:
        (N, J, 3) new positions for frames [frame_start..frame_end].
    """
    T, J, _ = pts3d.shape
    fs = max(0, frame_start)
    fe = min(T - 1, frame_end)
    n = fe - fs + 1

    original = pts3d[fs:fe + 1].copy()  # (N, J, 3)

    if taper == "none":
        weights = np.ones(n)
    elif taper == "linear":
        half = n / 2.0
        weights = np.minimum(np.arange(n), np.arange(n - 1, -1, -1)) / half
        weights = np.clip(weights, 0, 1)
    elif taper == "cosine":
        t = np.linspace(0, np.pi, n)
        weights = 0.5 * (1 - np.cos(t))
        if weights.max() > 0:
            weights /= weights.max()
    else:
        weights = np.ones(n)

    # weights: (N,) -> (N, 1, 1), deltas: (J, 3) -> (1, J, 3)
    return original + deltas[None, :, :] * weights[:, None, None]

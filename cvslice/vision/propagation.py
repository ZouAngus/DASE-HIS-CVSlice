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

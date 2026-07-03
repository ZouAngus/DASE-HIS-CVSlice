"""Two-bone limb IK for the Skeleton Corrector's IK drag mode.

Design contract (zero-ambiguity rules, mirrored in the README):

* Dragging an END EFFECTOR (wrist / ankle) solves the whole limb:
  - the root (shoulder / hip) never moves;
  - both bone lengths are locked to per-clip reference lengths (median over
    finite frames), so the solved pose is always physically possible;
  - if the drag target is beyond reach the limb goes perfectly STRAIGHT and
    the effector is clamped onto the reach sphere — a straight arm is NEVER
    folded (this replaces the old bone-length post-constraint that bent
    straight arms);
  - if the target is closer than |l1 - l2| it is clamped outward likewise;
  - the mid joint (elbow / knee) stays in the swivel plane defined by its
    current position (falling back to the previous frame's, then to an
    arbitrary perpendicular) so the solve follows the user's intent.

* Dragging a MID joint (elbow / knee) only slides it along the circle of
  positions allowed by the locked bone lengths (root and effector fixed) —
  i.e. it adjusts the swivel. If the limb is fully straight there is no
  circle and the drag is refused (the caller shows a message).

All functions are pure numpy — no Qt — so they are unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class LimbChain:
    """A two-bone chain root->mid->effector (+ optional rigid rider)."""
    name: str
    root: int
    mid: int
    eff: int
    rider: int | None = None   # hand/foot: follows the effector rigidly


# SMPL-24 layout (also valid for SMPL-22 = first 22 joints; riders 22/23
# are skipped automatically when J <= rider index).
_CHAINS_24 = [
    LimbChain("左臂", 16, 18, 20, 22),
    LimbChain("右臂", 17, 19, 21, 23),
    LimbChain("左腿", 1, 4, 7, 10),
    LimbChain("右腿", 2, 5, 8, 11),
]

# Human3.6M-style 17-joint layout.
_CHAINS_17 = [
    LimbChain("右腿", 1, 2, 3),
    LimbChain("左腿", 4, 5, 6),
    LimbChain("左臂", 11, 12, 13),
    LimbChain("右臂", 14, 15, 16),
]


def limb_chains(num_joints: int) -> list[LimbChain]:
    """Chains available for a skeleton with ``num_joints`` joints.

    Returns [] for topologies IK does not support (e.g. 37 raw markers) —
    callers should disable the IK checkbox in that case.
    """
    if num_joints in (22, 24):
        return _CHAINS_24
    if num_joints == 17:
        return _CHAINS_17
    return []


def chain_maps(num_joints: int) -> tuple[dict[int, LimbChain], dict[int, LimbChain]]:
    """(effector_joint -> chain, mid_joint -> chain) lookup dicts."""
    chains = limb_chains(num_joints)
    return ({c.eff: c for c in chains}, {c.mid: c for c in chains})


def reference_lengths(pts3d: np.ndarray, chain: LimbChain,
                      min_frames: int = 3) -> tuple[float, float] | None:
    """Per-clip locked bone lengths (l1 = root->mid, l2 = mid->eff).

    Median over all frames where the three joints are finite — robust to the
    very distortions being corrected. Falls back to None if fewer than
    ``min_frames`` usable frames exist (caller refuses the IK drag).
    """
    a = pts3d[:, chain.root]
    b = pts3d[:, chain.mid]
    c = pts3d[:, chain.eff]
    ok = (np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(c).all(1))
    if ok.sum() < min_frames:
        return None
    l1 = float(np.median(np.linalg.norm(b[ok] - a[ok], axis=1)))
    l2 = float(np.median(np.linalg.norm(c[ok] - b[ok], axis=1)))
    if l1 < _EPS or l2 < _EPS:
        return None
    return l1, l2


def _any_perpendicular(n: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to unit vector ``n`` (deterministic)."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ ref)) > 0.9:          # n nearly vertical -> use X axis
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    return u / (np.linalg.norm(u) + _EPS)


def _swivel_dir(n: np.ndarray, root: np.ndarray,
                hints: list[np.ndarray | None]) -> np.ndarray:
    """Unit direction ⟂ n selecting the swivel plane, from the first usable
    hint (a 3D point whose off-axis component is non-degenerate)."""
    for h in hints:
        if h is None or not np.all(np.isfinite(h)):
            continue
        v = h - root
        u = v - (v @ n) * n
        norm = float(np.linalg.norm(u))
        if norm > 1e-6:
            return u / norm
    return _any_perpendicular(n)


def solve_effector(root: np.ndarray, target: np.ndarray,
                   l1: float, l2: float,
                   hint_mid: np.ndarray | None,
                   prev_mid: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Two-bone IK: place mid + effector so the effector reaches ``target``.

    Returns (mid, effector, clamped) or None if the solve is degenerate
    (target coincides with root). ``clamped`` is True when the target was
    outside the reachable annulus and the effector was clamped onto it —
    in the far case the limb is perfectly straight (never folded).
    """
    root = np.asarray(root, float)
    target = np.asarray(target, float)
    d = target - root
    dist = float(np.linalg.norm(d))
    if dist < _EPS:
        return None
    n = d / dist

    reach = l1 + l2
    min_reach = abs(l1 - l2)
    dist_c = float(np.clip(dist, min_reach, reach))
    clamped = dist_c != dist
    eff = root + n * dist_c

    # Cosine rule along the root->target axis; r is the off-axis offset.
    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2.0 * dist_c)
    r = float(np.sqrt(max(l1 * l1 - a * a, 0.0)))
    if r < 1e-9:                     # straight (or fully folded): collinear
        mid = root + n * a
        return mid, eff, clamped

    u = _swivel_dir(n, root, [hint_mid, prev_mid])
    mid = root + a * n + r * u
    return mid, eff, clamped


def swivel_circle(root: np.ndarray, eff: np.ndarray,
                  l1: float, l2: float
                  ) -> tuple[np.ndarray, float, np.ndarray] | None:
    """The circle of valid mid-joint positions for fixed root & effector.

    Returns (center, radius, unit_axis) or None when no circle exists:
    root==eff, limb straight (|eff-root| >= l1+l2), or over-folded
    (|eff-root| <= |l1-l2|).
    """
    root = np.asarray(root, float)
    eff = np.asarray(eff, float)
    d = eff - root
    dist = float(np.linalg.norm(d))
    if dist < _EPS or dist >= l1 + l2 - 1e-9 or dist <= abs(l1 - l2) + 1e-9:
        return None
    n = d / dist
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
    r = float(np.sqrt(max(l1 * l1 - a * a, 0.0)))
    if r < 1e-9:
        return None
    return root + a * n, r, n


def solve_swivel(root: np.ndarray, eff: np.ndarray,
                 l1: float, l2: float,
                 drag_pt: np.ndarray) -> np.ndarray | None:
    """Slide the mid joint onto the valid circle, nearest to ``drag_pt``.

    Root and effector stay fixed; bone lengths stay locked. Returns the new
    mid position, or None when there is no circle (limb straight/folded —
    the caller should tell the user to drag the effector first).
    """
    circ = swivel_circle(root, eff, l1, l2)
    if circ is None:
        return None
    center, r, n = circ
    v = np.asarray(drag_pt, float) - center
    u = v - (v @ n) * n
    norm = float(np.linalg.norm(u))
    if norm < 1e-6:
        return None                  # drag point on the axis: direction unclear
    return center + r * (u / norm)


def sample_circle(center: np.ndarray, radius: float, axis: np.ndarray,
                  n_pts: int = 36) -> np.ndarray:
    """(n_pts, 3) points of the swivel circle, for drawing an overlay."""
    axis = axis / (np.linalg.norm(axis) + _EPS)
    u = _any_perpendicular(axis)
    v = np.cross(axis, u)
    ang = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    return (center[None, :]
            + radius * (np.cos(ang)[:, None] * u[None, :]
                        + np.sin(ang)[:, None] * v[None, :]))
